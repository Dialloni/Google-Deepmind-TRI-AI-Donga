"""Colab round 5: synthetic-pretrain curriculum (round 4 fixed).

Round 4 mixed 2x more synthetic than real triplets into the same batches and
holdout DROPPED to 0.9165 (round 2: 0.9225): synthetic queries come in many
near-duplicates per topic, and with in-batch negatives the model learns to
separate documents that are both relevant. Round 5 stages them instead:

  stage 1: 1 epoch on synthetic triplets only (vocabulary exposure for
           test-only topics like iron/magnesium/sulphur deficiency)
  stage 2: the exact round-2 recipe on real triplets (proven 0.9225),
           applied last so the real signal dominates.

Paste into a Colab cell after `kagglehub.login()` (T4 GPU runtime). ~25 min.

Prints the honest topic-grouped holdout first — same split and seed as
rounds 2 and 4 (0.9225 / 0.9165) — then exports ft_embs5.npz. Drop it in
the repo root and run:
    python scripts/ltr.py eval
    python scripts/ltr.py submit
"""

# !pip install -q kagglehub sentence-transformers kaggle
# !git clone -q https://github.com/Dialloni/Google-Deepmind-TRI-AI-Donga.git repo

import sys

import kagglehub
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

sys.path.insert(0, "repo")
from scripts.synth import build_synthetic  # noqa: E402

COMP = "agricultural-extension-rag-smart-retrieval-for-farmers"
DATA = Path(kagglehub.competition_download(COMP))

BASE = "BAAI/bge-large-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR = 3, 8, 1e-5
MAX_SEQ = 256
N_HARD = 2  # hard negatives per positive, real and synthetic alike

docs = pd.read_csv(DATA / "documents.csv")
train_q = pd.read_csv(DATA / "train_queries.csv")
qrels = pd.read_csv(DATA / "qrels_train.csv")
test_q = pd.read_csv(DATA / "test_queries.csv")

corpus = [
    " ".join([r.title] * 2 + [r.text, str(r.crop), str(r.country)])
    for r in docs.itertuples()
]
doc_ids = docs.document_id.values
doc_index = {d: i for i, d in enumerate(doc_ids)}
titles_lc = docs.title.str.lower().tolist()

# --- synthetic pairs ---------------------------------------------------------
synth = build_synthetic(docs)
# The generator reproduces the competition's own process, so some outputs
# collide verbatim with real train queries; the real qrels already cover those.
synth = synth[~synth["query"].isin(set(train_q["query"]))].reset_index(drop=True)
print(f"{len(synth)} synthetic pairs after dropping verbatim train collisions")


def score(model, queries):
    q = model.encode([PREFIX + x for x in queries],
                     normalize_embeddings=True, batch_size=64)
    d = model.encode(corpus, normalize_embeddings=True, batch_size=64)
    return q @ d.T, q, d


# --- hard negatives from the base model --------------------------------------
print("mining hard negatives with the base model...")
base = SentenceTransformer(BASE)
s_real, _, _ = score(base, train_q["query"].tolist())
s_syn, _, _ = score(base, synth["query"].tolist())
del base
torch.cuda.empty_cache()

judged_pos = {
    qid: set(g[g.relevance >= 1].document_id)
    for qid, g in qrels.groupby("query_id")
}
hard_negs = {}
for i, qid in enumerate(train_q.query_id):
    pos = judged_pos.get(qid, set())
    ranked = doc_ids[np.argsort(-s_real[i])]
    hard_negs[qid] = [int(d) for d in ranked if int(d) not in pos][:10]


# Docs sharing the source doc's topic phrase must not become negatives: many
# titles repeat one topic across agro-zones AND across families ('Preventing /
# Managing / Symptoms of X in Maize'), and all of those are relevant to a
# synthetic query about X. synth.topic is that phrase.
syn_negs = []
for i, r in enumerate(synth.itertuples()):
    ranked = doc_ids[np.argsort(-s_syn[i])]
    negs = [int(d) for d in ranked
            if int(d) != r.document_id
            and r.topic not in titles_lc[doc_index[int(d)]]]
    syn_negs.append(negs[:10])


# --- triplets ----------------------------------------------------------------
def real_triplets(tq):
    out = []
    for qid, g in qrels[qrels.query_id.isin(tq.query_id)].groupby("query_id"):
        query = PREFIX + tq.loc[tq.query_id == qid, "query"].iloc[0]
        pos = g[g.relevance >= 2].document_id.tolist()
        negs = hard_negs.get(qid, [])
        if not negs:
            continue
        for j, p in enumerate(pos):
            for h in range(N_HARD):
                n = negs[(j * N_HARD + h) % len(negs)]
                out.append(InputExample(
                    texts=[query, corpus[doc_index[p]], corpus[doc_index[n]]]))
    return out


def synth_triplets(exclude_docs: set):
    """One triplet per (synthetic query, hard neg); skip leakage into holdout."""
    out = []
    for i, r in enumerate(synth.itertuples()):
        if r.document_id in exclude_docs:
            continue
        negs = syn_negs[i]
        for h in range(N_HARD):
            if h >= len(negs):
                break
            out.append(InputExample(
                texts=[PREFIX + r.query,
                       corpus[doc_index[r.document_id]],
                       corpus[doc_index[negs[h]]]]))
    return out


def finetune(synth_ex, real_ex, seed):
    """Stage 1: one epoch on synthetic only. Stage 2: round-2 recipe on real."""
    torch.manual_seed(seed)
    model = SentenceTransformer(BASE)
    model.max_seq_length = MAX_SEQ
    loss = losses.MultipleNegativesRankingLoss(model)
    if synth_ex:
        loader = DataLoader(synth_ex, shuffle=True, batch_size=BATCH)
        model.fit(train_objectives=[(loader, loss)], epochs=1,
                  warmup_steps=int(0.1 * len(loader)),
                  optimizer_params={"lr": LR}, show_progress_bar=True)
    loader = DataLoader(real_ex, shuffle=True, batch_size=BATCH)
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=int(0.1 * len(loader) * EPOCHS),
              optimizer_params={"lr": LR}, show_progress_bar=True)
    return model


def ndcg5(scores, qids, rels):
    disc = 1 / np.log2(np.arange(2, 7))
    total = 0.0
    for i, qid in enumerate(qids):
        rel = rels.get(qid, {})
        top = doc_ids[np.argsort(-scores[i])[:5]]
        dcg = sum((2 ** rel.get(int(d), 0) - 1) * disc[j] for j, d in enumerate(top))
        ideal = sorted(rel.values(), reverse=True)[:5]
        idcg = sum((2 ** g - 1) * disc[j] for j, g in enumerate(ideal))
        total += dcg / idcg if idcg else 0.0
    return total / len(qids)


rels = {qid: dict(zip(g.document_id, g.relevance))
        for qid, g in qrels.groupby("query_id")}

# --- 1. honest holdout (same split + seed as round 2: compare to 0.9225) -----
keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
topics = keys.map({k: i for i, k in enumerate(keys.unique())}).values
rng = np.random.default_rng(0)
held = rng.choice(np.unique(topics), size=len(np.unique(topics)) // 5,
                  replace=False)
va_mask = np.isin(topics, held)
tr, va = train_q[~va_mask], train_q[va_mask]

# Leakage guard: synthetic queries generated FROM any doc judged relevant to a
# holdout query would leak those topics into training. Drop them.
holdout_docs = set(
    qrels[qrels.query_id.isin(va.query_id) & (qrels.relevance >= 1)].document_id
)
real_tr, syn_tr = real_triplets(tr), synth_triplets(holdout_docs)
print(f"holdout run: stage 1 = {len(syn_tr)} synthetic, "
      f"stage 2 = {len(real_tr)} real ({len(holdout_docs)} docs quarantined)")

model = finetune(syn_tr, real_tr, seed=0)
s_ft, _, _ = score(model, va["query"].tolist())
print(f"holdout nDCG@5, curriculum = "
      f"{ndcg5(s_ft, va.query_id.tolist(), rels):.4f}")
print("(round 2 real-only: 0.9225 | round 4 mixed: 0.9165)")
del model
torch.cuda.empty_cache()

# --- 2. full fine-tune, 2 seeds, averaged embeddings -------------------------
real_full, syn_full = real_triplets(train_q), synth_triplets(set())
q_tr_list, q_te_list, d_list = [], [], []
for seed in (0, 1):
    m = finetune(syn_full, real_full, seed=seed)
    _, q_tr, d_emb = score(m, train_q["query"].tolist())
    _, q_te, _ = score(m, test_q["query"].tolist())
    q_tr_list.append(q_tr); q_te_list.append(q_te); d_list.append(d_emb)
    del m
    torch.cuda.empty_cache()


def avg_norm(mats):
    m = np.mean(mats, axis=0)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


np.savez("ft_embs5.npz", qtrain=avg_norm(q_tr_list),
         qtest=avg_norm(q_te_list), docs=avg_norm(d_list))
print("saved ft_embs5.npz")

from google.colab import files
files.download("ft_embs5.npz")
