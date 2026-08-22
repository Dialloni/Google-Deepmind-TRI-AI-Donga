"""Colab round 2: fine-tune bge-LARGE with mined hard negatives, 2-seed average.

Paste into a Colab cell after `kagglehub.login()` (T4 GPU runtime). ~15 min.

Upgrades over round 1 (colab_finetune.py):
- BAAI/bge-large-en-v1.5 (335M vs 109M)
- Hard negatives mined from the base model's own top-ranked mistakes,
  instead of random judged negatives
- Final embeddings averaged over 2 fine-tune seeds to cut variance

Prints an honest topic-grouped holdout score first, then exports
ft_embs2.npz. Drop it in the repo root next to ft_embs.npz and run:
    python scripts/ltr.py eval
    python scripts/ltr.py submit
"""

# !pip install -q kagglehub sentence-transformers kaggle

import kagglehub
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

COMP = "agricultural-extension-rag-smart-retrieval-for-farmers"
DATA = Path(kagglehub.competition_download(COMP))

BASE = "BAAI/bge-large-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR = 3, 8, 1e-5
MAX_SEQ = 256  # docs are short factsheets; capping halves attention memory
N_HARD = 2  # hard negatives per positive

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


def score(model, queries):
    q = model.encode([PREFIX + x for x in queries],
                     normalize_embeddings=True, batch_size=64)
    d = model.encode(corpus, normalize_embeddings=True, batch_size=64)
    return q @ d.T, q, d


# --- hard negatives: base model's top-ranked docs that are NOT relevant -----
print("mining hard negatives with the base model...")
base = SentenceTransformer(BASE)
s_all, _, _ = score(base, train_q["query"].tolist())
del base
torch.cuda.empty_cache()

judged_pos = {
    qid: set(g[g.relevance >= 1].document_id)
    for qid, g in qrels.groupby("query_id")
}
hard_negs = {}
for i, qid in enumerate(train_q.query_id):
    pos = judged_pos.get(qid, set())
    ranked = doc_ids[np.argsort(-s_all[i])]
    hard_negs[qid] = [int(d) for d in ranked if int(d) not in pos][:10]


def triplets(tq):
    """(query, positive rel>=2, mined hard negative) examples."""
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


def finetune(examples, seed):
    torch.manual_seed(seed)
    model = SentenceTransformer(BASE)
    model.max_seq_length = MAX_SEQ
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
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

# --- 1. honest holdout check (1 seed) ---------------------------------------
keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
topics = keys.map({k: i for i, k in enumerate(keys.unique())}).values
rng = np.random.default_rng(0)
held = rng.choice(np.unique(topics), size=len(np.unique(topics)) // 5,
                  replace=False)
va_mask = np.isin(topics, held)
tr, va = train_q[~va_mask], train_q[va_mask]

model = finetune(triplets(tr), seed=0)
s_ft, _, _ = score(model, va["query"].tolist())
print(f"holdout finetuned (large+hard) nDCG@5 = "
      f"{ndcg5(s_ft, va.query_id.tolist(), rels):.4f}")
print("(round-1 numbers were: baseline 0.8191, finetuned 0.8893)")
del model
torch.cuda.empty_cache()

# --- 2. full fine-tune, 2 seeds, averaged embeddings ------------------------
q_tr_list, q_te_list, d_list = [], [], []
for seed in (0, 1):
    m = finetune(triplets(train_q), seed=seed)
    _, q_tr, d_emb = score(m, train_q["query"].tolist())
    _, q_te, _ = score(m, test_q["query"].tolist())
    q_tr_list.append(q_tr); q_te_list.append(q_te); d_list.append(d_emb)
    del m
    torch.cuda.empty_cache()


def avg_norm(mats):
    m = np.mean(mats, axis=0)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


np.savez("ft_embs2.npz", qtrain=avg_norm(q_tr_list),
         qtest=avg_norm(q_te_list), docs=avg_norm(d_list))
print("saved ft_embs2.npz")

from google.colab import files
files.download("ft_embs2.npz")
