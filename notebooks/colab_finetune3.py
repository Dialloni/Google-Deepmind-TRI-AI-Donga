"""Colab round 3: fine-tune e5-large-v2 (different family from bge) with hard
negatives, weak positives, 2-seed average.

Paste into ONE cell of a FRESH Colab notebook (T4 GPU) after cell 1
(`pip install` + `kagglehub.login()`). ~35 min.

Changes vs round 2 (colab_finetune2.py):
- intfloat/e5-large-v2: a different pretraining family from bge, so its errors
  decorrelate from the two bge fine-tunes already in the ensemble. Raw e5 is
  the weakest single encoder we have (0.747), but dropping it from the blend
  made the blend worse, so its disagreement is worth something.
- rel==1 docs join training as weak positives (1 triplet each,
  vs 2 for rel>=2)

thenlper/gte-large was the original choice here and had to be abandoned: two
runs produced all-NaN embeddings, and the training loss sat at 0.77 where
bge-large reached 0.14, so it was never converging under this recipe.

Exports ft_embs3.npz -> drop in the repo root, then:
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

BASE = "intfloat/e5-large-v2"
PREFIX = "query: "       # e5 wants these exact markers on each side
DOC_PREFIX = "passage: "
EPOCHS, BATCH, LR = 3, 8, 1e-5
MAX_SEQ = 256


def load_base():
    """Force fp32 on load. Some encoders ship fp16 weights on the Hub, and
    training those with Adam and no loss scaling overflows into NaN.

    model_kwargs only exists on sentence-transformers 3+; .float() alone is
    enough on older releases.
    """
    try:
        m = SentenceTransformer(BASE,
                                model_kwargs={"torch_dtype": torch.float32})
    except TypeError:
        m = SentenceTransformer(BASE)
    m.float()
    m.max_seq_length = MAX_SEQ
    return m

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
    d = model.encode([DOC_PREFIX + x for x in corpus],
                     normalize_embeddings=True, batch_size=64)
    return q @ d.T, q, d


# --- hard negatives: base model's top-ranked docs that are NOT relevant -----
print("mining hard negatives with the base model...")
base = load_base()
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
    """Strong positives (rel>=2): 2 triplets each. Weak (rel==1): 1 each."""
    out = []
    for qid, g in qrels[qrels.query_id.isin(tq.query_id)].groupby("query_id"):
        query = PREFIX + tq.loc[tq.query_id == qid, "query"].iloc[0]
        negs = hard_negs.get(qid, [])
        if not negs:
            continue
        pool = ([(p, 2) for p in g[g.relevance >= 2].document_id]
                + [(p, 1) for p in g[g.relevance == 1].document_id])
        k = 0
        for p, copies in pool:
            for _ in range(copies):
                # Documents must carry DOC_PREFIX here exactly as they do at
                # inference, or the model trains on a format it never sees.
                out.append(InputExample(
                    texts=[query,
                           DOC_PREFIX + corpus[doc_index[p]],
                           DOC_PREFIX + corpus[doc_index[negs[k % len(negs)]]]]))
                k += 1
    return out


def finetune(examples, seed):
    torch.manual_seed(seed)
    model = load_base()
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    kw = dict(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=int(0.1 * len(loader) * EPOCHS),
              optimizer_params={"lr": LR}, show_progress_bar=True)
    try:
        model.fit(**kw, use_amp=False, max_grad_norm=1.0)
    except TypeError:
        model.fit(**kw)
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
del model
torch.cuda.empty_cache()

# Stop here rather than burn another 25 minutes on two more diverged seeds.
assert not np.isnan(s_ft).any(), (
    "holdout run produced NaN -- still diverging. Halve LR and rerun.")

print(f"holdout finetuned (gte-large) nDCG@5 = "
      f"{ndcg5(s_ft, va.query_id.tolist(), rels):.4f}")
print("(round-2 bge-large was 0.9225; round-1 bge-base 0.8893)")

# --- 2. full fine-tune, 2 seeds, averaged embeddings ------------------------
q_tr_list, q_te_list, d_list = [], [], []
for seed in (0, 1):
    m = finetune(triplets(train_q), seed=seed)
    _, q_tr, d_emb = score(m, train_q["query"].tolist())
    _, q_te, _ = score(m, test_q["query"].tolist())
    del m
    torch.cuda.empty_cache()
    # A diverged seed exports NaN; keep it out of the average rather than
    # letting one bad run poison the export.
    if any(np.isnan(a).any() for a in (q_tr, q_te, d_emb)):
        print(f"seed {seed} diverged (NaN) -- discarded", flush=True)
        continue
    q_tr_list.append(q_tr); q_te_list.append(q_te); d_list.append(d_emb)

assert q_tr_list, "every seed diverged -- lower LR and rerun"
print(f"{len(q_tr_list)}/2 seeds usable")


def avg_norm(mats):
    m = np.mean(mats, axis=0).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


np.savez("ft_embs3.npz", qtrain=avg_norm(q_tr_list),
         qtest=avg_norm(q_te_list), docs=avg_norm(d_list))
print("saved ft_embs3.npz")

from google.colab import files
files.download("ft_embs3.npz")
