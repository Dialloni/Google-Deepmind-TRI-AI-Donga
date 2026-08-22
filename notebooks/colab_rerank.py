"""Colab round 3: fine-tune a cross-encoder reranker on the graded qrels.

Paste into ONE cell of a FRESH Colab notebook (T4 GPU) after cell 1
(`pip install` + `kagglehub.login()`). ~15 min.

Why this instead of another bi-encoder: a cross-encoder reads the query and
the document together in one pass, so it can judge word-level agreement that
separate embeddings can never see. It is a different model class from
everything already in the ensemble, which is where the remaining headroom is.

An off-the-shelf reranker LOST 0.006 here early on. This one is trained on
our own 4195 graded judgements, which is a different proposition.

Exports ce_scores.npz (raw pair scores for every query x document) -> drop in
the repo root, then:
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
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder

COMP = "agricultural-extension-rag-smart-retrieval-for-farmers"
DATA = Path(kagglehub.competition_download(COMP))

BASE = "BAAI/bge-reranker-base"
EPOCHS, BATCH, LR = 2, 16, 2e-5
MAX_LEN = 320

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
qtext = dict(zip(train_q.query_id, train_q["query"]))


def pairs(tq):
    """Every judged (query, doc) row, relevance rescaled to 0..1."""
    sub = qrels[qrels.query_id.isin(tq.query_id)]
    return [
        InputExample(texts=[qtext[r.query_id], corpus[doc_index[r.document_id]]],
                     label=float(r.relevance) / 3.0)
        for r in sub.itertuples()
        if r.query_id in qtext and r.document_id in doc_index
    ]


def train_ce(examples, seed=0):
    torch.manual_seed(seed)
    ce = CrossEncoder(BASE, num_labels=1, max_length=MAX_LEN)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    ce.fit(train_dataloader=loader, epochs=EPOCHS,
           warmup_steps=int(0.1 * len(loader) * EPOCHS),
           optimizer_params={"lr": LR}, show_progress_bar=True)
    return ce


def score_all(ce, queries):
    """(n_queries, n_docs) matrix of cross-encoder scores."""
    out = np.zeros((len(queries), len(corpus)), dtype=np.float32)
    for i, q in enumerate(queries):
        out[i] = ce.predict([[q, d] for d in corpus], batch_size=128,
                            show_progress_bar=False)
        if i % 50 == 0:
            print(f"  scored query {i}/{len(queries)}", flush=True)
    return out


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

# --- 1. honest holdout check ------------------------------------------------
keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
topics = keys.map({k: i for i, k in enumerate(keys.unique())}).values
rng = np.random.default_rng(0)
held = rng.choice(np.unique(topics), size=len(np.unique(topics)) // 5,
                  replace=False)
va_mask = np.isin(topics, held)
tr, va = train_q[~va_mask], train_q[va_mask]

ce = train_ce(pairs(tr))
s_va = score_all(ce, va["query"].tolist())
print(f"holdout cross-encoder nDCG@5 = "
      f"{ndcg5(s_va, va.query_id.tolist(), rels):.4f}")
print("(bi-encoder holdouts: bge-base 0.8893, bge-large 0.9225)")
del ce
torch.cuda.empty_cache()

# --- 2. train on everything, export score matrices --------------------------
ce = train_ce(pairs(train_q))
print("scoring train queries...")
s_tr = score_all(ce, train_q["query"].tolist())
print("scoring test queries...")
s_te = score_all(ce, test_q["query"].tolist())

assert not np.isnan(s_tr).any() and not np.isnan(s_te).any(), \
    "NaN in cross-encoder scores -- training diverged, do not use"

np.savez("ce_scores.npz", train=s_tr, test=s_te)
print("saved ce_scores.npz")

from google.colab import files
files.download("ce_scores.npz")
