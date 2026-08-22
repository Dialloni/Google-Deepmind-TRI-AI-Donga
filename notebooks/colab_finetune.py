"""Self-contained Colab cell: fine-tune bge-base on the train qrels, export
embeddings for the local pipeline.

Paste into a Colab notebook after `kagglehub.login()`. Needs a GPU runtime
(Runtime -> Change runtime type -> T4). ~5-10 min total.

Steps:
1. Topic-grouped 80/20 holdout fine-tune -> prints honest nDCG@5 gain.
2. Re-fine-tune on ALL train queries.
3. Save train/test/doc embeddings to ft_embs.npz and download it.

Back on the laptop, drop the file into the repo root and run:
    python scripts/ltr.py eval    # picks up ft_embs.npz as a 5th model
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

BASE = "BAAI/bge-base-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR = 3, 32, 2e-5

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


def triplets(tq):
    """(query, positive rel>=2, judged negative rel==0) examples."""
    rng = np.random.default_rng(0)
    out = []
    for qid, g in qrels[qrels.query_id.isin(tq.query_id)].groupby("query_id"):
        query = PREFIX + tq.loc[tq.query_id == qid, "query"].iloc[0]
        pos = g[g.relevance >= 2].document_id.tolist()
        neg = g[g.relevance == 0].document_id.tolist()
        for p in pos:
            if neg:
                n = int(rng.choice(neg))
                out.append(InputExample(
                    texts=[query, corpus[doc_index[p]], corpus[doc_index[n]]]))
    return out


def finetune(examples):
    torch.manual_seed(0)
    model = SentenceTransformer(BASE)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=int(0.1 * len(loader) * EPOCHS),
              optimizer_params={"lr": LR}, show_progress_bar=True)
    return model


def score(model, queries):
    q = model.encode([PREFIX + x for x in queries],
                     normalize_embeddings=True, batch_size=64)
    d = model.encode(corpus, normalize_embeddings=True, batch_size=64)
    return q @ d.T, q, d


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

# --- 1. grouped holdout: fine-tune on 80% of topics, eval on the other 20% --
keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
topics = keys.map({k: i for i, k in enumerate(keys.unique())}).values
rng = np.random.default_rng(0)
held = rng.choice(np.unique(topics), size=len(np.unique(topics)) // 5,
                  replace=False)
va_mask = np.isin(topics, held)
tr, va = train_q[~va_mask], train_q[va_mask]

base = SentenceTransformer(BASE)
s_base, _, _ = score(base, va["query"].tolist())
print(f"holdout baseline  nDCG@5 = {ndcg5(s_base, va.query_id.tolist(), rels):.4f}")

model = finetune(triplets(tr))
s_ft, _, _ = score(model, va["query"].tolist())
print(f"holdout finetuned nDCG@5 = {ndcg5(s_ft, va.query_id.tolist(), rels):.4f}")

# --- 2. full fine-tune + 3. export ------------------------------------------
model = finetune(triplets(train_q))
_, q_tr, d_emb = score(model, train_q["query"].tolist())
_, q_te, _ = score(model, test_q["query"].tolist())
np.savez("ft_embs.npz", qtrain=q_tr, qtest=q_te, docs=d_emb)
print("saved ft_embs.npz")

from google.colab import files
files.download("ft_embs.npz")
