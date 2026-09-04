"""Stage-2-only rerun: train the cross-encoder on ALL train queries and
export ce2_scores.npz. Holdout already measured: 0.9169 -> 0.9578.

Original docstring follows.

Two teams jumped to ~0.968 on the public LB; a reranker fine-tuned on the
competition's own qrels is the standard method jump of that size. The earlier
negative result (README: "cross-encoder rerank hurt") used an OFF-THE-SHELF
reranker; this one is trained on this corpus's own relevance judgments with
mined hard negatives.

Stage 1 candidates come from the two fine-tuned bi-encoders already in the
repo (ft_embs.npz + ft_embs2.npz), so the clone provides everything.

Prints an honest topic-grouped holdout comparison (same split as rounds 2/4/5)
of blend-only vs blend+reranker, then scores the top-50 candidates of every
train and test query and exports ce2_scores.npz for the local pipeline.

~20 min on a T4.
"""

# !pip install -q kagglehub sentence-transformers kaggle
# !git clone -q https://github.com/Dialloni/Google-Deepmind-TRI-AI-Donga.git repo

import kagglehub
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

COMP = "agricultural-extension-rag-smart-retrieval-for-farmers"
DATA = Path(kagglehub.competition_download(COMP))

RERANKER = "BAAI/bge-reranker-v2-m3"
EPOCHS, BATCH, LR = 3, 16, 2e-5
MAX_LEN = 384
RERANK_K = 20   # candidates reranked per query in the holdout eval
EXPORT_K = 50   # candidates scored per query in the export
N_NEG = 3       # mined hard negatives per positive

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

# --- stage-1 candidates from the repo's fine-tuned bi-encoders ---------------
# NOTE: these encoders trained on all 308 train queries, so holdout candidate
# recall is slightly optimistic, but the reranker itself never sees holdout
# queries and recall@20 is near-perfect either way.
blend_tr, blend_te = None, None
for f in ("repo/ft_embs.npz", "repo/ft_embs2.npz"):
    z = np.load(f)
    s_tr = z["qtrain"] @ z["docs"].T
    s_te = z["qtest"] @ z["docs"].T
    blend_tr = s_tr if blend_tr is None else blend_tr + s_tr
    blend_te = s_te if blend_te is None else blend_te + s_te

rels = {qid: dict(zip(g.document_id, g.relevance))
        for qid, g in qrels.groupby("query_id")}


def ndcg5(order_docids, qid):
    rel = rels.get(qid, {})
    disc = 1 / np.log2(np.arange(2, 7))
    dcg = sum((2 ** rel.get(int(d), 0) - 1) * disc[j]
              for j, d in enumerate(order_docids[:5]))
    ideal = sorted(rel.values(), reverse=True)[:5]
    idcg = sum((2 ** g - 1) * disc[j] for j, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# --- training pairs: pos rel>=2 -> 1.0, mined hard negatives -> 0.0 ----------
# rel==1 docs are skipped: too ambiguous for a binary target.
def ce_pairs(tq):
    out = []
    for i in tq.index:  # positional: train_q has a clean RangeIndex
        qid = train_q.query_id[i]
        query = train_q["query"][i]
        rel = rels.get(qid, {})
        pos = [d for d, r in rel.items() if r >= 2]
        if not pos:
            continue
        ranked = doc_ids[np.argsort(-blend_tr[i])]
        negs = [int(d) for d in ranked if rel.get(int(d), 0) == 0][:N_NEG * len(pos)]
        for p in pos:
            out.append(InputExample(texts=[query, corpus[doc_index[p]]], label=1.0))
        for n in negs:
            out.append(InputExample(texts=[query, corpus[doc_index[n]]], label=0.0))
    return out


def train_ce(examples, seed=0):
    torch.manual_seed(seed)
    ce = CrossEncoder(RERANKER, num_labels=1, max_length=MAX_LEN)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    ce.fit(train_dataloader=loader, epochs=EPOCHS,
           warmup_steps=int(0.1 * len(loader) * EPOCHS),
           optimizer_params={"lr": LR}, show_progress_bar=True)
    return ce


# --- 2. retrain on ALL train, export candidate scores ------------------------
ce = train_ce(ce_pairs(train_q))


def score_block(queries, blend):
    cand = np.argsort(-blend, axis=1)[:, :EXPORT_K]
    pairs = [(q, corpus[c]) for i, q in enumerate(queries) for c in cand[i]]
    s = ce.predict(pairs, batch_size=64, show_progress_bar=True)
    return cand, s.reshape(len(queries), EXPORT_K)


cand_tr, s_tr = score_block(train_q["query"].tolist(), blend_tr)
cand_te, s_te = score_block(test_q["query"].tolist(), blend_te)
np.savez("ce2_scores.npz", cand_train=cand_tr, score_train=s_tr,
         cand_test=cand_te, score_test=s_te)
print("saved ce2_scores.npz")

from google.colab import files
files.download("ce2_scores.npz")
