"""Self-contained Colab cell: download data, retrieve, write submission.

Paste into a Colab notebook after `kagglehub.login()`. Runs in ~1 min on a
T4 GPU (Runtime -> Change runtime type -> T4).
"""

# !pip install -q kagglehub sentence-transformers kaggle

import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

COMP = "agricultural-extension-rag-smart-retrieval-for-farmers"
DATA = Path(kagglehub.competition_download(COMP))

docs = pd.read_csv(DATA / "documents.csv")
test_q = pd.read_csv(DATA / "test_queries.csv")

# Title repeated twice so it carries more weight than a single mention in body.
corpus = [
    " ".join([r.title] * 2 + [r.text, str(r.crop), str(r.country)])
    for r in docs.itertuples()
]

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "

q_emb = model.encode(
    [PREFIX + q for q in test_q["query"]], normalize_embeddings=True, batch_size=64
)
d_emb = model.encode(corpus, normalize_embeddings=True, batch_size=64)
scores = q_emb @ d_emb.T

doc_ids = docs.document_id.values
rows = [
    (qid, int(d))
    for i, qid in enumerate(test_q.query_id)
    for d in doc_ids[np.argsort(-scores[i])[:5]]
]
sub = pd.DataFrame(rows, columns=["QueryId", "DocumentId"])
assert len(sub) == 1000, f"expected 1000 rows, got {len(sub)}"
sub.to_csv("submission.csv", index=False)
print(sub.head())

# --- Submit ---------------------------------------------------------------
# Put your Kaggle API token in Colab secrets (key icon, left sidebar) as
# KAGGLE_USERNAME and KAGGLE_KEY, then:
#
# import os
# from google.colab import userdata
# os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
# os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
# !kaggle competitions submit -c {COMP} -f submission.csv -m "bge-base dense"
