"""Sweep bi-encoders on the training queries and ensemble the best of them.

Each family wants its own prefix convention, so they are declared per model.
Score matrices are cached to /tmp so the ensemble step is free.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import doc_text, evaluate, load, rrf  # noqa: E402

BGE_Q = "Represent this sentence for searching relevant passages: "

# name -> (query_prefix, doc_prefix)
MODELS = {
    "BAAI/bge-base-en-v1.5": (BGE_Q, ""),
    "BAAI/bge-large-en-v1.5": (BGE_Q, ""),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "thenlper/gte-large": ("", ""),
}


def encode(model_name, queries, corpus):
    from sentence_transformers import SentenceTransformer

    q_pre, d_pre = MODELS[model_name]
    m = SentenceTransformer(model_name)
    q = m.encode(
        [q_pre + x for x in queries], normalize_embeddings=True, batch_size=64
    )
    d = m.encode(
        [d_pre + x for x in corpus], normalize_embeddings=True, batch_size=64
    )
    del m
    return q @ d.T


def main():
    docs, train_q, qrels, _ = load()
    corpus = doc_text(docs)
    doc_ids = docs.document_id.values
    queries = train_q["query"].tolist()
    qids = train_q.query_id.tolist()

    mats = {}
    for name in MODELS:
        cache = Path("/tmp") / f"sweep_{name.replace('/', '_')}.npy"
        mats[name] = np.load(cache) if cache.exists() else encode(name, queries, corpus)
        if not cache.exists():
            np.save(cache, mats[name])
        print(f"{name:28} {evaluate(mats[name], qids, doc_ids, qrels):.4f}", flush=True)

    print("\n--- ensembles ---", flush=True)
    names = list(mats)
    # Mean of cosine matrices: all are normalized, so scales already agree.
    print(
        f"{'mean(all)':28} "
        f"{evaluate(np.mean(list(mats.values()), axis=0), qids, doc_ids, qrels):.4f}",
        flush=True,
    )
    print(
        f"{'rrf(all)':28} "
        f"{evaluate(rrf(list(mats.values())), qids, doc_ids, qrels):.4f}",
        flush=True,
    )
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = np.mean([mats[names[i]], mats[names[j]]], axis=0)
            tag = f"{names[i].split('/')[-1]}+{names[j].split('/')[-1]}"
            print(
                f"{tag:28} {evaluate(pair, qids, doc_ids, qrels):.4f}", flush=True
            )


if __name__ == "__main__":
    main()
