"""Retrieval pipeline for the agricultural-extension RAG competition.

Stages: BM25 (lexical) + dense bi-encoder, fused with reciprocal rank fusion,
then an optional cross-encoder rerank of the fused top-k.

Scored with nDCG@5, the leaderboard metric.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path.home() / (
    ".cache/kagglehub/competitions/"
    "agricultural-extension-rag-smart-retrieval-for-farmers"
)

# Query templates repeat across the whole query set, so their wording carries no
# signal about which document is relevant -- only the topic phrase does.
BOILERPLATE = re.compile(
    r"\b(how do i|how can i|how does|what is the|what should i do about|"
    r"what makes|on my farm|to farming|my farm|i |my )\b",
    re.IGNORECASE,
)


def load():
    """Return (documents, train_queries, qrels, test_queries)."""
    read = lambda name: pd.read_csv(DATA_DIR / f"{name}.csv")
    return (
        read("documents"),
        read("train_queries"),
        read("qrels_train"),
        read("test_queries"),
    )


def strip_boilerplate(q: str) -> str:
    return re.sub(r"\s+", " ", BOILERPLATE.sub(" ", q)).strip(" ?")


def doc_text(docs: pd.DataFrame, title_weight: int = 2) -> list[str]:
    """Title repeated `title_weight` times so it carries more lexical mass."""
    return [
        " ".join([r.title] * title_weight + [r.text, str(r.crop), str(r.country)])
        for r in docs.itertuples()
    ]


# --------------------------------------------------------------------------- #
# Retrievers -- each returns a (n_queries, n_docs) score matrix
# --------------------------------------------------------------------------- #

def bm25_scores(queries: list[str], corpus: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    bm25 = BM25Okapi([tok(d) for d in corpus])
    return np.vstack([bm25.get_scores(tok(q)) for q in queries])


@lru_cache(maxsize=4)
def _encoder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def dense_scores(
    queries: list[str],
    corpus: list[str],
    model_name: str = "BAAI/bge-base-en-v1.5",
) -> np.ndarray:
    model = _encoder(model_name)
    # bge expects this instruction on the query side only.
    prefix = "Represent this sentence for searching relevant passages: "
    q_emb = model.encode(
        [prefix + q for q in queries], normalize_embeddings=True, batch_size=64
    )
    d_emb = model.encode(corpus, normalize_embeddings=True, batch_size=64)
    return q_emb @ d_emb.T


def rrf(score_mats: list[np.ndarray], weights=None, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion. Rank-based, so the scales never need calibrating."""
    weights = weights or [1.0] * len(score_mats)
    fused = np.zeros_like(score_mats[0], dtype=float)
    for mat, w in zip(score_mats, weights):
        # ranks: 0 = best scoring document for that query
        ranks = (-mat).argsort(axis=1).argsort(axis=1)
        fused += w / (k + ranks + 1)
    return fused


def rerank(
    queries: list[str],
    corpus: list[str],
    fused: np.ndarray,
    top_k: int = 30,
    model_name: str = "BAAI/bge-reranker-base",
) -> np.ndarray:
    """Cross-encode the fused top_k per query; leave the tail ordered as-is."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name)
    out = fused.copy()
    cand = np.argsort(-fused, axis=1)[:, :top_k]

    pairs, index = [], []
    for qi, docs in enumerate(cand):
        for di in docs:
            pairs.append((queries[qi], corpus[di]))
            index.append((qi, di))
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=True)

    # Lift reranked docs above every untouched document.
    floor = out.max() + 1.0
    for (qi, di), s in zip(index, scores):
        out[qi, di] = floor + s
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def ndcg_at_k(ranked_ids: list[int], rel: dict[int, float], k: int = 5) -> float:
    gains = [(2 ** rel.get(d, 0.0) - 1) for d in ranked_ids[:k]]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts[: len(gains)]))

    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = float(np.sum([(2 ** g - 1) for g in ideal] * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(scores: np.ndarray, query_ids, doc_ids, qrels: pd.DataFrame) -> float:
    """Mean nDCG@5. Unjudged documents count as relevance 0, per TREC."""
    by_query = {
        qid: dict(zip(g.document_id, g.relevance))
        for qid, g in qrels.groupby("query_id")
    }
    doc_ids = np.asarray(doc_ids)
    per_query = [
        ndcg_at_k(doc_ids[np.argsort(-scores[i])[:5]].tolist(), by_query.get(qid, {}))
        for i, qid in enumerate(query_ids)
    ]
    return float(np.mean(per_query))


def make_submission(scores, query_ids, doc_ids, path: Path) -> pd.DataFrame:
    """Long format: 5 rows per query, row order within a query is the rank."""
    doc_ids = np.asarray(doc_ids)
    rows = [
        (qid, int(d))
        for i, qid in enumerate(query_ids)
        for d in doc_ids[np.argsort(-scores[i])[:5]]
    ]
    sub = pd.DataFrame(rows, columns=["QueryId", "DocumentId"])
    sub.to_csv(path, index=False)
    return sub
