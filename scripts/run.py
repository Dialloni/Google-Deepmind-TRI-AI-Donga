"""Evaluate retrieval variants on the training queries, or build a submission.

    python scripts/run.py eval      # score every variant against train qrels
    python scripts/run.py submit    # write submission.csv from the best variant
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import (  # noqa: E402
    bm25_scores,
    dense_scores,
    doc_text,
    evaluate,
    load,
    make_submission,
    rerank,
    rrf,
    strip_boilerplate,
)

ROOT = Path(__file__).resolve().parents[1]


def build(queries, corpus, stages=("bm25", "dense", "rerank")):
    """Return {variant_name: score_matrix} for the requested stages."""
    stripped = [strip_boilerplate(q) for q in queries]
    out = {}

    if "bm25" in stages:
        out["bm25_raw"] = bm25_scores(queries, corpus)
        out["bm25_stripped"] = bm25_scores(stripped, corpus)

    if "dense" in stages:
        out["dense"] = dense_scores(queries, corpus)

    if "bm25" in stages and "dense" in stages:
        out["hybrid"] = rrf([out["bm25_stripped"], out["dense"]])

    if "rerank" in stages and "hybrid" in out:
        out["hybrid_rerank"] = rerank(queries, corpus, out["hybrid"])

    return out


def main(mode: str):
    docs, train_q, qrels, test_q = load()
    corpus = doc_text(docs)
    doc_ids = docs.document_id.values

    if mode == "eval":
        queries = train_q["query"].tolist()
        for name, scores in build(queries, corpus).items():
            score = evaluate(scores, train_q.query_id.tolist(), doc_ids, qrels)
            print(f"{name:16} nDCG@5 = {score:.4f}")
        print("\nTF-IDF baseline to beat: 0.5510 (public leaderboard)")

    elif mode == "submit":
        queries = test_q["query"].tolist()
        scores = build(queries, corpus)["hybrid_rerank"]
        path = ROOT / "submission.csv"
        sub = make_submission(scores, test_q.query_id.tolist(), doc_ids, path)
        assert len(sub) == 5 * len(test_q), f"expected 1000 rows, got {len(sub)}"
        print(f"Wrote {path} ({len(sub)} rows)")

    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval")
