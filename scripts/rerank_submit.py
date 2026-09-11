"""Turn Colab's ce2_scores.npz into a submission.

    python scripts/rerank_submit.py             # pure rerank of top-20 (matches Colab holdout)
    python scripts/rerank_submit.py mix 0.7     # 0.7*z(ce) + 0.3*z(blend) over top-50
    python scripts/rerank_submit.py pure --topic # + topic averaging over template families

Below the reranked window the fine-tuned-blend order is kept, so ranks 6+ are
sane even though only the top 5 are scored.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import doc_text, load, make_submission  # noqa: E402
from scripts.ltr import topic_average  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RERANK_K = 20  # how many blend candidates the CE reorders (Colab holdout used 20)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "pure"
    alpha = float(argv[2]) if mode == "mix" and len(argv) > 2 else 0.7
    use_topic = "--topic" in argv

    docs, train_q, qrels, test_q = load()
    doc_ids = docs.document_id.values

    z = np.load(ROOT / "ce2_scores.npz")
    cand, ce = z["cand_test"], z["score_test"]  # (200, 50) doc indices + scores

    # same blend the candidates came from, for scores below the rerank window
    blend = None
    for f in ("ft_embs.npz", "ft_embs2.npz"):
        d = np.load(ROOT / f)
        s = d["qtest"] @ d["docs"].T
        blend = s if blend is None else blend + s

    full = blend / 1e3  # blend order, squashed under every reranked candidate
    n_q = len(test_q)
    if mode == "pure":
        # ranks 1..RERANK_K: CE order; below: blend order
        for i in range(n_q):
            top = cand[i, :RERANK_K]
            full[i, top] = 1.0 + (ce[i, :RERANK_K] - ce[i, :RERANK_K].min()) / (
                np.ptp(ce[i, :RERANK_K]) + 1e-9)
    elif mode == "mix":
        for i in range(n_q):
            c = cand[i]
            zc = (ce[i] - ce[i].mean()) / (ce[i].std() + 1e-9)
            zb = (blend[i, c] - blend[i, c].mean()) / (blend[i, c].std() + 1e-9)
            full[i, c] = 1.0 + alpha * zc + (1 - alpha) * zb
    else:
        sys.exit(__doc__)

    if use_topic:
        full = topic_average(full, test_q["query"].tolist())

    path = ROOT / "submission.csv"
    sub = make_submission(full, test_q.query_id.tolist(), doc_ids, path)
    assert len(sub) == 5 * len(test_q)
    print(f"Wrote {path} ({len(sub)} rows)  mode={mode}"
          + (f" alpha={alpha}" if mode == "mix" else "")
          + (" topic-averaged" if use_topic else ""))


if __name__ == "__main__":
    main(sys.argv)
