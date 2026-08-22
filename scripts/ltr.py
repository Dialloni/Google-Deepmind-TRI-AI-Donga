"""Label-powered retrieval without gradient training (MPS-friendly).

Three stacked ideas, each trained in seconds on CPU:

1. Multi-encoder ensemble - four bi-encoders, embeddings cached to disk.
2. Linear adapter - closed-form ridge map from query embedding to the
   relevance-weighted mean of its positive doc embeddings (a "linear
   fine-tune" using the train qrels).
3. LambdaRank fusion - LightGBM ranker over per-(query,doc) features:
   raw + adapted cosines, BM25, crop/country match.

Validation is 5-fold CV grouped by topic family (queries sharing a positive
doc set), which is what the leaderboard actually tests: unseen topics.

    python scripts/ltr.py eval      # CV every stage
    python scripts/ltr.py submit    # fit on all train, write submission.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import bm25_scores, doc_text, evaluate, load, make_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/donga_emb")
BGE_Q = "Represent this sentence for searching relevant passages: "

# name -> (query_prefix, doc_prefix)
MODELS = {
    "BAAI/bge-base-en-v1.5": (BGE_Q, ""),
    "BAAI/bge-large-en-v1.5": (BGE_Q, ""),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "thenlper/gte-large": ("", ""),
}

TEMPLATES = [
    r"^how do i cope with ",
    r"^how can i adapt my farming to ",
    r"^how does ",
    r"^what is the risk of ",
    r"^what should i do about ",
    r"^what makes ",
]
SUFFIXES = r" (on my farm|affect my crops|to farming|my farm)$"


def topic(q: str) -> str:
    q = q.lower().rstrip("?").strip()
    for pat in TEMPLATES:
        q = re.sub(pat, "", q)
    return re.sub(SUFFIXES, "", q).strip()


# --------------------------------------------------------------------------- #
# Embeddings (cached)
# --------------------------------------------------------------------------- #

def embed(model_name: str, texts: list[str], prefix: str, tag: str) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{model_name.replace('/', '_')}_{tag}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    emb = m.encode([prefix + t for t in texts], normalize_embeddings=True,
                   batch_size=32, show_progress_bar=True)
    del m
    np.save(path, emb)
    return emb


def all_embeddings(train_queries, test_queries, corpus):
    """{model: (q_train, q_test, d)} with everything L2-normalized."""
    out = {}
    for name, (q_pre, d_pre) in MODELS.items():
        out[name] = (
            embed(name, train_queries, q_pre, "qtrain"),
            embed(name, test_queries, q_pre, "qtest"),
            embed(name, corpus, d_pre, "docs"),
        )
        print(f"embedded {name}", flush=True)
    for ft in sorted(ROOT.glob("ft_embs*.npz")):
        z = np.load(ft)
        mats = (z["qtrain"], z["qtest"], z["docs"])
        # A diverged fine-tune exports NaN; averaging it would poison every score.
        if any(np.isnan(m).any() for m in mats):
            print(f"SKIPPED {ft.name}: contains NaN", flush=True)
            continue
        out[ft.stem] = mats
        print(f"loaded {ft.name} (Colab fine-tuned model)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Feature matrix for the ranker
# --------------------------------------------------------------------------- #

def crop_country_match(queries, docs):
    """(n_q, n_docs) binary: doc crop / country mentioned in query text."""
    crops = docs.crop.str.lower().fillna("").values
    countries = docs.country.str.lower().fillna("").values
    qs = [q.lower() for q in queries]
    crop_m = np.array([[c != "(general)" and c in q for c in crops] for q in qs])
    ctry_m = np.array([[c in q for c in countries] for q in qs])
    return crop_m.astype(float), ctry_m.astype(float)


def feature_block(score_mats: list[np.ndarray], cand: np.ndarray):
    """Stack per-candidate features. cand: (n_q, k) doc indices.

    Raw cosines are not comparable across queries -- an easy query scores 0.9
    everywhere -- so each matrix also contributes a per-query z-score and the
    margin to that query's best document.
    """
    n_q, k = cand.shape
    feats = []
    for mat in score_mats:
        vals = np.take_along_axis(mat, cand, axis=1)
        ranks = (-mat).argsort(1).argsort(1)
        feats.append(vals)
        feats.append(np.take_along_axis(ranks, cand, axis=1))

        mu = mat.mean(1, keepdims=True)
        sd = mat.std(1, keepdims=True) + 1e-9
        feats.append((vals - mu) / sd)
        feats.append(vals - mat.max(1, keepdims=True))
    return np.stack(feats, axis=2).reshape(n_q * k, -1)


def ranker_fit_predict(tr_feats, tr_labels, tr_groups, va_feats):
    import lightgbm as lgb

    rk = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[5],
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, verbose=-1,
    )
    rk.fit(tr_feats, tr_labels, group=tr_groups)
    return rk.predict(va_feats)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

# Candidates per query fed to the ranker. The grouped CV rewards a deep list
# (50 -> 0.966, 350 -> 0.985) but the leaderboard does not: 350 scored 0.92686
# against 50's 0.92734. Only ~14 documents per query are judged, so every extra
# candidate is an assumed-irrelevant unjudged document, and a deep list mostly
# teaches the ranker to recognise the training pool. Trust the leaderboard here.
TOP_K = 50


def qrels_labels(qrels, query_ids, doc_ids, cand):
    rel = {(r.query_id, r.document_id): r.relevance for r in qrels.itertuples()}
    doc_ids = np.asarray(doc_ids)
    return np.array([
        rel.get((qid, int(doc_ids[di])), 0.0)
        for qid, row in zip(query_ids, cand)
        for di in row
    ])


def run_cv(emb, docs, train_q, qrels, bm25_tr, crop_tr, ctry_tr):
    from sklearn.model_selection import GroupKFold

    doc_ids = docs.document_id.values
    keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
    groups = keys.map({k: i for i, k in enumerate(keys.unique())}).values
    qids_all = train_q.query_id.tolist()

    raw = {name: q_tr @ d.T for name, (q_tr, _, d) in emb.items()}
    blend_raw = np.mean(list(raw.values()), axis=0)
    blend_topic = topic_average(blend_raw, train_q["query"].tolist())

    per_model = {n: [] for n in raw}
    results = {k: [] for k in ["blend_raw", "blend_topic", "lgbm", "lgbm_topic"]}
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(train_q, groups=groups):
        va_q = train_q.iloc[va_idx]
        va_qids = va_q.query_id.tolist()
        va_qrels = qrels[qrels.query_id.isin(va_qids)]
        ev = lambda mat: evaluate(mat[va_idx], va_qids, doc_ids, va_qrels)

        for n, mat in raw.items():
            per_model[n].append(ev(mat))
        results["blend_raw"].append(ev(blend_raw))
        results["blend_topic"].append(ev(blend_topic))

        # ranker on top of raw signals only
        mats = list(raw.values()) + [bm25_tr, crop_tr, ctry_tr]
        cand = np.argsort(-blend_raw, axis=1)[:, :TOP_K]
        feats = feature_block(mats, cand)
        labels = qrels_labels(qrels, qids_all, doc_ids, cand)
        n = len(train_q)
        f = feats.reshape(n, TOP_K, -1)
        l = labels.reshape(n, TOP_K)
        pred = ranker_fit_predict(
            f[tr_idx].reshape(-1, f.shape[2]), l[tr_idx].ravel(),
            [TOP_K] * len(tr_idx), f[va_idx].reshape(-1, f.shape[2]),
        ).reshape(len(va_idx), TOP_K)
        full = np.full((len(va_idx), len(doc_ids)), -1e9)
        np.put_along_axis(full, cand[va_idx], pred, axis=1)
        results["lgbm"].append(evaluate(full, va_qids, doc_ids, va_qrels))
        full_t = topic_average(full, va_q["query"].tolist())
        results["lgbm_topic"].append(evaluate(full_t, va_qids, doc_ids, va_qrels))

    for n, v in per_model.items():
        print(f"{n:28} {np.mean(v):.4f}", flush=True)
    for k, v in results.items():
        print(f"{k:16} {np.mean(v):.4f}  (folds: "
              + " ".join(f"{x:.3f}" for x in v) + ")", flush=True)


def topic_average(scores: np.ndarray, queries: list[str]) -> np.ndarray:
    """Average score rows across queries that share a topic phrase."""
    topics = [topic(q) for q in queries]
    out = scores.copy()
    df = pd.DataFrame({"t": topics})
    for _, idx in df.groupby("t").groups.items():
        idx = list(idx)
        if len(idx) > 1:
            out[idx] = scores[idx].mean(0)
    return out


def run_submit(emb, docs, train_q, test_q, qrels, bm25_all, crop_all, ctry_all,
               ranker: bool = True):
    """Fit the ranker on ALL train, predict test, topic-average.

    With ranker=False the plain encoder blend is written instead, which is how
    we measure on the leaderboard whether the ranker stage is worth anything --
    local CV cannot tell us, because the fine-tuned features are in-sample.
    """
    doc_ids = docs.document_id.values
    qids_all = train_q.query_id.tolist()

    raw_tr = {name: q_tr @ d.T for name, (q_tr, _, d) in emb.items()}
    raw_te = {name: q_te @ d.T for name, (_, q_te, d) in emb.items()}
    blend_tr = np.mean(list(raw_tr.values()), axis=0)
    blend_te = np.mean(list(raw_te.values()), axis=0)

    bm25_tr, bm25_te = bm25_all
    crop_tr, crop_te = crop_all
    ctry_tr, ctry_te = ctry_all

    mats_tr = list(raw_tr.values()) + [bm25_tr, crop_tr, ctry_tr]
    mats_te = list(raw_te.values()) + [bm25_te, crop_te, ctry_te]

    if ranker:
        cand_tr = np.argsort(-blend_tr, axis=1)[:, :TOP_K]
        cand_te = np.argsort(-blend_te, axis=1)[:, :TOP_K]
        f_tr = feature_block(mats_tr, cand_tr)
        l_tr = qrels_labels(qrels, qids_all, doc_ids, cand_tr)
        f_te = feature_block(mats_te, cand_te)

        pred = ranker_fit_predict(f_tr, l_tr, [TOP_K] * len(train_q), f_te)
        pred = pred.reshape(len(test_q), TOP_K)
        full = np.full((len(test_q), len(doc_ids)), -1e9)
        np.put_along_axis(full, cand_te, pred, axis=1)
    else:
        full = blend_te

    full = topic_average(full, test_q["query"].tolist())
    path = ROOT / "submission.csv"
    sub = make_submission(full, test_q.query_id.tolist(), doc_ids, path)
    assert len(sub) == 5 * len(test_q)
    print(f"Wrote {path} ({len(sub)} rows)")


def main(mode: str):
    docs, train_q, qrels, test_q = load()
    corpus = doc_text(docs)
    tr_queries = train_q["query"].tolist()
    te_queries = test_q["query"].tolist()

    emb = all_embeddings(tr_queries, te_queries, corpus)
    bm25_tr = bm25_scores(tr_queries, corpus)
    crop_tr, ctry_tr = crop_country_match(tr_queries, docs)

    if mode == "eval":
        run_cv(emb, docs, train_q, qrels, bm25_tr, crop_tr, ctry_tr)
    elif mode in ("submit", "submit-blend"):
        bm25_te = bm25_scores(te_queries, corpus)
        crop_te, ctry_te = crop_country_match(te_queries, docs)
        run_submit(emb, docs, train_q, test_q, qrels,
                   (bm25_tr, bm25_te), (crop_tr, crop_te), (ctry_tr, ctry_te),
                   ranker=(mode == "submit"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval")
