"""Fine-tune the bi-encoder on the train qrels, with topic-grouped CV.

Queries come in template families that share the same positive documents, so
folds must be grouped by positive-doc-set (topic proxy) or the CV leaks.

    python scripts/finetune.py cv      # 5-fold grouped CV: baseline vs fine-tuned
    python scripts/finetune.py full    # fine-tune on all train, save model
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import doc_text, evaluate, load  # noqa: E402

BASE_MODEL = "BAAI/bge-base-en-v1.5"
Q_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_OUT = Path(__file__).resolve().parents[1] / "models" / "bge-base-ft"

EPOCHS = 3
BATCH = 16
LR = 2e-5


def topic_groups(train_q: pd.DataFrame) -> np.ndarray:
    """Group id per query: queries sharing a positive-doc set are one topic."""
    keys = train_q.positive_docs.map(lambda s: frozenset(s.split()))
    return keys.map({k: i for i, k in enumerate(keys.unique())}).values


def triplets(train_q, qrels, corpus, doc_ids):
    """(query, positive, hard-negative) texts. Positives rel>=2, negs rel==0."""
    from sentence_transformers import InputExample

    idx = {d: i for i, d in enumerate(doc_ids)}
    rng = np.random.default_rng(0)
    examples = []
    for qid, g in qrels.groupby("query_id"):
        row = train_q[train_q.query_id == qid]
        if row.empty:
            continue
        query = Q_PREFIX + row["query"].iloc[0]
        pos = g[g.relevance >= 2].document_id.tolist()
        neg = g[g.relevance == 0].document_id.tolist()
        if not pos or not neg:
            continue
        for p in pos:
            n = rng.choice(neg)
            examples.append(
                InputExample(texts=[query, corpus[idx[p]], corpus[idx[int(n)]]])
            )
    return examples


def finetune(examples, seed=0):
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    model = SentenceTransformer(BASE_MODEL)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(loader) * EPOCHS),
        optimizer_params={"lr": LR},
        show_progress_bar=True,
    )
    return model


def score(model, queries, corpus):
    q = model.encode(
        [Q_PREFIX + x for x in queries], normalize_embeddings=True, batch_size=64
    )
    d = model.encode(corpus, normalize_embeddings=True, batch_size=64)
    return q @ d.T


def cv():
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import GroupKFold

    docs, train_q, qrels, _ = load()
    corpus = doc_text(docs)
    doc_ids = docs.document_id.values
    groups = topic_groups(train_q)

    base = SentenceTransformer(BASE_MODEL)
    base_scores, ft_scores = [], []
    for fold, (tr_idx, va_idx) in enumerate(
        GroupKFold(n_splits=5).split(train_q, groups=groups)
    ):
        tr_q = train_q.iloc[tr_idx]
        va_q = train_q.iloc[va_idx]
        va_queries = va_q["query"].tolist()
        va_qids = va_q.query_id.tolist()
        va_qrels = qrels[qrels.query_id.isin(va_qids)]

        b = evaluate(score(base, va_queries, corpus), va_qids, doc_ids, va_qrels)

        ex = triplets(tr_q, qrels[qrels.query_id.isin(tr_q.query_id)], corpus, doc_ids)
        model = finetune(ex)
        f = evaluate(score(model, va_queries, corpus), va_qids, doc_ids, va_qrels)

        print(f"fold {fold}: baseline {b:.4f}  finetuned {f:.4f}", flush=True)
        base_scores.append(b)
        ft_scores.append(f)

    print(f"\nmean: baseline {np.mean(base_scores):.4f}  "
          f"finetuned {np.mean(ft_scores):.4f}")


def full():
    docs, train_q, qrels, _ = load()
    corpus = doc_text(docs)
    ex = triplets(train_q, qrels, corpus, docs.document_id.values)
    print(f"{len(ex)} triplets", flush=True)
    model = finetune(ex)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_OUT))
    print(f"saved to {MODEL_OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    cv() if mode == "cv" else full()
