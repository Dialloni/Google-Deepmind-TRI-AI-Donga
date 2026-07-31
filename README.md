# Donga — Agricultural Extension RAG (Kaggle)

Retrieval for [Agricultural Extension RAG: Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers).

Given a smallholder farmer's question, rank the 5 most relevant extension
documents from a 695-doc corpus. Metric: **nDCG@5**.

## Status

Rank 1, public LB **0.82640** (baseline to beat: 0.551).

## Results (train nDCG@5, 308 queries)

| Variant | Score |
|---|---|
| TF-IDF baseline (leaderboard) | 0.551 |
| BM25 raw | 0.513 |
| BM25 + boilerplate stripped | 0.502 |
| RRF hybrid, equal weight | 0.681 |
| RRF hybrid, bm25 weight 0.1 | 0.817 |
| **dense, bge-base-en-v1.5** | **0.820** |
| dense + cross-encoder rerank@10 | 0.814 |

Findings that shaped the pipeline:

- **Lexical retrieval is dead weight here.** Every RRF weight on BM25 lowered
  the score monotonically, so BM25 is out of the submission path.
- **Stripping query boilerplate hurt.** Queries come in template families
  ("How do I cope with X on my farm?"), and removing the template cost ~0.011
  on BM25 rather than helping.
- **Cross-encoder rerank hurt.** `bge-reranker-base` lost 0.006 against the
  bi-encoder alone; the documents are short factsheets the bi-encoder already
  separates cleanly.
- Train tracks the leaderboard within ~0.007, so the local harness is a
  trustworthy proxy. Public LB is only 53% of test — tune against train, which
  is the larger sample.

## Usage

```bash
python scripts/run.py eval      # score variants against train qrels
python scripts/run.py submit    # write submission.csv
python scripts/sweep.py         # compare bi-encoders and their ensembles

kaggle competitions submit \
  -c agricultural-extension-rag-smart-retrieval-for-farmers \
  -f submission.csv -m "message"
```

Colab equivalent: [notebooks/colab_submission.py](notebooks/colab_submission.py).

## Auth

Kaggle credentials live in `~/.kaggle/` (outside this repo) or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars. Never hardcode a token in a file
here.

## Layout

    src/pipeline.py             retrievers, fusion, nDCG@5, submission writer
    scripts/run.py              eval / submit entrypoint
    scripts/sweep.py            encoder + ensemble comparison
    scripts/download_data.py    fetch competition files via kagglehub
    notebooks/colab_submission.py
