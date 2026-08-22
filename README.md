# Donga — Agricultural Extension RAG (Kaggle)

Retrieval for [Agricultural Extension RAG: Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers).

Given a smallholder farmer's question, rank the 5 most relevant extension
documents from a 695-doc corpus. Metric: **nDCG@5**.

## Status

Rank 2, public LB **0.92734** (leader 0.94076, TF-IDF baseline 0.551).

## Leaderboard history

| Submission | Public LB |
|---|---|
| bge-base dense retrieval | 0.82640 |
| 4-encoder blend + LightGBM LambdaRank | 0.86954 |
| + bge-base fine-tuned on the qrels (Colab) | 0.91880 |
| **+ bge-large fine-tuned, hard negatives, 2 seeds** | **0.92734** |
| same, candidate list deepened to 350 | 0.92686 |

The fine-tunes are measured honestly in Colab on a topic-grouped holdout, where
the validation topics are excluded from training:

| Model | Holdout nDCG@5 |
|---|---|
| bge-base, off the shelf | 0.8191 |
| bge-base, fine-tuned, random negatives | 0.8893 |
| bge-large, fine-tuned, mined hard negatives | 0.9225 |

## Findings

- **Fine-tuning on the qrels is the whole game.** Every leaderboard jump came
  from a better fine-tuned encoder; feature work on top of them has been worth
  roughly nothing.
- **The local CV is inflated and cannot referee tuning decisions.** Both
  fine-tuned encoders trained on all 308 train queries, so in cross-validation
  the held-out queries were already seen by the encoder. Deepening the
  candidate list from 50 to 350 gained 0.019 in CV and *lost* 0.0005 on the
  leaderboard, because a deeper list makes the ranker lean harder on exactly
  those in-sample features. Trustworthy CV would need out-of-fold fine-tunes,
  one per fold.
- Not pooling bias: at K=350 the ranker's top-5 are *less* likely to be judged
  documents (0.856) than the plain blend's (0.870), so it is not learning to
  recognise the training pool.
- **gte-large diverges to NaN on Colab.** Its Hub weights are fp16 and
  sentence-transformers honours that dtype, so Adam without loss scaling
  overflows. Force `torch_dtype=torch.float32`.
- **Lexical retrieval is dead weight here.** Every RRF weight on BM25 lowered
  the score monotonically, so BM25 is out of the submission path.
- **Stripping query boilerplate hurt.** Queries come in template families
  ("How do I cope with X on my farm?"), and removing the template cost ~0.011
  on BM25 rather than helping.
- A ridge "linear adapter" mapping queries onto their positive centroid scored
  0.69 in CV -- it memorises train topics and does not transfer.
- Train and test topics are disjoint (0 of 305 overlap), so nothing that keys
  on a specific crop or disease will generalise.

## Usage

```bash
python scripts/ltr.py eval          # grouped CV over every stage (see caveat above)
python scripts/ltr.py submit        # write submission.csv from the full pipeline
python scripts/ltr.py submit-blend  # ablation: encoder blend, no ranker

python scripts/run.py eval          # the older single-encoder variants
python scripts/sweep.py             # compare bi-encoders and their ensembles

kaggle competitions submit \
  -c agricultural-extension-rag-smart-retrieval-for-farmers \
  -f submission.csv -m "message"
```

Encoder embeddings cache to `/tmp/donga_emb`, so only the first run pays for
them. Any `ft_embs*.npz` in the repo root is picked up as an extra encoder;
files containing NaN are skipped with a warning.

The fine-tunes need a GPU, so they run in Colab (T4, paste one file per cell
after `kagglehub.login()`) and export embeddings back into the repo root:

| Notebook | Exports | Holdout |
|---|---|---|
| [colab_finetune.py](notebooks/colab_finetune.py) | `ft_embs.npz` | 0.8893 |
| [colab_finetune2.py](notebooks/colab_finetune2.py) | `ft_embs2.npz` | 0.9225 |
| [colab_finetune3.py](notebooks/colab_finetune3.py) | `ft_embs3.npz` | gte-large, fp32-forced |
| [colab_rerank.py](notebooks/colab_rerank.py) | `ce_scores.npz` | cross-encoder, untried |

## Auth

Kaggle credentials live in `~/.kaggle/` (outside this repo) or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars. Never hardcode a token in a file
here.

## Layout

    src/pipeline.py             retrievers, fusion, nDCG@5, submission writer
    scripts/ltr.py              current pipeline: ensemble + LambdaRank
    scripts/run.py              eval / submit entrypoint
    scripts/sweep.py            encoder + ensemble comparison
    scripts/download_data.py    fetch competition files via kagglehub
    notebooks/colab_*.py        GPU fine-tunes, one file per Colab cell
    notebooks/colab_submission.py
