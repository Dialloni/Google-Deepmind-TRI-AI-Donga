"""Download the Kaggle competition dataset.

Auth comes from ~/.kaggle/access_token or the KAGGLE_API_TOKEN env var.
Never hardcode the token here — this file may be committed.
"""

import kagglehub

COMPETITION = "agricultural-extension-rag-smart-retrieval-for-farmers"

if __name__ == "__main__":
    path = kagglehub.competition_download(COMPETITION)
    print("Path to competition files:", path)
