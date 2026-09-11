"""Build the submission archive.

    python scripts/package.py              # code + README + embeddings + submission.csv
    python scripts/package.py --csv-only   # just submission.csv, for the Kaggle upload

The Kaggle leaderboard endpoint parses the upload as the predictions table, so
a whole-project archive fails there with "File appears to be empty"; --csv-only
produces an archive it accepts. The default archive is the write-up deliverable.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "donga_submission.zip"
CSV_OUT = ROOT / "donga_submission_csv.zip"

# Top-level entries that belong in the archive; everything else is left out.
INCLUDE = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "submission.csv",
    "src",
    "scripts",
    "notebooks",
]
INCLUDE_GLOBS = ["ft_embs*.npz"]

SKIP_DIRS = {"__pycache__", ".git", ".claude", ".claude-flow", "checkpoints",
             ".ipynb_checkpoints", "__MACOSX"}
SKIP_NAMES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

# Never ship credentials, whatever else the include list says.
SECRET_NAMES = {"kaggle.json", "access_token", ".env", "credentials.json",
                "client_secret.json", "id_rsa"}


def wanted(p: Path) -> bool:
    if p.name in SECRET_NAMES:
        raise SystemExit(f"refusing to package credential file: {p}")
    return not (
        p.name in SKIP_NAMES
        or p.suffix in SKIP_SUFFIXES
        or SKIP_DIRS & set(p.relative_to(ROOT).parts)
    )


def files_to_pack():
    entries = [ROOT / name for name in INCLUDE]
    entries += sorted(g for pat in INCLUDE_GLOBS for g in ROOT.glob(pat))
    for entry in entries:
        if not entry.exists():
            print(f"  (skipping missing {entry.name})")
            continue
        if entry.is_file():
            if wanted(entry):
                yield entry
        else:
            yield from (p for p in sorted(entry.rglob("*"))
                        if p.is_file() and wanted(p))


def write(path: Path, members, prefix: str = "donga") -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in members:
            rel = f.relative_to(ROOT)
            z.write(f, Path(prefix) / rel if prefix else rel)
    n = len(zipfile.ZipFile(path).namelist())
    print(f"wrote {path.name}  ({n} files, {path.stat().st_size / 1e6:.1f} MB)")


def main(argv):
    sub = ROOT / "submission.csv"
    if not sub.exists():
        sys.exit("submission.csv missing -- run `python scripts/ltr.py submit` first")

    # A malformed predictions file is the one error the leaderboard reports
    # unhelpfully, so check the shape before shipping either archive.
    rows = sub.read_text().strip().splitlines()
    header, body = rows[0], rows[1:]
    assert header == "QueryId,DocumentId", f"unexpected header: {header}"
    assert len(body) % 5 == 0, f"{len(body)} rows is not 5 per query"
    print(f"submission.csv OK: {len(body) // 5} queries x 5 rows")

    if "--csv-only" in argv:
        # Kaggle reads the archive as the predictions table, so the CSV has to
        # sit at the root -- a nested path is what makes it report an empty file.
        write(CSV_OUT, [sub], prefix="")
    else:
        write(OUT, list(files_to_pack()))


if __name__ == "__main__":
    main(sys.argv)
