"""Synthetic query generation for fine-tuning (mentor suggestion).

Both the competition queries and the document titles are templated, so we can
run the competition's own generation process in reverse: parse the topic
phrase out of each title and re-wrap it in the query template families
observed in train. Every generated wording below is copied verbatim from a
real train query, with only the topic/crop slots changed.

Why this should transfer: train and test topics are disjoint (README), and
test asks about e.g. iron/magnesium/sulphur deficiency, which the train qrels
never mention. Synthetic queries over all 695 docs put those exact topic
phrases into the encoder's fine-tuning set.

    python scripts/synth.py         # preview counts + samples
Import `build_synthetic()` from Colab for the actual fine-tune.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import load  # noqa: E402

# (regex over the title with the trailing "(zone)" stripped, family key)
PATTERNS = [
    (r"^Preventing (.+) in (.+)$", "prevent"),
    (r"^Correcting (.+) in (.+)$", "correct"),
    (r"^Managing (.+) in (.+)$", "manage"),
    (r"^Identifying (.+) in (.+)$", "identify"),
    (r"^Telling (.+) in (.+) apart from look-alikes$", "tell_apart"),
    (r"^What causes (.+) in (.+)$", "causes"),
    (r"^Controlling (.+) on (.+)$", "control"),
    (r"^How (.+) spreads in (.+)$", "spreads"),
    (r"^Symptoms of (.+) in (.+)$", "symptoms"),
    (r"^Damage caused by (.+) on (.+)$", "damage"),
    (r"^Fertiliser recommendation for (.+)$", "fert_rec"),
    (r"^Adapting to (.+)$", "adapt"),
    (r"^Assessing (.+)$", "assess"),
    (r"^(.+): the problem$", "problem"),
    (r"^(.+): the risk to crops$", "risk"),
]

# {x} = topic phrase, {c} = crop, all lowercased. Wordings verbatim from train.
TEMPLATES = {
    "prevent": [
        "How can I prevent {x} in {c}?",
        "How do I stop {x} before it hits my {c}?",
    ],
    "correct": [
        "How do I fix {x} in {c}?",
        "What fertiliser corrects {x} in {c}?",
        "Why does my {c} get {x}?",
    ],
    "manage": [
        "How do I manage {x} in {c}?",
        "What should I do about an outbreak of {x} in {c}?",
    ],
    "identify": [
        "What does {x} look like in {c}?",
        "How do I know if my {c} has {x}?",
    ],
    "tell_apart": [
        "How can I tell {x} in {c} apart from other problems?",
    ],
    "causes": [
        "What causes {x} in {c}?",
        "Why does my {c} get {x}?",
    ],
    "control": [
        "How do I control {x} on {c}?",
        "What is the best way to deal with {x} on {c}?",
    ],
    "spreads": [
        "How does {x} spread in {c}?",
        "What makes {x} get worse on {c}?",
    ],
    "symptoms": [
        "What are the symptoms of {x} in {c}?",
        "My {c} shows spots and damage — could it be {x}?",
    ],
    "damage": [
        "What damage does {x} do to {c}?",
        "How much yield can {x} cost me on {c}?",
    ],
    "fert_rec": [
        "What is the best fertiliser programme for {c}?",
        "What fertiliser should I use for {c}?",
    ],
    "adapt": [
        "How can I adapt my farming to {x}?",
        "How do I cope with {x} on my farm?",
        "How does {x} affect my crops?",
    ],
    "assess": [
        "How do I check for {x} in my soil?",
        "How can I tell if my field has {x}?",
    ],
    "problem": [
        "What can I do about {x} in my field?",
        "Why is {x} bad for my crops?",
        "How can I tell if my field has {x}?",
    ],
    "risk": [
        "What is the risk of {x} to farming?",
        "How does {x} affect my crops?",
    ],
    "fallback": [
        "How do I manage {x}?",
        "What should I do about {x}?",
    ],
}

# The deficiency titles use disambiguating wordings of their own.
DEFICIENCY_EXTRA = "Is it {x} or another deficiency in my {c}?"


def _known_crops(docs) -> set[str]:
    return {c.lower() for c in docs.crop.dropna().unique() if c != "(general)"}


def build_synthetic(docs: pd.DataFrame) -> pd.DataFrame:
    """One row per (synthetic query, source document_id)."""
    crops = _known_crops(docs)
    rows = []
    for r in docs.itertuples():
        core = re.sub(r"\s*\(.*?\)\s*$", "", r.title).strip()
        # Long-tail titles often lead with a gerund ("Understanding Fall
        # Armyworm Impact...") that reads badly inside a question template.
        fallback_x = re.sub(
            r"^(understanding|improving|enhancing|revitalizing|optimizing|"
            r"addressing|refining|assessing) ",
            "", core.lower())
        family, x, c = "fallback", fallback_x, None
        for pat, fam in PATTERNS:
            m = re.match(pat, core)
            if not m:
                continue
            g = [s.lower() for s in m.groups()]
            # Two-slot patterns only count when the second slot really is a
            # crop ("Managing Soil Acidity in Ethiopia" must not become
            # "How do I manage soil acidity in ethiopia?" with crop=ethiopia).
            if len(g) == 2 and g[1] not in crops:
                continue
            family, x = fam, g[0]
            c = g[1] if len(g) == 2 else None
            if fam == "fert_rec":  # its single capture is the crop, not a topic
                x, c = None, g[0]
            break
        # `topic` is what a negative document must NOT mention: for the
        # crop-only fertiliser family that is the crop itself.
        topic = x if family != "fert_rec" else c
        for t in TEMPLATES[family]:
            if "{c}" in t and c is None:
                continue
            rows.append((t.format(x=x, c=c), r.document_id, family, topic))
        if family in ("correct", "identify", "tell_apart") and "deficiency" in x:
            rows.append(
                (DEFICIENCY_EXTRA.format(x=x, c=c), r.document_id, family, topic))
    out = pd.DataFrame(rows, columns=["query", "document_id", "family", "topic"])
    return out.drop_duplicates(subset=["query", "document_id"])


if __name__ == "__main__":
    docs, train_q, _, _ = load()
    synth = build_synthetic(docs)
    print(f"{len(synth)} synthetic pairs over {synth.document_id.nunique()} docs")
    print(synth.family.value_counts().to_string())
    print("\nsamples:")
    print(synth.sample(15, random_state=0)[["query", "document_id"]].to_string())
    overlap = set(synth["query"]) & set(train_q["query"])
    print(f"\nverbatim overlap with real train queries: {len(overlap)}")
