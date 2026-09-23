"""
Verify the translator before serving it. From this folder:

    set -a && source .env && set +a
    .venv/bin/python smoke_test.py

Two stages, cheap first:

1. Tag check (~30 MB). Resolve the language tags against the fine-tuned
   tokenizer and assert they are real tokens. If training had customized the
   tokenizer, the tags would silently resolve to <unk> and generation would
   emit fluent-looking noise -- this catches that for the price of one
   tokenizer download instead of two 2.4 GB checkpoints.

2. Generation (~4.8 GB, both directions). Translate a few corpus pairs and
   print the reference beside the output. The pairs come from the baseline
   experiment's test split, which these older checkpoints were not trained
   against, so they are a coherence check, not a held-out score.

Samples are read from the split artifacts, never written into this file:
corpus text is not ours to redistribute.
"""

import os
import sys

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit(
        "HF_TOKEN is not set.\n"
        "Without it huggingface_hub silently falls back to your cached login in\n"
        "~/.cache/huggingface/token, so this test would pass on credentials a\n"
        "deployment will never have. Run: set -a && source .env && set +a"
    )

from transformers import AutoTokenizer  # noqa: E402

from app import DIRECTIONS, REPO, VARIANT, subfolder, translate  # noqa: E402
from qomlaq.config import DIRECTIONS as LANGUAGES  # noqa: E402
from qomlaq.pipeline import build, load_pinned_split  # noqa: E402

N_SAMPLES = 3


def check_tags():
    """Stage 1: the language tags must be real tokens, not <unk>."""
    print(f"Tag check against {REPO} (variant: {VARIANT})")
    ok = True
    for name, direction in DIRECTIONS.items():
        tok = AutoTokenizer.from_pretrained(REPO, subfolder=subfolder(direction), token=TOKEN)
        language = LANGUAGES[direction]
        for role, tag in (("src_lang", language.src_lang), ("tgt_lang", language.tgt_lang)):
            tag_id = tok.convert_tokens_to_ids(tag)
            bad = tag_id is None or tag_id == tok.unk_token_id
            ok &= not bad
            print(f"  {name:18s} {role}={tag} -> id {tag_id}"
                  f"{'   *** UNK, tag logic is wrong ***' if bad else ''}")
    if not ok:
        sys.exit("\nA language tag resolved to <unk>. Do not serve translations: "
                 "generation would produce plausible-looking garbage.")
    print("  all tags resolve to real tokens\n")


def load_samples():
    """A few non-Bible pairs from the baseline experiment's base/stratified test set."""
    built = build("baseline", verbose=False)
    test = load_pinned_split(built, "base/stratified").test
    rows = test[~test["source_doc"].str.contains("Biblia")]
    rows = rows[rows["qom"].str.len().between(20, 90)]
    return built.corpus.sha8, rows.head(N_SAMPLES).to_dict("records")


def main():
    check_tags()

    try:
        corpus_sha8, rows = load_samples()
    except Exception as exc:
        sys.exit(f"Could not load sample pairs ({exc}).\n"
                 "Build them first from the repo root: qomlaq build baseline")

    print(f"Generation check on {len(rows)} corpus pairs (corpus {corpus_sha8})")
    print("Downloads ~4.8 GB on first run.\n")

    for row in rows:
        print(f"[{row['source_doc']} / {row['pair_uid']}]")
        for name, direction in DIRECTIONS.items():
            language = LANGUAGES[direction]
            print(f"  {name}")
            print(f"    in  : {row[language.src_col]}")
            print(f"    out : {translate(row[language.src_col], name)}")
            print(f"    ref : {row[language.tgt_col]}")
        print()


if __name__ == "__main__":
    main()
