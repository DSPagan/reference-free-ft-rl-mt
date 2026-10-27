"""
Builds the parquet files main.py / generate_translations.py expect from the UJI Valencian-
English news corpus (data/va_en_uji/va-en-sentence.length.ner.curated.jsonl, one JSON
object per line with fields VA, EN and source = the news article the sentence comes from).

The task is English -> Catalan (cat_Latn) with the Valencian side as the reference: src is
the English sentence, ref the Valencian one. Writes
data/uji_eng/cat_Latn/{train,dev,devtest}.parquet (columns src, ref, article), the same
layout as the FLORES-200 files, so the rest of the code only needs `--dataset uji`.

The split is by news article (all sentences of an article land in the same split), so
neighbouring sentences of one text never leak from train into dev/devtest, and is seeded so
it is reproducible. Rows are shuffled within each split.

If the dialectal subset built by filter_dialectal.py exists (--filtered_input; only the
sentences whose Valencian side has a word that differs in Catalan), the same splits restricted
to those sentences are also written to a sibling folder (--filtered_out_dir, default
data/uji_eng_dialectal). Reusing the full split keeps both datasets consistent: a sentence
that is in train for one is in train for the other, so there is no leakage between them.

Cleaning: most sentences (both sides) end in a doubled full stop ('..', an artifact of the
corpus build); a trailing '..' is collapsed to '.', while a genuine ellipsis ('...') is kept.
Pass --keep_double_period to leave the text exactly as it is in the jsonl.

Usage:
    uv run python filter_dialectal.py
    uv run python prepare_uji.py
    uv run python prepare_uji.py --n_devtest 1000 --n_dev 500 --seed 42
"""
import argparse
import json
import os
import random
import re

import pandas as pd

DOUBLE_PERIOD = re.compile(r"(?<!\.)\.\.$")


def clean(text, keep_double_period):
    text = text.strip()
    return text if keep_double_period else DOUBLE_PERIOD.sub(".", text)


def load_rows(path, keep_double_period):
    """Reads the jsonl, dropping rows with an empty or missing side. Returns (rows, n_dropped)."""
    rows, dropped = [], 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            va, en = r.get("VA"), r.get("EN")
            if not isinstance(va, str) or not isinstance(en, str) or not va.strip() or not en.strip():
                dropped += 1
                continue
            rows.append({"src": clean(en, keep_double_period), "ref": clean(va, keep_double_period),
                         "article": r.get("source", "")})
    return rows, dropped


def split_by_article(rows, n_devtest, n_dev, seed):
    """Shuffles the articles and fills devtest, then dev, then train, until each has at
    least its target number of sentences (the last article of a split may overshoot it)."""
    by_article = {}
    for r in rows:
        by_article.setdefault(r["article"], []).append(r)
    articles = sorted(by_article)
    rng = random.Random(seed)
    rng.shuffle(articles)

    splits = {"devtest": [], "dev": [], "train": []}
    targets = {"devtest": n_devtest, "dev": n_dev}
    for a in articles:
        name = next((s for s in ("devtest", "dev") if len(splits[s]) < targets[s]), "train")
        splits[name].extend(by_article[a])
    for sents in splits.values():
        rng.shuffle(sents)
    return splits


def write_splits(splits, out_dir, lang):
    lang_dir = os.path.join(out_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    for name, sents in splits.items():
        out_path = os.path.join(lang_dir, f"{name}.parquet")
        pd.DataFrame(sents, columns=["src", "ref", "article"]).to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(sents)} rows, {len({s['article'] for s in sents})} articles)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/va_en_uji/va-en-sentence.length.ner.curated.jsonl")
    p.add_argument("--out_dir", default="data/uji_eng")
    p.add_argument("--filtered_input", default="data/va_en_uji/va-en-sentence.length.ner.curated.dialectal.jsonl",
                   help="Output of filter_dialectal.py. Skipped, with a message, if it does not exist.")
    p.add_argument("--filtered_out_dir", default="data/uji_eng_dialectal")
    p.add_argument("--lang", default="cat_Latn", help="Target-language directory name to write under out_dir.")
    p.add_argument("--n_devtest", type=int, default=1000, help="Held-out test sentences (FLORES devtest has 1012).")
    p.add_argument("--n_dev", type=int, default=500,
                   help="Dev sentences: decoding-setting tuning (first 100) and the intermediate eval.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_double_period", action="store_true",
                   help="Do not collapse a trailing '..' into '.'.")
    args = p.parse_args()

    rows, dropped = load_rows(args.input, args.keep_double_period)
    print(f"Read {len(rows)} sentence pairs from {args.input} ({dropped} dropped: empty side)")

    splits = split_by_article(rows, args.n_devtest, args.n_dev, args.seed)
    write_splits(splits, args.out_dir, args.lang)
    print(f"\nDone. Output under {args.out_dir}/ (EN -> {args.lang}, Valencian references)")

    if not os.path.exists(args.filtered_input):
        print(f"\n{args.filtered_input} not found: skipping the dialectal subset (run filter_dialectal.py first)")
        return
    filtered, _ = load_rows(args.filtered_input, args.keep_double_period)
    keep = {(r["src"], r["ref"], r["article"]) for r in filtered}
    subset = {name: [r for r in sents if (r["src"], r["ref"], r["article"]) in keep]
              for name, sents in splits.items()}
    print(f"\nDialectal subset: {sum(map(len, subset.values()))} of {len(rows)} sentence pairs")
    write_splits(subset, args.filtered_out_dir, args.lang)
    print(f"\nDone. Dialectal subset under {args.filtered_out_dir}/")


if __name__ == "__main__":
    main()
