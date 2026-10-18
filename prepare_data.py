"""
Builds the FLORES-200 parquet files main.py / generate_translations.py expect, from the
raw FLORES-200 flat-text dump (data/flores200_dataset/{dev,devtest}/{lang}.{split},
aligned by line number -- download from https://github.com/facebookresearch/flores and
point --raw_dir at it).

Writes to data/flores200_<src>/{target_lang}/{split}.parquet -- a separate tree per
source language, so English- and Spanish-source runs never share a path.

Usage:
    python3 prepare_data.py --source eng_Latn --languages arb_Arab eus_Latn khm_Khmr ...
    python3 prepare_data.py --source spa_Latn --languages arb_Arab eus_Latn khm_Khmr ...
"""
import argparse
import os

import pandas as pd

RAW_DIR = "data/flores200_dataset"
SPLITS = ["dev", "devtest"]

LANGUAGES = [
    "eus_Latn", "swh_Latn", "bod_Tibt", "tur_Latn", "jpn_Jpan", "zho_Hant", "zho_Hans",
    "yor_Latn", "arb_Arab", "bel_Cyrl", "ben_Beng", "ces_Latn", "pol_Latn",
    "ell_Grek", "kor_Hang", "tha_Thai", "heb_Hebr", "hye_Armn", "kat_Geor", "amh_Ethi",
    "khm_Khmr", "mar_Deva", "fin_Latn", "tam_Taml", "khk_Cyrl", "plt_Latn",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="e.g. eng_Latn or spa_Latn")
    p.add_argument("--languages", nargs="+", default=LANGUAGES)
    p.add_argument("--raw_dir", default=RAW_DIR)
    p.add_argument("--out_dir", default=None,
                    help="Defaults to data/flores200_<src_3letter>, e.g. data/flores200_eng")
    args = p.parse_args()

    out_dir = args.out_dir or f"data/flores200_{args.source.split('_')[0]}"

    for split in SPLITS:
        src_path = os.path.join(args.raw_dir, split, f"{args.source}.{split}")
        if not os.path.exists(src_path):
            print(f"[ERROR] source file missing: {src_path}")
            continue
        with open(src_path, encoding="utf-8") as f:
            src_sents = f.read().splitlines()

        for lang in args.languages:
            tgt_path = os.path.join(args.raw_dir, split, f"{lang}.{split}")
            if not os.path.exists(tgt_path):
                print(f"  [SKIP] {lang} {split}: no target file at {tgt_path}")
                continue
            with open(tgt_path, encoding="utf-8") as f:
                tgt_sents = f.read().splitlines()
            if len(tgt_sents) != len(src_sents):
                print(f"  [ERROR] {lang} {split}: length mismatch "
                      f"({len(src_sents)} src vs {len(tgt_sents)} tgt)")
                continue

            lang_dir = os.path.join(out_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            out_path = os.path.join(lang_dir, f"{split}.parquet")
            pd.DataFrame({"src": src_sents, "ref": tgt_sents}).to_parquet(out_path, index=False)
            print(f"  wrote {out_path} ({len(src_sents)} rows)")

    print(f"\nDone. Output under {out_dir}/")


if __name__ == "__main__":
    main()
