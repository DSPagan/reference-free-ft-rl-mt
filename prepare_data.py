"""
Builds the FLORES-200 parquet files main.py / generate_translations.py expect, downloading
the per-language parquet files from the HuggingFace dataset facebook/flores
(data/language/{lang}/{split}-00000-of-00001.parquet, aligned by row order).

English source (eng_Latn). Writes to data/flores200_eng/{target_lang}/{split}.parquet.

Usage:
    uv run python prepare_data.py --languages eus_Latn ...
"""
import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO = "facebook/flores"
SOURCE = "eng_Latn"
SPLITS = ["dev", "devtest"]

LANGUAGES = [
    "eus_Latn", "swh_Latn", "bod_Tibt", "tur_Latn", "jpn_Jpan", "zho_Hant", "zho_Hans",
    "yor_Latn", "arb_Arab", "bel_Cyrl", "ben_Beng", "ces_Latn", "pol_Latn",
    "ell_Grek", "kor_Hang", "tha_Thai", "heb_Hebr", "hye_Armn", "kat_Geor", "amh_Ethi",
    "khm_Khmr", "mar_Deva", "fin_Latn", "tam_Taml", "khk_Cyrl", "plt_Latn",
]


def load_sentences(lang, split):
    """Returns the list of sentences for `lang`/`split`, or None if HF has no such file."""
    try:
        path = hf_hub_download(HF_REPO, f"data/language/{lang}/{split}-00000-of-00001.parquet",
                               repo_type="dataset")
    except Exception as e:
        print(f"  [SKIP] {lang} {split}: could not download from {HF_REPO} ({type(e).__name__})")
        return None
    return pd.read_parquet(path)["sentence"].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=LANGUAGES)
    p.add_argument("--out_dir", default="data/flores200_eng")
    args = p.parse_args()

    out_dir = args.out_dir

    for split in SPLITS:
        src_sents = load_sentences(SOURCE, split)
        if src_sents is None:
            continue

        for lang in args.languages:
            tgt_sents = load_sentences(lang, split)
            if tgt_sents is None:
                continue
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
