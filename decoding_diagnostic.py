"""
Decoding/precision diagnostic for the vanilla baseline: scores the un-adapted model on
the full devtest split under every setting of the fair-decoding grid (no dev-tuning)
plus the trainer's intermediate-eval setting,
once per weight precision -- 4-bit NF4 (as loaded everywhere else in this repo) and
unquantized fp32. Used to see which configuration reproduces the paper's reported
baseline numbers.

Usage:
    python3 decoding_diagnostic.py --lang eus_Latn --scale 600m --source en

Writes {output_dir}/{lang}_decoding_grid.csv (precision, setting, chrF++) and one
{output_dir}/{lang}_{precision}_{setting}.csv (src,mt,ref) per run.
"""
import argparse
import gc
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from metrics.chrf import chrf_corpus
from metrics.decoding import GRID, translate
from src.models import DEVICE, load_vanilla_model

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
SOURCE_TO_CODE = {"en": "eng_Latn", "es": "spa_Latn"}
SOURCE_TO_DATA_DIR = {"en": "data/flores200_eng", "es": "data/flores200_spa"}
PRECISIONS = ("4bit", "fp32")
# The fair-decoding grid plus the setting src/trainer.py's intermediate eval uses, which
# matches the paper's Table 6 "Max tokens (eval) 128".
SETTINGS = {**GRID, "trainer_eval(beam4,lp1.0,cap128)": dict(num_beams=4, max_new_tokens=128)}


def parse_args():
    p = argparse.ArgumentParser(description="Score vanilla NLLB under every decoding setting and precision.")
    p.add_argument("--lang", required=True, help="FLORES-200 target language code, e.g. eus_Latn")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--source", choices=SOURCE_TO_CODE, default="en")
    p.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    p.add_argument("--batch_size", type=int, default=16,
                    help="Lower it (e.g. 8) if fp32 runs out of GPU memory.")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None,
                    help="Defaults to results/baseline/diagnostic/{scale}_{source}")
    return p.parse_args()


def load_model(model_name, precision):
    if precision == "4bit":
        return load_vanilla_model(model_name)
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32).to(DEVICE).eval()


def main():
    args = parse_args()
    model_name = SCALE_TO_MODEL[args.scale]
    data_dir = args.data_dir or SOURCE_TO_DATA_DIR[args.source]
    out_dir = args.output_dir or f"results/baseline/diagnostic/{args.scale}_{args.source}"
    os.makedirs(out_dir, exist_ok=True)

    devtest = pd.read_parquet(os.path.join(data_dir, args.lang, "devtest.parquet"))
    srcs, refs = devtest.src.astype(str).tolist(), devtest.ref.astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SOURCE_TO_CODE[args.source])
    tgt_token_id = tokenizer.convert_tokens_to_ids(args.lang)

    rows = []
    for precision in args.precisions:
        print(f"Loading vanilla {model_name} ({precision})...")
        model = load_model(model_name, precision)
        for name, kwargs in SETTINGS.items():
            hyps = translate(model, tokenizer, tgt_token_id, srcs, kwargs,
                             batch_size=args.batch_size, device=DEVICE)
            score = chrf_corpus(hyps, refs)
            print(f"  {precision} | {name} | chrF++={score:.3f}")
            rows.append({"lang": args.lang, "scale": args.scale, "source": args.source,
                         "precision": precision, "setting": name, "chrf": score})
            slug = re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")
            pd.DataFrame({"src": srcs, "mt": hyps, "ref": refs}).to_csv(
                os.path.join(out_dir, f"{args.lang}_{precision}_{slug}.csv"), index=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, f"{args.lang}_decoding_grid.csv"), index=False)
    print("\n" + summary.pivot(index="setting", columns="precision", values="chrf").round(2).to_string())


if __name__ == "__main__":
    main()
