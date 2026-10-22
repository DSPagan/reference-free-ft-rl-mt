"""
Loads a trained GRPO adapter (and, for comparison, the vanilla base model), translates
a test set under the paper's fair-decoding protocol (each system independently dev-tuned
over a 3-setting beam-search grid, see metrics/decoding.py), and scores chrF++.

Usage:
    uv run python generate_translations.py --adapter_path results/eus_Latn_600m/adapters_best/eus_Latn \
        --lang eus_Latn --scale 600m

Omit --adapter_path to evaluate only the vanilla baseline. Pass --setting to decode both
systems with one fixed grid setting instead of dev-tuning it.

Writes {output_dir}/{lang}_vanilla.csv, {output_dir}/{lang}_trained.csv (src,mt,ref),
{output_dir}/{lang}_summary.json (chosen settings + chrF++) and prints the chrF++ gain.
"""
import argparse
import json
import os

import pandas as pd
import torch
from peft import PeftModel

from metrics.chrf import chrf_corpus
from metrics.decoding import GRID, fair_decode, translate
from src.models import load_vanilla_model

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
DATA_DIR = "data/flores200_eng"


def parse_args():
    p = argparse.ArgumentParser(description="Translate + evaluate a trained adapter against vanilla.")
    p.add_argument("--adapter_path", default=None,
                    help="Path to a trained LoRA adapter directory. Omit to evaluate only vanilla.")
    p.add_argument("--lang", required=True, help="FLORES-200 target language code, e.g. eus_Latn")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--data_dir", default=DATA_DIR,
                    help="Expects {lang}/dev.parquet (for decoding-setting dev-tuning) and "
                         "{lang}/devtest.parquet (held-out test).")
    p.add_argument("--output_dir", default="results/translations")
    p.add_argument("--setting", choices=GRID, default=None,
                    help="Decode both systems with this fixed setting instead of dev-tuning the "
                         "grid. 'default(beam5,lp1.0)' reproduces the paper's Table 7 baseline.")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Sentences per generate() call. Lower it (e.g. 8) if the GPU runs out of memory.")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = SCALE_TO_MODEL[args.scale]
    data_dir = args.data_dir
    os.makedirs(args.output_dir, exist_ok=True)

    dev = pd.read_parquet(os.path.join(data_dir, args.lang, "dev.parquet")).iloc[:100]
    devtest = pd.read_parquet(os.path.join(data_dir, args.lang, "devtest.parquet"))
    dev_srcs, dev_refs = dev.src.astype(str).tolist(), dev.ref.astype(str).tolist()
    test_srcs, test_refs = devtest.src.astype(str).tolist(), devtest.ref.astype(str).tolist()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    tgt_token_id = tokenizer.convert_tokens_to_ids(args.lang)

    def decode(model):
        if args.setting:
            hyps = translate(model, tokenizer, tgt_token_id, test_srcs, GRID[args.setting], args.batch_size)
            return hyps, args.setting
        return fair_decode(model, tokenizer, tgt_token_id, dev_srcs, dev_refs, test_srcs, args.batch_size)

    mode = f"fixed setting {args.setting}" if args.setting else "dev-tuning 3-setting grid"
    print(f"Loading vanilla {model_name}...")
    base = load_vanilla_model(model_name)
    print(f"Decoding vanilla ({mode})...")
    van_hyps, van_setting = decode(base)
    van_chrf = chrf_corpus(van_hyps, test_refs)
    pd.DataFrame({"src": test_srcs, "mt": van_hyps, "ref": test_refs}).to_csv(
        os.path.join(args.output_dir, f"{args.lang}_vanilla.csv"), index=False)
    print(f"  Vanilla | setting={van_setting} | chrF++={van_chrf:.3f}")

    summary = {"lang": args.lang, "scale": args.scale,
               "vanilla_setting": van_setting, "vanilla_chrf": van_chrf}
    summary_path = os.path.join(args.output_dir, f"{args.lang}_summary.json")

    if args.adapter_path is None:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        torch.cuda.empty_cache()
        return

    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base, args.adapter_path).eval()
    print(f"Decoding trained adapter ({mode})...")
    trained_hyps, trained_setting = decode(model)
    trained_chrf = chrf_corpus(trained_hyps, test_refs)
    pd.DataFrame({"src": test_srcs, "mt": trained_hyps, "ref": test_refs}).to_csv(
        os.path.join(args.output_dir, f"{args.lang}_trained.csv"), index=False)
    print(f"  Trained | setting={trained_setting} | chrF++={trained_chrf:.3f}")

    print(f"\nchrF++ gain: {trained_chrf - van_chrf:+.3f}")
    summary.update(adapter_path=args.adapter_path, trained_setting=trained_setting,
                   trained_chrf=trained_chrf, chrf_gain=trained_chrf - van_chrf)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
