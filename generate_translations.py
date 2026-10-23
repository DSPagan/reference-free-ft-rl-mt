"""
Loads a trained GRPO adapter (and, for comparison, the vanilla base model), translates
a test set under the paper's fair-decoding protocol (each system independently dev-tuned
over a 3-setting beam-search grid, see metrics/decoding.py), and scores it with the same
metrics main.py logs during training (src/eval_metrics.py): chrF++, BLEU, COMET-Kiwi and,
for Catalan targets, the dialectal purity index (IPD).

Usage:
    uv run python generate_translations.py --adapter_path results/eus_Latn_600m/adapters_best/eus_Latn \
        --lang eus_Latn --scale 600m

Omit --adapter_path to evaluate only the vanilla baseline. Pass --setting to decode both
systems with one fixed grid setting instead of dev-tuning it. --metrics picks the extra
metrics next to chrF++ (default: bleu, comet, and ipd for cat_Latn; no values = chrF++ only).

Writes {output_dir}/{lang}_vanilla.csv, {output_dir}/{lang}_trained.csv (src,mt,ref),
{output_dir}/{lang}_summary.json (chosen settings + every metric, for vanilla, trained, the
reference and the trained-minus-vanilla gain) and prints the same comparison.
"""
import argparse
import json
import os

import pandas as pd
import torch
from peft import PeftModel

from metrics.decoding import GRID, fair_decode, translate
from src.data import DATASETS, resolve_data_dir
from src.eval_metrics import OPTIONAL_METRICS, EvalMetrics, default_metrics, format_metrics, metric_gain
from src.models import load_vanilla_model

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
SOURCE_TO_CODE = {"en": "eng_Latn", "es": "spa_Latn"}


def parse_args():
    p = argparse.ArgumentParser(description="Translate + evaluate a trained adapter against vanilla.")
    p.add_argument("--adapter_path", default=None,
                    help="Path to a trained LoRA adapter directory. Omit to evaluate only vanilla.")
    p.add_argument("--lang", required=True, help="FLORES-200 target language code, e.g. eus_Latn")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--source", choices=SOURCE_TO_CODE, default="en")
    p.add_argument("--dataset", choices=DATASETS, default="flores",
                    help="flores: FLORES-200. uji: UJI Valencian-English news corpus (English source, "
                         "Valencian references; build it with prepare_uji.py).")
    p.add_argument("--data_dir", default=None,
                    help="Defaults to the dataset's directory for --source (data/flores200_eng, "
                         "data/flores200_spa or data/uji_eng). Expects {lang}/dev.parquet (for "
                         "decoding-setting dev-tuning) and {lang}/devtest.parquet (held-out test).")
    p.add_argument("--output_dir", default=None,
                    help="Defaults to results/translations, or results/translations_{dataset} for "
                         "datasets other than flores (so runs on different data never overwrite each other).")
    p.add_argument("--setting", choices=GRID, default=None,
                    help="Decode both systems with this fixed setting instead of dev-tuning the "
                         "grid. 'default(beam5,lp1.0)' reproduces the paper's Table 7 baseline.")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Sentences per generate() call (and per COMET-Kiwi batch). Lower it "
                         "(e.g. 8) if the GPU runs out of memory.")
    p.add_argument("--metrics", nargs="*", choices=OPTIONAL_METRICS, default=None,
                    help="Metrics reported next to chrF++ (always computed), the same as the training-time "
                         "eval. Default: bleu, comet, and ipd when --lang is cat_Latn. Pass the flag with "
                         "no values for chrF++ only. comet needs the gated COMET-Kiwi model (HF_TOKEN).")
    p.add_argument("--ipd", action="store_true",
                    help="Kept for compatibility: make sure ipd (AVLizador's corpus-level IPD, the Valencian "
                         "share of dialectally marked words) is among the metrics, even with --metrics.")
    return p.parse_args()


def ipd_summary(m, n_sentences):
    """The IPD fields of a metrics dict, shaped like src.ipd_reward.corpus_ipd's result."""
    return {"ipd": m["ipd"], "western": m["ipd_western"], "eastern": m["ipd_eastern"],
            "unmarked_sentences": m["ipd_unmarked"], "sentences": n_sentences,
            "mean_sentence_ipd": m["ipd_mean"]}


def main():
    args = parse_args()
    model_name = SCALE_TO_MODEL[args.scale]
    data_dir = resolve_data_dir(args.dataset, args.source, args.data_dir)
    if args.output_dir is None:
        args.output_dir = "results/translations" + ("" if args.dataset == "flores" else f"_{args.dataset}")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = default_metrics(args.lang) if args.metrics is None else list(args.metrics)
    if args.ipd and "ipd" not in metrics:
        metrics.append("ipd")
    if "ipd" in metrics and args.lang != "cat_Latn":
        raise ValueError("the ipd metric only makes sense for Catalan targets (--lang cat_Latn)")

    dev = pd.read_parquet(os.path.join(data_dir, args.lang, "dev.parquet")).iloc[:100]
    devtest = pd.read_parquet(os.path.join(data_dir, args.lang, "devtest.parquet"))
    dev_srcs, dev_refs = dev.src.astype(str).tolist(), dev.ref.astype(str).tolist()
    test_srcs, test_refs = devtest.src.astype(str).tolist(), devtest.ref.astype(str).tolist()

    comet_model = None
    if "comet" in metrics:
        from src.reward import load_comet_kiwi
        comet_model = load_comet_kiwi()
    evaluator = EvalMetrics(metrics, comet_model, comet_batch_size=args.batch_size)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SOURCE_TO_CODE[args.source])
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
    van = evaluator.compute(test_srcs, van_hyps, test_refs)
    pd.DataFrame({"src": test_srcs, "mt": van_hyps, "ref": test_refs}).to_csv(
        os.path.join(args.output_dir, f"{args.lang}_vanilla.csv"), index=False)
    print(f"  Vanilla | setting={van_setting} | {format_metrics(van)}")

    # What the references themselves score: only the reference-free metrics mean anything.
    ref_all = evaluator.compute(test_srcs, test_refs, test_refs)
    ref = {k: v for k, v in ref_all.items() if k not in ("chrf", "bleu")}
    if ref:
        print(f"  Reference | {format_metrics(ref)}")

    summary = {"lang": args.lang, "scale": args.scale, "source": args.source, "dataset": args.dataset,
               "metrics": ["chrf"] + metrics, "vanilla_setting": van_setting, "vanilla_chrf": van["chrf"],
               "vanilla_metrics": van, "reference_metrics": ref}
    summary_path = os.path.join(args.output_dir, f"{args.lang}_summary.json")
    if "ipd" in metrics:
        summary["reference_ipd"] = ipd_summary(ref, len(test_refs))
        summary["vanilla_ipd"] = ipd_summary(van, len(test_refs))

    if args.adapter_path is None:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        torch.cuda.empty_cache()
        return

    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base, args.adapter_path).eval()
    print(f"Decoding trained adapter ({mode})...")
    trained_hyps, trained_setting = decode(model)
    trained = evaluator.compute(test_srcs, trained_hyps, test_refs)
    pd.DataFrame({"src": test_srcs, "mt": trained_hyps, "ref": test_refs}).to_csv(
        os.path.join(args.output_dir, f"{args.lang}_trained.csv"), index=False)
    print(f"  Trained | setting={trained_setting} | {format_metrics(trained)}")

    gain = metric_gain(trained, van)
    print(f"\nTrained - vanilla | {format_metrics(gain, signed=True)}")
    summary.update(adapter_path=args.adapter_path, trained_setting=trained_setting,
                   trained_chrf=trained["chrf"], chrf_gain=gain["chrf"],
                   trained_metrics=trained, metric_gain=gain)
    if "ipd" in metrics:
        summary["trained_ipd"] = ipd_summary(trained, len(test_refs))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
