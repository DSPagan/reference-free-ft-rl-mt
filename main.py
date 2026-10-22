"""
Trains one GRPO adapter for one language (English source). Single run per invocation.

Usage:
    uv run python main.py --lang eus_Latn --scale 600m

Outputs, under --output_dir (default results/{lang}_{scale}):
    adapters_best/{lang}/   -- highest-intermediate-chrF++ checkpoint (what the paper
                                evaluates; load this with generate_translations.py)
    adapters/{lang}/        -- final-step adapter
    training_metrics.csv    -- per-step loss/reward/KL/temperature log
    config.txt              -- resolved run configuration, for reproducibility
"""
import argparse
import gc
import os

import pandas as pd
import torch

from src.models import load_actor
from src.reward import GRPORewarder, RewardResources
from src.trainer import TrainingLogger, run_training

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
DATA_DIR = "data/flores200_eng"


def parse_args():
    p = argparse.ArgumentParser(description="Train one GRPO adapter for one language.")
    p.add_argument("--lang", required=True, help="FLORES-200 target language code, e.g. eus_Latn")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--data_dir", default=DATA_DIR,
                    help="Expects {lang}/dev.parquet (columns src, ref), built by "
                         "prepare_data.py. Used for both training and the sparse "
                         "intermediate chrF++ eval.")
    p.add_argument("--output_dir", default=None, help="Defaults to results/{lang}_{scale}")

    p.add_argument("--labse_weight", type=float, default=None,
                    help="Hybrid-reward blend (reward = w*LaBSE + (1-w)*COMET-Kiwi). If not "
                         "given, looked up from reward_weights.csv for --lang (falls back to "
                         "0.5 if the language isn't in that file). The paper uses 0.5.")

    p.add_argument("--n_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--k", type=int, default=12, help="Samples per GRPO group.")
    p.add_argument("--kl_beta", type=float, default=0.001)
    p.add_argument("--max_new_tokens_train", type=int, default=64)
    p.add_argument("--max_new_tokens_eval", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_n_samples", type=int, default=100)
    p.add_argument("--eval_batch_size", type=int, default=8)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--temp_initial", type=float, default=1.2)
    p.add_argument("--temp_max", type=float, default=1.6)
    p.add_argument("--temp_step", type=float, default=0.05)
    p.add_argument("--temp_patience", type=int, default=500)
    p.add_argument("--temp_cooldown", type=int, default=800)

    p.add_argument("--n_segments", type=int, default=None, help="Truncate training data to this many sentences.")
    p.add_argument("--seed", type=int, default=None, help="Seed python/numpy/torch RNGs (default: unseeded).")
    return p.parse_args()


def resolve_labse_weight(args):
    if args.labse_weight is not None:
        return args.labse_weight
    if os.path.exists("reward_weights.csv"):
        weights = pd.read_csv("reward_weights.csv").set_index("lang")["labse_weight"]
        if args.lang in weights.index:
            w = float(weights[args.lang])
            print(f"  LaBSE weight for {args.lang}: {w} (from reward_weights.csv)")
            return w
    print(f"  [WARN] No calibrated weight for {args.lang} in reward_weights.csv -- using 0.5")
    return 0.5


def main():
    args = parse_args()
    if args.seed is not None:
        import random

        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model_name = SCALE_TO_MODEL[args.scale]
    out_dir = args.output_dir or f"results/{args.lang}_{args.scale}"
    os.makedirs(out_dir, exist_ok=True)

    labse_weight = resolve_labse_weight(args)
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"model_name: {model_name}\nlabse_weight: {labse_weight}\n")

    dev_path = os.path.join(args.data_dir, args.lang, "dev.parquet")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"No training/eval data at {dev_path}")
    df_train = pd.read_parquet(dev_path)
    if args.n_segments:
        df_train = df_train.iloc[:args.n_segments]
    df_eval = df_train  # same FLORES-200 dev split; intermediate eval is sparse and sampled

    print(f"\n{'=' * 60}\n  Language:    {args.lang}\n  Model:       {model_name}\n"
          f"  Reward:      hybrid (w={labse_weight})\n  Epochs:      {args.n_epochs} "
          f"({len(df_train)} sentences/epoch)\n  Output:      {out_dir}\n{'=' * 60}\n")

    resources = RewardResources(model_name)
    rewarder = GRPORewarder(resources, labse_weight)
    logger = TrainingLogger(os.path.join(out_dir, "training_metrics.csv"))
    actor = load_actor(model_name, args.lora_r, args.lora_alpha, args.lora_dropout)

    try:
        best_chrf, best_step = run_training(
            actor, resources.tokenizer, rewarder, logger, df_train, df_eval, args.lang, out_dir,
            n_epochs=args.n_epochs, lr=args.lr, k=args.k,
            max_new_tokens_train=args.max_new_tokens_train, max_new_tokens_eval=args.max_new_tokens_eval,
            kl_beta=args.kl_beta, eval_every=args.eval_every,
            eval_n_samples=args.eval_n_samples, eval_batch_size=args.eval_batch_size,
            temp_kwargs=dict(initial=args.temp_initial, max_temp=args.temp_max, step=args.temp_step,
                              patience=args.temp_patience, cooldown=args.temp_cooldown),
        )
    finally:
        logger.close()

    final_adapter_path = os.path.join(out_dir, "adapters", args.lang)
    os.makedirs(final_adapter_path, exist_ok=True)
    actor.save_pretrained(final_adapter_path)

    del actor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nDone. Outputs in {out_dir}/")
    print(f"  adapters_best/{args.lang}/  -- best checkpoint (chrF++={best_chrf:.4f} @ step {best_step})")
    print(f"  adapters/{args.lang}/       -- final-step adapter")
    print("  training_metrics.csv        -- per-step training log")
    print("\nEvaluate with: uv run python generate_translations.py "
          f"--adapter_path {out_dir}/adapters_best/{args.lang} --lang {args.lang} --scale {args.scale}")


if __name__ == "__main__":
    main()
