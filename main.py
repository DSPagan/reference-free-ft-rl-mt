"""
Trains one GRPO adapter for one language pair. Single run per invocation.

Usage:
    uv run python main.py --lang eus_Latn --scale 600m                          # paper's hybrid reward
    uv run python main.py --lang cat_Latn --scale 600m --source es --reward ipd   # Valencian (IPD) reward
    uv run python main.py --lang cat_Latn --scale 600m --dataset uji --reward ipd # UJI corpus, en -> Valencian

Outputs, under --output_dir (default results/{lang}_{scale}_{source}_{reward}[_{dataset}]):
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

from transformers import AutoTokenizer

from src.data import DATASETS, resolve_data_dir
from src.models import load_actor
from src.trainer import TrainingLogger, run_training

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
SOURCE_TO_CODE = {"en": "eng_Latn", "es": "spa_Latn"}
REWARDS = ("hybrid", "ipd")


def parse_args():
    p = argparse.ArgumentParser(description="Train one GRPO adapter for one language.")
    p.add_argument("--lang", required=True, help="FLORES-200 target language code, e.g. eus_Latn")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--source", choices=SOURCE_TO_CODE, default="en")
    p.add_argument("--dataset", choices=DATASETS, default="flores",
                    help="flores: FLORES-200 (prepare_data.py). uji: UJI Valencian-English news corpus, "
                         "English source and Valencian references (prepare_uji.py).")
    p.add_argument("--data_dir", default=None,
                    help="Defaults to the dataset's directory for --source (data/flores200_eng, "
                         "data/flores200_spa or data/uji_eng). Expects {lang}/{split}.parquet "
                         "(columns src, ref).")
    p.add_argument("--train_split", default=None,
                    help="Split trained on. Defaults to dev for flores and train for uji.")
    p.add_argument("--eval_split", default=None,
                    help="Split the sparse intermediate chrF++ eval samples from (picks adapters_best). "
                         "Defaults to dev for both datasets; when it equals --train_split the "
                         "(--n_segments-truncated) training data is reused, as in the paper's setup.")
    p.add_argument("--output_dir", default=None,
                    help="Defaults to results/{lang}_{scale}_{source}_{reward}, with a _{dataset} "
                         "suffix for datasets other than flores")

    p.add_argument("--reward", choices=REWARDS, default="hybrid",
                    help="hybrid: the paper's LaBSE + COMET-Kiwi reward. ipd: AVLizador's dialectal "
                         "purity index (Valencian share of marked words; 0 when none is marked).")
    p.add_argument("--labse_weight", type=float, default=0.5,
                    help="Hybrid-reward blend: reward = w*LaBSE + (1-w)*COMET-Kiwi. The paper uses 0.5.")

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
    data_dir = resolve_data_dir(args.dataset, args.source, args.data_dir)
    train_split = args.train_split or DATASETS[args.dataset]["train_split"]
    eval_split = args.eval_split or DATASETS[args.dataset]["eval_split"]
    suffix = "" if args.dataset == "flores" else f"_{args.dataset}"
    out_dir = args.output_dir or f"results/{args.lang}_{args.scale}_{args.source}_{args.reward}{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"resolved_data_dir: {data_dir}\n")
        f.write(f"resolved_train_split: {train_split}\n")
        f.write(f"resolved_eval_split: {eval_split}\n")

    def load_split(split):
        path = os.path.join(data_dir, args.lang, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No training/eval data at {path}")
        return pd.read_parquet(path)

    df_train = load_split(train_split)
    if args.n_segments:
        df_train = df_train.iloc[:args.n_segments]
    # FLORES-200: eval and train are the same dev split (intermediate eval is sparse and sampled).
    # UJI: eval is a held-out dev split, so the best checkpoint is not picked on training data.
    df_eval = df_train if eval_split == train_split else load_split(eval_split)

    reward_desc = f"hybrid (w={args.labse_weight})" if args.reward == "hybrid" else "ipd"
    print(f"\n{'=' * 60}\n  Language:    {SOURCE_TO_CODE[args.source]} -> {args.lang}\n"
          f"  Model:       {model_name}\n  Reward:      {reward_desc}\n  Data:        {args.dataset} "
          f"(train={train_split}, eval={eval_split}, {len(df_eval)} eval rows)\n"
          f"  Epochs:      {args.n_epochs} ({len(df_train)} sentences/epoch)\n"
          f"  Output:      {out_dir}\n{'=' * 60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.reward == "ipd":
        from src.ipd_reward import IPDRewarder
        rewarder = IPDRewarder()
    else:
        from src.reward import GRPORewarder, RewardResources
        rewarder = GRPORewarder(RewardResources(), args.labse_weight)
    logger = TrainingLogger(os.path.join(out_dir, "training_metrics.csv"))
    actor = load_actor(model_name, args.lora_r, args.lora_alpha, args.lora_dropout)

    try:
        best_chrf, best_step = run_training(
            actor, tokenizer, rewarder, logger, df_train, df_eval, args.lang, out_dir,
            n_epochs=args.n_epochs, lr=args.lr, k=args.k,
            max_new_tokens_train=args.max_new_tokens_train, max_new_tokens_eval=args.max_new_tokens_eval,
            kl_beta=args.kl_beta, eval_every=args.eval_every,
            eval_n_samples=args.eval_n_samples, eval_batch_size=args.eval_batch_size,
            temp_kwargs=dict(initial=args.temp_initial, max_temp=args.temp_max, step=args.temp_step,
                              patience=args.temp_patience, cooldown=args.temp_cooldown),
            src_lang=SOURCE_TO_CODE[args.source],
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
          f"--adapter_path {out_dir}/adapters_best/{args.lang} --lang {args.lang} --scale {args.scale} "
          f"--source {args.source} --dataset {args.dataset}" + (" --ipd" if args.lang == "cat_Latn" else ""))


if __name__ == "__main__":
    main()
