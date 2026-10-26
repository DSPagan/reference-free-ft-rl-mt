"""
Shows the GRPO group for some source sentences: samples K hypotheses exactly as
src/trainer.py's train_step does (sampling, temperature, forced target language), scores
each with the IPD reward and prints them, plus the group statistics that decide whether the
update has anything to learn from. Use it to check whether all K samples of a sentence are
equally "Catalan" (reward std ~ 0 -> advantages ~ 0 -> no learning signal).

Usage:
    uv run python inspect_group.py                                   # built-in test sentences, base model
    uv run python inspect_group.py --sentence "The boy ate a tomato." --sentence "..."
    uv run python inspect_group.py --adapter_path results/cat_Latn_600m_en_ipd_uji/adapters_best/cat_Latn
    uv run python inspect_group.py --from_data 50 --dataset uji      # aggregate over dev sentences

With --adapter_path, each sentence is sampled from the trained model and from the base model
(adapter disabled). --from_data N samples N sentences of the dataset's eval split and prints
only the aggregate group statistics.
"""
import argparse
import contextlib
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.data import DATASETS, resolve_data_dir
from src.ipd_reward import corpus_ipd
from src.models import DEVICE, load_vanilla_model

SCALE_TO_MODEL = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
SOURCE_TO_CODE = {"en": "eng_Latn", "es": "spa_Latn"}
LANG = "cat_Latn"

# Sentences meant to force Valencian/Catalan choices (verb forms, demonstratives, lexical
# variants). Candidates only: which forms count as marked is decided by AVLizador's lexicon.
DEFAULT_SENTENCES = [
    "I want you to come to my house and eat a tomato with the boy.",
    "This morning we have to buy beans and peaches at the market.",
    "That teacher serves coffee to the students and they thank him.",
    "I hope they work hard so that we finish the project on time.",
    "Those girls never go out on Tuesday because they are always studying.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Inspect the K sampled hypotheses of a GRPO group.")
    p.add_argument("--sentence", action="append", default=None,
                    help="Source sentence (repeatable). Default: a few built-in conflict sentences.")
    p.add_argument("--from_data", type=int, default=None,
                    help="Instead, sample this many sentences from the dataset's eval split and print "
                         "aggregate statistics only.")
    p.add_argument("--dataset", choices=DATASETS, default="uji")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--split", default="dev")
    p.add_argument("--adapter_path", default=None, help="Trained LoRA adapter; also sample the base model.")
    p.add_argument("--scale", choices=SCALE_TO_MODEL, default="600m")
    p.add_argument("--source", choices=SOURCE_TO_CODE, default="en")
    p.add_argument("--k", type=int, default=12, help="Samples per group (training default 12).")
    p.add_argument("--temperature", type=float, default=1.2, help="Sampling temperature (training default 1.2).")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Training default 64.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def sample_group(model, tokenizer, src, tgt_token_id, k, temperature, max_new_tokens):
    """K sampled hypotheses, as in train_step."""
    inputs = tokenizer(src, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, forced_bos_token_id=tgt_token_id, do_sample=True,
                                  num_return_sequences=k, max_new_tokens=max_new_tokens,
                                  temperature=temperature)
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def beam_translation(model, tokenizer, src, tgt_token_id):
    inputs = tokenizer(src, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, forced_bos_token_id=tgt_token_id, num_beams=5, max_new_tokens=128)
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


def score_group(hyps):
    """Per-hypothesis IPD reward (unmarked -> 0, as IPDRewarder) and western/eastern counts."""
    rows = [corpus_ipd([h]) for h in hyps]
    rewards = np.array([r["mean_sentence_ipd"] for r in rows], dtype=np.float32)
    return rewards, [r["western"] for r in rows], [r["eastern"] for r in rows]


def group_stats(hyps, rewards, west, east):
    std = float(rewards.std())
    return {"reward_mean": float(rewards.mean()), "reward_std": std,
            "distinct": len(set(hyps)), "k": len(hyps),
            "with_western": sum(w > 0 for w in west), "with_eastern": sum(e > 0 for e in east),
            "unmarked": sum(w + e == 0 for w, e in zip(west, east)),
            "no_signal": std == 0.0}


def print_group(label, hyps, rewards, west, east):
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)  # as in train_step
    print(f"\n  [{label}]")
    print(f"  {'#':>2}  {'reward':>6}  {'W':>2} {'E':>2}  {'adv':>6}  hypothesis")
    for i, (h, r, w, e, a) in enumerate(zip(hyps, rewards, west, east, adv), 1):
        print(f"  {i:>2}  {r:>6.3f}  {w:>2} {e:>2}  {a:>+6.2f}  {h}")
    s = group_stats(hyps, rewards, west, east)
    print(f"  reward mean {s['reward_mean']:.3f}, std {s['reward_std']:.3f} | {s['distinct']}/{s['k']} distinct | "
          f"hyps with a western form: {s['with_western']}, eastern: {s['with_eastern']}, unmarked: {s['unmarked']}")
    if s["no_signal"]:
        print("  -> all rewards are identical: advantages are 0, this group gives no learning signal")


def print_aggregate(label, stats):
    n = len(stats)
    print(f"\n  [{label}] {n} groups")
    print(f"    groups with no signal (reward std = 0):      {sum(s['no_signal'] for s in stats)}/{n}")
    print(f"    mean reward std within a group:              {np.mean([s['reward_std'] for s in stats]):.3f}")
    print(f"    mean reward:                                 {np.mean([s['reward_mean'] for s in stats]):.3f}")
    print(f"    groups where NO hypothesis has a western form: {sum(s['with_western'] == 0 for s in stats)}/{n}")
    print(f"    mean share of hypotheses with a western form: "
          f"{np.mean([s['with_western'] / s['k'] for s in stats]):.3f}")
    print(f"    mean share of fully unmarked hypotheses:      "
          f"{np.mean([s['unmarked'] / s['k'] for s in stats]):.3f}")
    print(f"    mean distinct hypotheses per group:           {np.mean([s['distinct'] for s in stats]):.1f}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.from_data:
        data_dir = resolve_data_dir(args.dataset, args.source, args.data_dir)
        df = pd.read_parquet(os.path.join(data_dir, LANG, f"{args.split}.parquet"))
        sentences = df.sample(n=min(args.from_data, len(df)), random_state=args.seed or 0)["src"].tolist()
    else:
        sentences = args.sentence or DEFAULT_SENTENCES

    model_name = SCALE_TO_MODEL[args.scale]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = SOURCE_TO_CODE[args.source]
    tgt_token_id = tokenizer.convert_tokens_to_ids(LANG)

    model = load_vanilla_model(model_name)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path).eval()

    def systems():
        """(label, context manager) pairs: the trained model and/or the base model."""
        if args.adapter_path:
            yield "trained", contextlib.nullcontext()
            yield "base (adapter disabled)", model.disable_adapter()
        else:
            yield "base", contextlib.nullcontext()

    stats = {}
    for si, src in enumerate(sentences):
        if not args.from_data:
            print(f"\n{'=' * 78}\nSOURCE: {src}")
        for label, ctx in systems():
            with ctx:
                hyps = sample_group(model, tokenizer, src, tgt_token_id, args.k, args.temperature,
                                    args.max_new_tokens)
                if not args.from_data:
                    print(f"\n  beam-5 ({label}): {beam_translation(model, tokenizer, src, tgt_token_id)}")
            rewards, west, east = score_group(hyps)
            if args.from_data:
                stats.setdefault(label, []).append(group_stats(hyps, rewards, west, east))
            else:
                print_group(f"{label}: {args.k} samples at T={args.temperature}", hyps, rewards, west, east)
        if args.from_data and (si + 1) % 10 == 0:
            print(f"  ... {si + 1}/{len(sentences)} sentences", flush=True)

    if args.from_data:
        print(f"\nAggregate over {len(sentences)} sentences from {args.dataset}/{args.split} "
              f"(K={args.k}, T={args.temperature}, max_new_tokens={args.max_new_tokens})")
        for label, s in stats.items():
            print_aggregate(label, s)


if __name__ == "__main__":
    main()
