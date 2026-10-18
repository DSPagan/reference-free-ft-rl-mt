"""
Reward-weight calibration: for each target language, builds a synthetic 6-level quality
cline per FLORES-200 dev sentence (the human reference as best, four progressively
corrupted degradations, and one off-topic candidate as worst), scores every candidate
under a grid of LaBSE/COMET-Kiwi weight splits, and picks the weight that best
discriminates quality (most negative Pearson r between reward score and cline level).
Writes reward_weights.csv (lang -> labse_weight), consumed by main.py's --labse_weight
auto-lookup. Already-computed weights ship in this repo (reward_weights.csv); re-run
this only if you want to reproduce the calibration itself.

Inference-only, no GRPO training, runs in well under an hour for all languages on one GPU.

Usage:
    python3 -m src.calibration --eval_data_dir data/flores200_eng --output reward_weights.csv
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_GRID = [1.00, 0.75, 0.50, 0.25, 0.00]  # LaBSE weight; COMET-Kiwi weight = 1 - w

# Languages lacking reliable whitespace word boundaries: corruption is always applied at
# the character level for these (a per-sentence word-count heuristic was tried first and
# found unreliable -- e.g. Thai/Khmer sentences often contain enough incidental
# whitespace, around numbers/punctuation/loanwords, to wrongly select word-level
# corruption for most sentences).
SCRIPTIO_CONTINUA = {"jpn_Jpan", "zho_Hans", "zho_Hant", "tha_Thai", "khm_Khmr", "bod_Tibt"}


def _units(text, force_char=False):
    if not force_char:
        words = text.split()
        if len(words) > 3:
            return words, " "
    chars = list(text)
    return ["".join(chars[i:i + 2]) for i in range(0, len(chars), 2)], ""


def _degrade(text, severity, rng, force_char=False):
    """severity in {1,2,3,4}: 1 = mild corruption ... 4 = severe corruption."""
    units, sep = _units(text, force_char=force_char)
    n = len(units)
    if n < 2:
        return text
    units = units.copy()

    n_swap = max(1, int(n * 0.08 * severity))
    n_delete = max(0, int(n * 0.05 * severity))
    n_blocks = max(0, severity - 2)  # local 3-unit block shuffles, only at severity 3-4

    for _ in range(n_swap):
        i, j = rng.randrange(n), rng.randrange(n)
        units[i], units[j] = units[j], units[i]

    if n_delete and len(units) > 1:
        drop = set(rng.sample(range(len(units)), min(n_delete, len(units) - 1)))
        units = [u for i, u in enumerate(units) if i not in drop]

    for _ in range(n_blocks):
        if len(units) < 4:
            break
        i = rng.randrange(len(units) - 3)
        block = units[i:i + 3]
        rng.shuffle(block)
        units[i:i + 3] = block

    return sep.join(units)


def build_cline(df, rng, lang):
    """Returns a list of dicts: sent_id, src, level (0=best..5=worst), candidate."""
    refs, srcs = df["ref"].tolist(), df["src"].tolist()
    force_char = lang in SCRIPTIO_CONTINUA
    rows = []
    for i, (src, ref) in enumerate(zip(srcs, refs)):
        rows.append({"sent_id": i, "src": src, "level": 0, "candidate": ref})
        for sev in range(1, 5):
            rows.append({"sent_id": i, "src": src, "level": sev,
                         "candidate": _degrade(ref, sev, rng, force_char=force_char)})
        off_topic = rng.choice([j for j in range(len(refs)) if j != i])
        rows.append({"sent_id": i, "src": src, "level": 5, "candidate": refs[off_topic]})
    return rows


def score_labse(model, srcs, cands, batch_size=64):
    scores = np.zeros(len(srcs), dtype=np.float64)
    for i in range(0, len(srcs), batch_size):
        s_emb = model.encode(srcs[i:i + batch_size], convert_to_tensor=True, normalize_embeddings=True)
        c_emb = model.encode(cands[i:i + batch_size], convert_to_tensor=True, normalize_embeddings=True)
        scores[i:i + batch_size] = (s_emb * c_emb).sum(dim=1).cpu().numpy()
    return scores


def score_comet_kiwi(model, srcs, cands, batch_size=64):
    data = [{"src": s, "mt": c} for s, c in zip(srcs, cands)]
    with torch.no_grad():
        out = model.predict(data, batch_size=batch_size, gpus=1 if DEVICE == "cuda" else 0)
    return np.array(out.scores)


def parse_args():
    p = argparse.ArgumentParser(description="Reward-weight calibration")
    p.add_argument("--languages", nargs="+", required=True)
    p.add_argument("--eval_data_dir", default="data/flores200_eng")
    p.add_argument("--n_sentences", type=int, default=50)
    p.add_argument("--output", default="reward_weights.csv")
    p.add_argument("--detail_output", default="reward_weight_calibration_detail.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading LaBSE...")
    labse = SentenceTransformer("sentence-transformers/LaBSE").to(DEVICE)
    print("Loading COMET-Kiwi...")
    comet_kiwi = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    comet_kiwi.eval()

    best_rows, detail_rows = [], []
    for lang in args.languages:
        dev_path = os.path.join(args.eval_data_dir, lang, "dev.parquet")
        if not os.path.exists(dev_path):
            print(f"  [SKIP] {lang}: no dev.parquet at {dev_path}")
            continue

        print(f"\n{'=' * 20} {lang} {'=' * 20}")
        rng = random.Random(args.seed)
        full_df = pd.read_parquet(dev_path)
        df = full_df.sample(n=min(args.n_sentences, len(full_df)), random_state=args.seed).reset_index(drop=True)

        cline = build_cline(df, rng, lang)
        srcs = [r["src"] for r in cline]
        cands = [r["candidate"] for r in cline]
        levels = np.array([r["level"] for r in cline])

        labse_scores = score_labse(labse, srcs, cands)
        comet_scores = score_comet_kiwi(comet_kiwi, srcs, cands)
        torch.cuda.empty_cache()

        best_w, best_r = None, 1.0  # want most negative r (higher score -> lower level)
        for w in WEIGHT_GRID:
            reward = w * labse_scores + (1 - w) * comet_scores
            r, _ = pearsonr(reward, levels)
            detail_rows.append({"lang": lang, "labse_weight": w, "pearson_r": round(r, 4)})
            print(f"  LaBSE={w:.2f}/CometKiwi={1 - w:.2f}  r={r:+.4f}")
            if r < best_r:
                best_r, best_w = r, w

        print(f"  -> best: LaBSE weight {best_w:.2f} (r={best_r:+.4f})")
        best_rows.append({"lang": lang, "labse_weight": best_w, "pearson_r": round(best_r, 4)})

    pd.DataFrame(best_rows).to_csv(args.output, index=False)
    pd.DataFrame(detail_rows).to_csv(args.detail_output, index=False)
    print(f"\nWrote {args.output} ({len(best_rows)} languages)")
    print(f"Wrote {args.detail_output} (full weight-grid detail)")


if __name__ == "__main__":
    main()
