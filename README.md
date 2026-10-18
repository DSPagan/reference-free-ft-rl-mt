# Monolingual Reinforcement Learning Fine-Tuning for MT

Code to reproduce the GRPO training and evaluation behind **"Reference-Free
Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective."** We fine-tune
NLLB-200 (600M/1.3B) with Group Relative Policy Optimization using a *reference-free*
reward: a per-language-calibrated hybrid of LaBSE cosine similarity and COMET-Kiwi
(source+hypothesis only, no reference translation used at training time).

This is a minimal reproduction repo — training pipeline and evaluation only, not the
full research/analysis codebase behind the paper's tables and figures.

## Repository layout

```
src/                  GRPO training package
  models.py             4-bit NLLB-200 base + rank-16 LoRA adapter loading
  reward.py             the reference-free reward (hybrid/semantic/comet/random modes)
  temperature.py        adaptive sampling-temperature controller
  trainer.py            the GRPO update step and training loop
  calibration.py         per-language reward-weight calibration (Phase 0)
metrics/              Evaluation package
  chrf.py                chrF++ scoring (the paper's primary metric)
  decoding.py            fair-decoding protocol (3-setting grid, dev-tuned per system)
main.py                Training entry point -- trains one adapter per invocation
generate_translations.py   Inference entry point -- loads an adapter, fair-decodes,
                        scores chrF++ against vanilla
prepare_data.py        Builds the FLORES-200 parquet files the above expect
reward_weights.csv     Calibrated per-language LaBSE/COMET-Kiwi blend (already computed)
```

## Setup

```bash
docker build -t monolingual-rl-mt .
docker run --gpus '"device=0"' -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/workspace/.cache/huggingface -it monolingual-rl-mt bash
```

Without Docker: `pip install -r requirements.txt` (CUDA 12.1 required for the pinned
`torch` build). COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`) and LaBSE are pulled from the
Hugging Face Hub on first run and cached.

## Data

Download the [FLORES-200 dataset](https://github.com/facebookresearch/flores) and point
`prepare_data.py` at the raw dump:

```bash
python3 prepare_data.py --source eng_Latn --languages arb_Arab eus_Latn khm_Khmr pol_Latn swh_Latn zho_Hant
python3 prepare_data.py --source spa_Latn --languages arb_Arab eus_Latn khm_Khmr pol_Latn swh_Latn zho_Hant
```

This writes `data/flores200_eng/{lang}/{dev,devtest}.parquet` (and `flores200_spa/` for
Spanish-source). Omit `--languages` to prepare the full 26-language set used in the paper.

## Training

One language per invocation:

```bash
python3 main.py --lang arb_Arab --scale 600m --source en
```

Defaults match the paper's recipe (LoRA r=16/α=32/dropout=0.05 on `q_proj`/`v_proj`,
4-bit NF4 double-quant, Adam lr=5e-6, K=12, β=0.001, 3 epochs, temperature 1.2→1.6). The
hybrid-reward weight is looked up automatically from `reward_weights.csv`, or override
with `--labse_weight`. Key options:

| Flag | Purpose |
|---|---|
| `--reward {hybrid,semantic,comet,random}` | Reward-component ablation / random-reward control |
| `--advantage_type {grpo,rloo}` | GRPO (std-normalized) vs. RLOO advantage estimator |
| `--init_adapter_path PATH` | Continue from an existing adapter (SFT→GRPO cascade) |
| `--kl_beta`, `--n_epochs`, `--k`, `--seed`, ... | See `python3 main.py --help` |

For a full sweep, loop over languages from a shell script:

```bash
for lang in arb_Arab eus_Latn khm_Khmr pol_Latn swh_Latn zho_Hant; do
    python3 main.py --lang "$lang" --scale 600m --source en
done
```

Outputs land in `results/{lang}_{scale}_{source}/`: `adapters_best/{lang}/` (the
checkpoint with the highest intermediate chrF++, what the paper evaluates),
`adapters/{lang}/` (final-step adapter), and `training_metrics.csv` (per-step log).

## Evaluation

```bash
python3 generate_translations.py \
    --adapter_path results/arb_Arab_600m_en/adapters_best/arb_Arab \
    --lang arb_Arab --scale 600m --source en
```

Dev-tunes a 3-setting beam-search grid independently for the vanilla model and the
trained adapter (see the paper's "Decoding protocol" section for why this matters —
comparing both systems under one fixed setting is biased), applies each system's own
winning setting to the held-out `devtest` split, and reports the chrF++ gain.

## Pretrained adapters

Every adapter from the paper's main sweep and ablations is published at
[`ergar/monolingual-reinforcement-learning-mt`](https://huggingface.co/ergar/monolingual-reinforcement-learning-mt)
(see that repo's model card for the loading snippet) — you don't need to retrain
everything from scratch to reproduce the evaluation numbers.

## Known limitation: the KL penalty term

`src/trainer.py`'s `train_step` documents a known deviation from the textbook k3
KL-divergence estimator (the implemented term is a biased proxy, not
`exp(-Δ)+Δ-1`). Every adapter in this release and on the Hub was trained with the
formula as implemented, not a corrected one — see the docstring in `train_step` for the
derivation of why it differs, and the paper's Methodology section for the full
discussion.

## License

MIT — see [LICENSE](LICENSE).
