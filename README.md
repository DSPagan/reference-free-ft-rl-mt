# Monolingual Reinforcement Learning Fine-Tuning for MT

Code to reproduce the GRPO training and evaluation behind **"Reference-Free
Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective."** We fine-tune
NLLB-200 (600M/1.3B) with Group Relative Policy Optimization using a *reference-free*
reward: a hybrid of LaBSE cosine similarity and COMET-Kiwi (source+hypothesis only, no
reference translation used at training time).

On top of that reproduction we add a dialectal reward for English/Spanish → Catalan:
AVLizador's Índex de Puresa Dialectal (IPD), the share of dialectally marked words that
are Valencian (AVL) rather than Eastern Catalan forms, to steer NLLB's `cat_Latn` output
towards Valencian. See [REPRODUCTION.md](REPRODUCTION.md) for our experiment log.

## Repository layout

```
src/                  GRPO training package
  data.py               dataset registry (FLORES-200, UJI): data dirs and default splits
  models.py             4-bit NLLB-200 base + rank-16 LoRA adapter loading
  reward.py             the reference-free hybrid reward (LaBSE + COMET-Kiwi)
  ipd_reward.py         the dialectal IPD reward and corpus-level IPD metric
  temperature.py        adaptive sampling-temperature controller (never updated during
                        training, so the temperature stays at --temp_initial)
  trainer.py            the GRPO update step and training loop
metrics/              Evaluation package
  chrf.py                chrF++ scoring (the paper's primary metric)
  decoding.py            fair-decoding protocol (3-setting grid, dev-tuned per system)
main.py                Training entry point -- trains one adapter per invocation
generate_translations.py   Inference entry point -- loads an adapter, decodes, scores
                        chrF++ against vanilla
decoding_diagnostic.py Scores vanilla under every decoding setting, 4-bit and fp32
prepare_data.py        Builds the FLORES-200 parquet files the above expect
prepare_uji.py         Builds the train/dev/devtest parquet files from the UJI va-en corpus
third_party/AVLizador  git submodule (pinned commit) providing the IPD
slurm/run.sbatch       SLURM job wrapper for any of the scripts
pyproject.toml, uv.lock   Python 3.10 environment, every dependency pinned
```

## Setup

The environment is a [uv](https://docs.astral.sh/uv/) project pinned to Python 3.10 and
the CUDA 12.1 `torch` build. It needs Linux x86_64 and an NVIDIA GPU whose driver
supports CUDA 12.1. `uv.lock` pins every transitive dependency; `uv sync` builds an
isolated `.venv` with its own Python 3.10, independent of the system Python. The IPD
code comes from the AVLizador submodule, so clone with `--recurse-submodules` (or run
`git submodule update --init` in an existing clone, and again after a pull that moves
the submodule):

```bash
git clone --recurse-submodules git@github.com:DSPagan/reference-free-ft-rl-mt.git
cd reference-free-ft-rl-mt
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Running on the SLURM cluster

Run `uv sync` once on the login node, then submit any script with
[slurm/run.sbatch](slurm/run.sbatch) (one GPU per job; stdout/stderr go to
`logs/<job-name>-<job-id>.out`):

```bash
export HF_TOKEN=...   # COMET-Kiwi is gated
jid=$(sbatch --parsable --job-name=train-eus slurm/run.sbatch \
    main.py --lang eus_Latn --scale 600m --n_epochs 1 --seed 42 --output_dir results/eus_Latn_600m)
sbatch --dependency=afterok:$jid --job-name=eval-eus slurm/run.sbatch \
    generate_translations.py --adapter_path results/eus_Latn_600m/adapters_best/eus_Latn \
    --lang eus_Latn --scale 600m --setting "default(beam5,lp1.0)" --output_dir results/grpo/600m
```

The second job starts only if training succeeds. Check jobs with `squeue -u $USER` and
follow one with `tail -f logs/train-eus-<jobid>.out`.

COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`) and LaBSE are pulled from the Hugging Face Hub
on first run and cached. COMET-Kiwi is gated: request access on its model page and
export `HF_TOKEN` before training.

## Data

The prepared [FLORES-200](https://github.com/facebookresearch/flores) files (CC-BY-SA
4.0) are committed, so no download is needed: English source for 27 target languages
(`data/flores200_eng/{lang}/{dev,devtest}.parquet`, including `cat_Latn`) and Spanish
source for `cat_Latn` (`data/flores200_spa/`). To rebuild them, `prepare_data.py`
fetches the gated `facebook/flores` dataset from the Hugging Face Hub (request access
and export `HF_TOKEN` first):

```bash
uv run python prepare_data.py --source eng_Latn
uv run python prepare_data.py --source spa_Latn --languages cat_Latn
```

The FLORES-200 `cat_Latn` references are Eastern (IEC) Catalan, IPD ≈ 0.19, so chrF++
against them does not reward Valencian output.

The UJI Valencian–English news corpus (`data/va_en_uji/*.jsonl`) is converted by
`prepare_uji.py` into `data/uji_eng/cat_Latn/{train,dev,devtest}.parquet` (English source,
Valencian references, split by news article) and selected with `--dataset uji`. See
[ADAPTACION_UJI.md](ADAPTACION_UJI.md) for the details and a full example.

## Training

One language pair per invocation (`--source en` or `es`):

```bash
uv run python main.py --lang eus_Latn --scale 600m                          # paper's hybrid reward
uv run python main.py --lang cat_Latn --scale 600m --source es --reward ipd   # Valencian (IPD) reward
```

`--reward ipd` scores each sampled translation with the IPD alone (0 when it contains no
dialectally marked word) and never looks at the source sentence.

Defaults match the paper's recipe (LoRA r=16/α=32/dropout=0.05 on `q_proj`/`v_proj`,
4-bit NF4 double-quant, AdamW lr=5e-6, K=12, β=0.001, 3 epochs, sampling temperature
1.2, equal 0.5 LaBSE/COMET-Kiwi reward blend). Other options: `--labse_weight`,
`--kl_beta`, `--n_epochs`, `--k`, `--seed`, ... (see `uv run python main.py --help`).

Outputs land in `results/{lang}_{scale}_{source}_{reward}/`: `adapters_best/{lang}/` (the checkpoint with
the highest intermediate chrF++, what the paper evaluates), `adapters/{lang}/`
(final-step adapter), and `training_metrics.csv` (per-step log).

## Evaluation

```bash
uv run python generate_translations.py \
    --adapter_path results/cat_Latn_600m_es_ipd/adapters_best/cat_Latn \
    --lang cat_Latn --scale 600m --source es --setting "default(beam5,lp1.0)" --ipd
```

With `--setting`, both the vanilla model and the trained adapter are decoded with that
fixed setting; `default(beam5,lp1.0)` reproduces the paper's Table 7 baseline (see
REPRODUCTION.md). Without it, a 3-setting beam-search grid is dev-tuned independently
for each system and each system's winning setting is applied to the held-out `devtest`
split. Omit `--adapter_path` to score only the vanilla model. `--ipd` adds the
corpus-level IPD (Valencian share of all marked words) of the reference and of each
system to the printed results and `{lang}_summary.json`.

## Pretrained adapters

Every adapter from the paper's main sweep and ablations is published at
[`ergar/monolingual-reinforcement-learning-mt`](https://huggingface.co/ergar/monolingual-reinforcement-learning-mt)
(see that repo's model card for the loading snippet).

## Known limitation: the KL penalty term

`src/trainer.py`'s `train_step` documents a known deviation from the textbook k3
KL-divergence estimator (the implemented term is a biased proxy, not
`exp(-Δ)+Δ-1`). Every adapter in this release and on the Hub was trained with the
formula as implemented, not a corrected one — see the docstring in `train_step` for the
derivation of why it differs, and the paper's Methodology section for the full
discussion.

## License

MIT — see [LICENSE](LICENSE).
