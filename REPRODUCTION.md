# Reproduction log

Running record of our attempt to reproduce the paper's Experiment A (English source,
FLORES-200 devtest, chrF++). Raw outputs (translations, `*_summary.json`,
`training_metrics.csv`, logs) live under `results/`; this file records what was run,
where, and how it compares to the paper.

## Environments

| Id | Where | Python | torch | Notes |
|---|---|---|---|---|
| A | Colab, Tesla T4 | 3.13 | Colab preinstalled | Ad hoc install into Colab's own Python (`numpy` 1.26.4 built from source, `accelerate==0.27.2`, `USE_FLAX=0`). Superseded by B. |
| B | Colab, Tesla T4 | 3.10.22 | 2.3.0+cu121 | Isolated venv installed from `requirements.lock`, since replaced by `uv.lock`, which pins the same 77 package versions (`uv sync`, see README). |
| C | Server `abaco` (SLURM), NVIDIA GeForce RTX 2080 Ti 11 GB (8 cards, 1 per job), driver 575.57.08 | 3.10 (uv) | 2.3.0+cu121 | `uv sync` from `uv.lock`; jobs via `slurm/run.sbatch`. Same GPU architecture as the T4 (Turing, bf16 emulated), so results should be comparable with B. Reference environment from 2026-10-23 on. |

## Baseline (vanilla NLLB-200, 4-bit NF4, fair decoding)

Decoding setting is the winner of the 3-setting dev-tuning grid (`metrics/decoding.py`).

| Date | Env | Lang | Scale | Decoding | chrF++ | Paper (Tab. 7) | Δ vs paper |
|---|---|---|---|---|---|---|---|
| 2026-10-20 | A | eus_Latn | 600m | lp2.0,cap256 | 49.445 | 47.75 | +1.70 |
| 2026-10-20 | B | eus_Latn | 600m | lp2.0,cap256 | 49.307 | 47.75 | +1.56 |

A vs B differ by only 0.14 chrF++, so library versions do not explain the gap to the
paper. The paper itself reports three different eus_Latn 600M baselines (Table 1: 52.69,
Table 7: 47.75, Table 3 / Appendix J: 47.64).

### Decoding / precision diagnostic

`decoding_diagnostic.py`, env B, 2026-10-22: vanilla eus_Latn 600m on the full devtest
under each setting, no dev-tuning.

| Setting | Beam | Length penalty | Max new tokens | 4-bit | fp32 | Paper |
|---|---|---|---|---|---|---|
| `default(beam5,lp1.0)` | 5 | 1.0 | 128 | **47.740** | 47.886 | **Table 7: 47.75** |
| `trainer_eval(beam4,lp1.0,cap128)` | 4 | 1.0 | 128 | 47.584 | OOM | Table 3 / App. J: 47.64 |
| `lp3.0,cap256,beam8` | 8 | 3.0 | 256 | 48.607 | OOM | |
| `lp2.0,cap256` (dev-tuned winner) | 5 | 2.0 | 256 | 49.307 | 49.964 | |

The two missing fp32 rows ran out of memory on the T4 (beam 8 / batch 16) and were not
rerun: they do not change the conclusions.

**Conclusions**

- The paper's Table 7 baseline is reproduced (47.740 vs 47.75) with the default beam
  search (beam 5, length penalty 1.0, 128 new tokens) on the 4-bit model.
- Table 3 / Appendix J (47.64) match the trainer's intermediate-eval setting (beam 4).
- Our earlier +1.56 gap was entirely the repo's dev-tuned decoding, which picks
  `lp2.0,cap256` for eus_Latn.
- 4-bit quantization costs only 0.15 chrF++ at the default setting.
- No setting reproduces Table 1 (52.69).

To compare GRPO against the paper's Table 7 (+2.44 for eus_Latn 600M), evaluate with
`generate_translations.py --setting "default(beam5,lp1.0)"`, which decodes both systems
with that fixed setting.

## GRPO training

| Date | Env | Lang | Scale | Run | Result |
|---|---|---|---|---|---|
| 2026-10-20 | A | eus_Latn | 600m | Smoke test: 20 sentences, 1 epoch, eval at step 10 | Pipeline runs end to end; ~6–7 s/step on T4; intermediate dev chrF++ 47.16 (sanity check only) |
| 2026-10-20 | B | eus_Latn | 600m | Same smoke test | Pipeline runs end to end; intermediate dev chrF++ 46.86 (sanity check only) |

## Pending

Scope: a single language, eus_Latn (600m, English source).

- [ ] GRPO eus_Latn 600m, 1 epoch, seed 42, LaBSE weight 0.5 (paper's equal
      weighting, now the default; the original code looked up per-language weights from
      a reward_weights.csv, 0.25 for eus_Latn, which the smoke tests above used) +
      devtest evaluation vs baseline (env B)
- [x] `decoding_diagnostic.py` for eus_Latn 600m (env B): Table 7 baseline reproduced
- [ ] Evaluate the published `exp_a_600m_en_eus_Latn` adapter from
      `ergar/monolingual-reinforcement-learning-mt` (env B)
- [ ] GRPO eus_Latn 600m, 3 epochs (paper recipe; Table 7 reports +2.44 chrF++)
