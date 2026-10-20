# Reproduction log

Running record of our attempt to reproduce the paper's Experiment A (English source,
FLORES-200 devtest, chrF++). Raw outputs (translations, `*_summary.json`,
`training_metrics.csv`, logs) live under `results/`; this file records what was run,
where, and how it compares to the paper.

## Environments

| Id | Where | Python | torch | Notes |
|---|---|---|---|---|
| A | Colab, Tesla T4 | 3.13 | Colab preinstalled | Ad hoc install into Colab's own Python (`numpy` 1.26.4 built from source, `accelerate==0.27.2`, `USE_FLAX=0`). Superseded by B. |
| B | Colab, Tesla T4 | 3.10.22 | 2.3.0+cu121 | Isolated venv installed from `requirements.lock` (see README). Reference environment from here on. |

## Baseline (vanilla NLLB-200, 4-bit NF4, fair decoding)

Decoding setting is the winner of the 3-setting dev-tuning grid (`metrics/decoding.py`).

| Date | Env | Lang | Scale | Decoding | chrF++ | Paper (Tab. 7) | Δ vs paper |
|---|---|---|---|---|---|---|---|
| 2026-10-20 | A | eus_Latn | 600m | lp2.0,cap256 | 49.445 | 47.75 | +1.70 |
| 2026-10-20 | B | eus_Latn | 600m | lp2.0,cap256 | 49.307 | 47.75 | +1.56 |

A vs B differ by only 0.14 chrF++, so library versions do not explain the gap to the
paper. The paper itself reports three different eus_Latn 600M baselines (Table 1: 52.69,
Table 7: 47.75, Table 3 / Appendix J: 47.64). Leading hypothesis: the paper's baseline
uses the default decoding setting (beam 5, length penalty 1.0) rather than the dev-tuned
one; `decoding_diagnostic.py` tests this (pending).

## GRPO training

| Date | Env | Lang | Scale | Run | Result |
|---|---|---|---|---|---|
| 2026-10-20 | A | eus_Latn | 600m | Smoke test: 20 sentences, 1 epoch, eval at step 10 | Pipeline runs end to end; ~6–7 s/step on T4; intermediate dev chrF++ 47.16 (sanity check only) |
| 2026-10-20 | B | eus_Latn | 600m | Same smoke test | Pipeline runs end to end; intermediate dev chrF++ 46.86 (sanity check only) |

## Pending

Scope: a single language, eus_Latn (600m, English source).

- [ ] GRPO eus_Latn 600m, 1 epoch, seed 42, `--labse_weight 0.5` (paper's equal
      weighting; the code's default from `reward_weights.csv` is 0.25 for eus_Latn) +
      devtest evaluation vs baseline (env B)
- [ ] `decoding_diagnostic.py` for eus_Latn 600m (env B)
- [ ] Evaluate the published `exp_a_600m_en_eus_Latn` adapter from
      `ergar/monolingual-reinforcement-learning-mt` (env B)
- [ ] GRPO eus_Latn 600m, 3 epochs (paper recipe; Table 7 reports +2.44 chrF++)
