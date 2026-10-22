#!/bin/bash
# Runs one of the repo's Python scripts on a single GPU inside the uv environment.
#
# Usage (from the repo root, after a one-off `uv sync` on the login node):
#   sbatch --job-name=train-eus slurm/run.sbatch main.py --lang eus_Latn --scale 600m --n_epochs 1
#
# Everything after the script name is passed through to it. Override any #SBATCH default
# on the command line (e.g. --time, --partition). The job inherits the submitting shell's
# environment, so export HF_TOKEN first (COMET-Kiwi is gated) unless a token is already
# stored with `uv run huggingface-cli login`.
#
#SBATCH --job-name=rfrl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Less fragmentation of the GPU memory cache during beam search on 11 GB cards.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "commit $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
echo "command: python $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

uv run --frozen python "$@"
