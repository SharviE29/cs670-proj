#!/bin/bash
# Unity (Slurm) driver for the GPU / long steps. Edit partition, account and paths.
# Usage:
#   sbatch --export=ALL,STEP=crops     data_pipeline/slurm/build.sbatch
#   sbatch --export=ALL,STEP=qwen      data_pipeline/slurm/build.sbatch
#   sbatch --export=ALL,STEP=bgaudit   data_pipeline/slurm/build.sbatch
#   sbatch --array=0-15 --export=ALL,STEP=render,SPLIT=train data_pipeline/slurm/build.sbatch
#   sbatch --export=ALL,STEP=checks    data_pipeline/slurm/build.sbatch
# Internet is only needed for `backgrounds fetch` and model/dataset downloads: run those on a
# login node first (models land in HF_HOME), then everything here runs offline.
#SBATCH --job-name=cira-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export CIRA_DATA_ROOT="${CIRA_DATA_ROOT:?set CIRA_DATA_ROOT to a /work or /scratch path}"
export HF_HOME="${HF_HOME:-$CIRA_DATA_ROOT/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
case "${STEP}" in
  crops)   python -m data_pipeline.crops extract ;;
  qwen)    python -m data_pipeline.crops qwen-check ;;
  bgaudit) python -m data_pipeline.backgrounds audit ;;
  render)  python -m data_pipeline.render --split "${SPLIT}" --shard "${SLURM_ARRAY_TASK_ID:-0}" \
             --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" ;;
  checks)  python -m data_pipeline.audit checks --use_saved && CIRA_TEST_FULL=1 pytest -q tests ;;
  *) echo "unknown STEP=${STEP}"; exit 2 ;;
esac
