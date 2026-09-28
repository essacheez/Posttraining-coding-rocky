#!/bin/bash
#SBATCH --job-name=rlvr-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/rlvr_train_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"  # submit from the repo root: mkdir -p logs && sbatch RLVR/training/train_rlvr.sbatch

.venv/bin/accelerate launch --num_processes 2 --mixed_precision bf16 RLVR/training/train_rlvr.py
