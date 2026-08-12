#!/usr/bin/env bash
set -euo pipefail

# MiM-ISTD local/global state-space variant.
python train.py --model_names SP_KAN --dataset_names SIRST3 \
  --mamba_branch --loss_name bce \
  --cross_view --cross_view_noise_std 0.03 \
  --cross_view_topk 0.2 --cross_view_consistency_weight 0.1 "$@"
