#!/usr/bin/env bash
set -euo pipefail

# Cross-view ISTD: clean/noisy views, channel alignment, and Top-K background fusion.
python train.py --model_names SP_KAN --dataset_names SIRST3 \
  --cross_view --cross_view_noise_std 0.03 --cross_view_topk 0.2 "$@"
