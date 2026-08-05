#!/usr/bin/env bash
set -euo pipefail

# Restormer-inspired gated restoration stem with synthetic infrared corruption.
# Override DATASET_DIR/DATASET_NAME and other train.py flags as needed.
DATASET_DIR="${DATASET_DIR:-./dataset}"
DATASET_NAME="${DATASET_NAME:-NUDT-SIRST}"

exec python train.py \
  --dataset_dir "$DATASET_DIR" \
  --dataset_name "$DATASET_NAME" \
  --restoration_branch \
  --noise_consistency_weight "${NOISE_CONSISTENCY_WEIGHT:-0.5}" \
  --noise_warmup_epochs "${NOISE_WARMUP_EPOCHS:-5}" \
  --noise_gaussian_std "${NOISE_GAUSSIAN_STD:-0.04}" \
  --noise_stripe_prob "${NOISE_STRIPE_PROB:-0.25}" \
  --noise_dead_pixel_prob "${NOISE_DEAD_PIXEL_PROB:-0.01}" \
  "$@"
