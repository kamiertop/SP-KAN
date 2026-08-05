#!/usr/bin/env bash
set -euo pipefail

# TGRS-inspired adaptive focal residual on top of TABDS.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spkan-uv-cache}" uv run train.py \
  --loss_focal_weight 0.5 \
  --dataset_names SIRST3 \
  --save ./runs/adaptive_focal_tabds \
  --auto_test \
  "$@"
