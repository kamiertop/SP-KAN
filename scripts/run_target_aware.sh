#!/usr/bin/env bash
set -euo pipefail

# TABDS innovation experiment. Additional train.py arguments can be appended.
# Example:
#   bash scripts/run_target_aware.sh --epochs 100 --max_train_steps 5
# Keep uv's cache in a writable location on managed/CI machines. Users can
# override it with UV_CACHE_DIR or replace the launcher with `python train.py`.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spkan-uv-cache}" uv run train.py \
  --loss_name target_aware \
  --dataset_names SIRST3 \
  --save ./runs/tabds \
  --auto_test \
  "$@"
