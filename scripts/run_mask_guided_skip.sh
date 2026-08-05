#!/usr/bin/env bash
set -euo pipefail

# Mask2Former-inspired soft gating on the deepest SP-KAN skip connection.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spkan-uv-cache}" uv run train.py \
  --mask_guided \
  --projection interp \
  --dataset_names SIRST3 \
  --save ./runs/mask_guided_skip \
  --auto_test \
  "$@"
