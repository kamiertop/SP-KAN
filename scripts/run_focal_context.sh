#!/usr/bin/env bash
set -euo pipefail

# FocalNet/SegNeXt-inspired context replacement at TransH3/TransH4.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spkan-uv-cache}" uv run train.py \
  --context focal \
  --dataset_names SIRST3 \
  --save ./runs/focal_context \
  --auto_test \
  "$@"
