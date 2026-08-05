#!/usr/bin/env bash
set -euo pipefail

# Paper-fidelity control: compare --kan_grid_size 12 against 5 with all else
# fixed. This branch is a reproduction control, not a new-method claim.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spkan-uv-cache}" uv run train.py \
  --kan_grid_size 12 \
  --dataset_names SIRST3 \
  --save ./runs/grid12_fidelity \
  --auto_test \
  "$@"
