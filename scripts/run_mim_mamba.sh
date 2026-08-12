#!/usr/bin/env bash
set -euo pipefail

# MiM-ISTD local/global state-space variant.
python train.py --model_names SP_KAN --dataset_names SIRST3 \
  --mamba_branch --loss_name bce "$@"
