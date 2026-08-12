#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: target-aware loss was removed; run the BCE baseline.
exec "$(dirname "$0")/run_mim_mamba.sh" "$@"
