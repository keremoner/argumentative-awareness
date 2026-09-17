#!/usr/bin/env bash
# Sequential chain for the switching studies. Each run is resumable: rerun this
# script and finished cells are skipped. B first (the main S2 study), then the
# C contrasts in the spec's order.
set -u
cd "$(dirname "$0")/../.."
PY=python
for g in B C1 C2 C3_B_amnesic; do
  echo "=== $(date +%H:%M:%S) starting $g"
  $PY experiments/switching/run.py --grid "experiments/switching/grids/$g.json" || echo "!!! $g failed"
  echo "=== $(date +%H:%M:%S) finished $g"
done
echo "=== $(date +%H:%M:%S) chain complete"
