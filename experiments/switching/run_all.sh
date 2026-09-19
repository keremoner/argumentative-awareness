#!/usr/bin/env bash
# Sequential chain for the switching studies. Each run is resumable: rerun this
# script and finished cells are skipped.
#   A     S1 speaker vs switching L1 (offline baseline)
#   B     S2 speaker with an exact replica of the switching L1 (feedback; the core study)
#   Fvig  S2 modelling a vigilant L1, audience actually vigilant   (Fang dyad; read the `vig` column)
#   Fcred S2 modelling a credulous L1, audience actually credulous (Fang dyad; read the `cred` column)
# All four share seed_base, so sim i sees the same observation stream in every study.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
for g in A B Fvig Fcred; do
  echo "=== $(date +%H:%M:%S) starting $g"
  $PY experiments/switching/run.py --grid "experiments/switching/grids/$g.json" || echo "!!! $g failed"
  echo "=== $(date +%H:%M:%S) finished $g"
done
echo "=== $(date +%H:%M:%S) chain complete"
