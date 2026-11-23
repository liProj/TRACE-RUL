#!/bin/bash
# faithful AIDGN on N-CMAPSS, marker-aware (several instances may run in parallel); seed-major order
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
mkdir -p ../results/runs/aidgn_n
for seed in 0 1 2; do for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  [ -f ../results/runs/aidgn_n/${ds}_s${seed}.json ] && continue
  ( set -o noclobber; echo $$ > ../results/runs/aidgn_n/${ds}_s${seed}.started ) 2>/dev/null || continue
  until [ $(free -g | awk '/^Mem|^内存/{print $7}') -ge 20 ]; do sleep 30; done
  $PY train_aidgn_n.py $ds $seed 2>&1 | grep -v Warn
done; done
echo QUEUE_DONE
