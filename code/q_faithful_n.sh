#!/bin/bash
# faithful AIDGN on N-CMAPSS, 3 seeds; starts after the C-MAPSS age-free queue finished
cd "$(dirname "$0")"
until grep -q QUEUE_DONE ../logs/q_noage.log; do sleep 60; done
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for seed in 0 1 2; do for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  [ -f ../results/runs/aidgn_n/${ds}_s${seed}.json ] && continue
  $PY train_aidgn_n.py $ds $seed 2>&1 | grep -v Warn
done; done
echo QUEUE_DONE
