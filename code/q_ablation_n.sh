#!/bin/bash
cd "$(dirname "$0")"
until grep -q QUEUE_DONE ../logs/q_trace_n.log; do sleep 60; done
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  [ -f ../results/trace_n/trace_noage/${ds}_s0.npz ] && continue
  $PY run_trace_ncmapss.py $ds 0 --experts aidgna,cnn,gru,lgbm --tag trace_noage --no_age 2>&1 | grep -v Warn
done
echo QUEUE_DONE
