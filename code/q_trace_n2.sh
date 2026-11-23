#!/bin/bash
# marker-aware N-CMAPSS queue (safe to run several in parallel). $1 experts, $2 tag, $3 seeds, $4 extra args
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
mkdir -p ../results/trace_n/$2
for seed in $3; do for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  mk=../results/trace_n/$2/${ds}_s${seed}.started
  [ -f ../results/trace_n/$2/${ds}_s${seed}.npz ] && continue
  ( set -o noclobber; echo $$ > $mk ) 2>/dev/null || continue
  until [ $(free -g | awk '/^Mem|^内存/{print $7}') -ge 20 ]; do sleep 30; done
  $PY -u run_trace_ncmapss.py $ds $seed --experts $1 --tag $2 $4 2>&1 | grep --line-buffered -v Warn
done; done
echo QUEUE_DONE
