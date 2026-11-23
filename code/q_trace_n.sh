#!/bin/bash
# N-CMAPSS TRACE cross-fitting queue. $1 = experts, $2 = tag, $3 = seeds
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for seed in $3; do for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  [ -f ../results/trace_n/$2/${ds}_s${seed}.npz ] && continue
  avail=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$avail" -lt 20 ] && { echo DISK_LOW; exit 1; }
  until [ $(free -g | awk '/^Mem|^内存/{print $7}') -ge 20 ]; do sleep 30; done
  $PY -u run_trace_ncmapss.py $ds $seed --experts $1 --tag $2 $4 2>&1 | grep --line-buffered -v Warn
done; done
echo QUEUE_DONE
