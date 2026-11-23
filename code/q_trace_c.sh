#!/bin/bash
# C-MAPSS TRACE cross-fitting queue. $1 = experts, $2 = tag
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for seed in 0 1 2 3 4; do for sub in FD001 FD002 FD003 FD004; do
  [ -f ../results/trace/$2/${sub}_s${seed}.npz ] && continue
  avail=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$avail" -lt 20 ] && { echo DISK_LOW; exit 1; }
  $PY run_trace_cmapss.py $sub $seed --experts $1 --tag $2 2>&1 | grep -v Warn
done; done
echo QUEUE_DONE
