#!/bin/bash
# distilled students for every finished N-CMAPSS LOUO run (loops until all 21 exist)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
while true; do
  n=0
  for f in ../results/trace_n/trace_louo/DS0?_s?.npz; do
    b=$(basename $f .npz); ds=${b%_s*}; seed=${b#*_s}
    [ -f ../results/trace_n/trace_louo/${b}_eval.json ] || $PY eval_ncmapss.py trace_louo $ds $seed > /dev/null 2>&1
    [ -f ../results/trace_n/trace_louo/${b}_student.json ] && { n=$((n+1)); continue; }
    $PY distill.py ncmapss $ds $seed 2>&1 | grep -v Warn
  done
  [ $n -ge 21 ] && break
  sleep 120
done
echo QUEUE_DONE
