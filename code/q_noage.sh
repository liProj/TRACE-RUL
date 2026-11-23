#!/bin/bash
# age-free faithful AIDGN (ablation), reverse order so it does not collide with q_faithful.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for sub in FD004 FD003 FD002 FD001; do
 for seed in 4 3 2 1 0; do
  [ -f ../results/runs/aidgn_noage/${sub}_s${seed}.json ] && continue
  $PY train_aidgn.py $sub $seed --tag aidgn_noage --no_age
 done
done
echo QUEUE_DONE
