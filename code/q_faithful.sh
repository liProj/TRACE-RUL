#!/bin/bash
# faithful AIDGN reruns (+ age-free variant for the ablation), sequential
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for tag in aidgn aidgn_noage; do
 for sub in FD001 FD002 FD003 FD004; do
  for seed in 0 1 2 3 4; do
   [ -f ../results/runs/$tag/${sub}_s${seed}.json ] && continue
   extra=""; [ $tag = aidgn_noage ] && extra="--no_age"
   $PY train_aidgn.py $sub $seed --tag $tag $extra
  done
 done
done
echo QUEUE_DONE
