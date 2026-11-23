#!/bin/bash
# pre-registered ablations, started after the main C-MAPSS queue (nets + LightGBM, no TabPFN; compared with
# the same expert set of the main run)
cd "$(dirname "$0")"
# (started in parallel with the main queue; seed 0 only, 2026-10-28)
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for seed in 0; do for sub in FD001 FD002 FD003 FD004; do
  for v in nodrift noage; do
    [ -f ../results/trace/trace_$v/${sub}_s${seed}.npz ] && continue
    $PY run_trace_cmapss.py $sub $seed --experts aidgna,cnn,gru,lgbm --tag trace_$v --no_${v#no} 2>&1 | grep -v Warn
  done
done; done
for sub in FD001 FD002 FD003 FD004; do
  [ -f ../results/trace/trace_noise/${sub}_s0.npz ] && continue
  $PY run_trace_cmapss.py $sub 0 --experts aidgna,cnn,gru,lgbm --tag trace_noise --noise_test 0.1 2>&1 | grep -v Warn
done
$PY noise_aidgn.py 0.1 2>&1 | grep -v Warn
echo QUEUE_DONE
