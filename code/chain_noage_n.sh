#!/bin/bash
# start the N-CMAPSS no-age ablation (seed 0, LOUO) when both faithful-N queues are done; two parallel instances
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE logs/q_faithful_n_1.log && grep -q QUEUE_DONE logs/q_faithful_n_2.log; do sleep 60; done
for i in 1 2; do code/q_trace_n2.sh aidgna,cnn,gru,lgbm trace_louo_noage "0" "--louo --no_age" > logs/q_louo_noage_$i.log 2>&1 & sleep 5; done
wait
