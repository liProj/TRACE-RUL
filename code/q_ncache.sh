#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
PY="${PYTHON:-python}"
for ds in DS01 DS02 DS03 DS04 DS05 DS06 DS07; do
  $PY -c "
import ncmapss as n; d=n.build('$ds')
print('$ds', 'train', d['X_tr'].shape, 'test', d['X_te'].shape, 'max_life', d['max_life'], 'mean_life', d['mean_life'], 'cols', list(d['cols']))" 2>&1 | grep -v Warn
  $PY train_aidgn_n.py $ds 0 --ckpt 2>&1 | grep -v Warn
done
echo CACHE_DONE
