# Third-party sources

## AIDGN-RUL

- Upstream: https://github.com/TCCofWANG/AIDGN-RUL
- Source revision: `d1ac173b13659af1854cf500a9b1cba8367353fe`
- Associated paper title: *Aging-Informed Dynamic Graph Networks for Aero-Engine RUL Estimation*.
- `code/aidgn.py` implements the upstream AIDGN architecture; `code/cmapss.py` and `code/ncmapss.py` document the upstream preprocessing they follow.
- `code/aidgn_official/` contains the source project's archived baseline implementation and local runner. Upstream README and requirements are retained there. Logs, datasets and model weights are excluded.
- No root LICENSE file was present in the inspected upstream checkout. This repository does not relicense upstream material or grant rights beyond those provided by its owners.

## Dependencies

PyTorch, NumPy, pandas, SciPy, scikit-learn, LightGBM, h5py, Numba, Matplotlib and TabPFN retain their respective licenses. TabPFN pretrained weights are obtained separately and are subject to their model terms. C-MAPSS and N-CMAPSS data are not bundled.

