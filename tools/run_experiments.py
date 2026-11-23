"""Portable orchestrator for the configurations used in the main experiments."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['cmapss', 'ncmapss'])
    parser.add_argument('--subsets', help='Comma-separated subset names')
    parser.add_argument('--seeds', help='Comma-separated integer seeds')
    parser.add_argument('--tag', help='Output configuration name')
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    is_c = a.benchmark == 'cmapss'
    subsets = a.subsets.split(',') if a.subsets else ([f'FD00{i}' for i in range(1, 5)] if is_c else [f'DS0{i}' for i in range(1, 8)])
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else list(range(3))
    tag = a.tag or ('trace' if is_c else 'trace_louo')
    if not tag or Path(tag).name != tag or '/' in tag or '\\' in tag or tag in ['.', '..']:
        parser.error('--tag must be a simple directory name')
    valid = {f'FD00{i}' for i in range(1, 5)} if is_c else {f'DS0{i}' for i in range(1, 8)}
    if not set(subsets) <= valid:
        parser.error('Subset name does not belong to the selected benchmark')
    experts = 'aidgna,cnn,gru,lgbm,tabpfn' if is_c else 'aidgna,cnn,gru,lgbm'
    for subset in subsets:
        for seed in seeds:
            base = ROOT / 'results' / ('trace' if is_c else 'trace_n') / tag / f'{subset}_s{seed}'
            train = [sys.executable, f'code/run_trace_{a.benchmark}.py', subset, str(seed), '--experts', experts, '--tag', tag]
            if not is_c:
                train.append('--louo')
            commands = []
            if not base.with_suffix('.npz').exists():
                commands.append(train)
            commands.append([sys.executable, f'code/eval_{a.benchmark}.py', tag, subset, str(seed)])
            if not Path(str(base) + '_student.json').exists():
                commands.append([sys.executable, 'code/distill.py', a.benchmark, subset, str(seed), '--tag', tag])
            for command in commands:
                print(subprocess.list2cmdline(command), flush=True)
                if not a.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
