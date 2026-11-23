"""Summarize archived main-experiment RMSE using only the standard library."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'reference_results'


def main():
    for benchmark, folder, subsets, seeds in [
        ('C-MAPSS', ROOT / 'trace/trace', [f'FD00{i}' for i in range(1, 5)], range(5)),
        ('N-CMAPSS', ROOT / 'trace_n/trace_louo', [f'DS0{i}' for i in range(1, 8)], range(3)),
    ]:
        print('\n' + benchmark)
        print('| Subset | TRACE-RUL RMSE (mean +/- sample SD) | TRACE-S RMSE | Runs |')
        print('| --- | --- | --- | --- |')
        averages = [[], []]
        for subset in subsets:
            vals, students = [], []
            for seed in seeds:
                base = folder / f'{subset}_s{seed}'
                ev = Path(str(base) + '_eval.json')
                st = Path(str(base) + '_student.json')
                if ev.exists():
                    vals.append(json.loads(ev.read_text())['stack']['tt_rmse'])
                if st.exists():
                    students.append(json.loads(st.read_text())['tt_rmse'])
            def fmt(v):
                return f'{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}' if len(v) > 1 else (f'{v[0]:.2f}' if v else 'missing')
            print(f'| {subset} | {fmt(vals)} | {fmt(students)} | {len(vals)} / {len(students)} |')
            if vals: averages[0].append(statistics.mean(vals))
            if students: averages[1].append(statistics.mean(students))
        print('Mean over available subsets:', ', '.join(f'{statistics.mean(v):.4f}' if v else 'missing' for v in averages))


if __name__ == '__main__':
    main()
