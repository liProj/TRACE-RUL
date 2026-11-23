"""Check archived sources and numerical TT behavior without training models."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'tools').glob('*.py'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for path in sources:
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    print(f'Python syntax: PASS ({len(sources)} files)')
    records = list((ROOT / 'reference_results').rglob('*.json'))
    for path in records:
        json.loads(path.read_text(encoding='utf-8'))
    print(f'Archived result parsing: PASS ({len(records)} files)')
    manifest = json.loads((ROOT / 'docs/source_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        digest = hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest()
        assert digest == entry['sha256_release'], entry['path']
    print('Source manifest: PASS')
    for bench in ['cmapss', 'ncmapss']:
        result = subprocess.run([sys.executable, str(ROOT / 'tools/run_experiments.py'), bench, '--dry-run'], capture_output=True, text=True, check=True)
        expected = 36 if bench == 'cmapss' else 63
        # On a fresh checkout, every run has train/evaluate/distill commands.
        assert f'code/run_trace_{bench}.py' in result.stdout or (ROOT / 'results').exists()
        print(f'{bench} command generation: PASS ({len(result.stdout.splitlines())} commands; fresh checkout {expected})')
    try:
        import numpy as np
    except ImportError:
        print('TT numerical checks: SKIPPED (numpy not installed)')
        return
    # Execute the exact pure TT function from source, without importing optional training dependencies.
    tree = ast.parse((ROOT / 'code/trace_post.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tt_filter')
    env = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'trace_post.py', 'exec'), env)
    tt = env['tt_filter']
    p = np.array([10., 10., 10., 20.])
    cycles = np.array([1., 2., 4., 1.])
    units = np.array([1, 1, 1, 2])
    np.testing.assert_allclose(tt(p, cycles, units, 1., 0., 100., 125.), [10., 9.5, 25./3., 20.])
    np.testing.assert_allclose(tt(p, cycles, units, .5, 1., 100., 125.), p)
    np.testing.assert_allclose(tt(p, cycles, units, .5, 0., 0., 125.), p)
    np.testing.assert_allclose(tt(p, cycles, units, 0., 0., 100., 125.), p)
    full = tt(p, cycles, units, .8, .3, 100., 125.)
    prefix = tt(p[:2], cycles[:2], units[:2], .8, .3, 100., 125.)
    np.testing.assert_allclose(full[:2], prefix)
    print('TT numerical checks: PASS (cycle gaps, unit reset, identity, fallback, zero forgetting, causal prefix)')
    print('Training, CUDA latency and full dataset evaluation: NOT rerun during packaging')


if __name__ == '__main__':
    main()
