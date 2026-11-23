"""Build gate: exit 1 on PENDING macros, empty cells in the main tables, undefined/unreferenced labels,
undefined citations, missing figures, or leftover <<markers>>."""
import os, re, sys, glob

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper')
errors = []
tex = {f: open(f).read() for f in glob.glob(f'{P}/*.tex')}
alltex = '\n'.join(tex.values())
log = open(f'{P}/main.log', errors='ignore').read() if os.path.exists(f'{P}/main.log') else ''
nums = open(f'{P}/numbers.tex').read()
if 'pending{' in nums:
    errors.append('numbers.tex PENDING: ' + ', '.join(re.findall(r'pending\{(\w+)\}', nums)))
for t in ('tab_cmapss', 'tab_ncmapss', 'tab_gate', 'tab_sig'):
    body = tex.get(f'{P}/{t}.tex', '')
    n = len(re.findall(r'(?<![\w-])--(?![\w-])', body.split(r'\midrule', 1)[-1]))
    if n:
        errors.append(f'{t}: {n} empty cells (--)')
labels = set(re.findall(r'\\label\{([^}]+)\}', alltex))
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', alltex))
for l in sorted(labels - refs):
    errors.append(f'label never referenced: {l}')
for r in sorted(refs - labels):
    errors.append(f'reference to undefined label: {r}')
for w in re.findall(r"(Reference `[^']+' on page \d+ undefined|Citation `[^']+' on page \d+ undefined)", log):
    errors.append(w)
for g in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', alltex):
    if not os.path.exists(os.path.join(P, '..', 'figures', g)):
        errors.append(f'missing figure {g}')
if re.search(r'<<[^>]+>>', alltex):
    errors.append('unfilled <<markers>>')
cites = set(k.strip() for c in re.findall(r'\\cite\{([^}]+)\}', alltex) for k in c.split(','))
bib = set(re.findall(r'@\w+\{([^,]+),', open(f'{P}/refs.bib').read()))
for c in sorted(cites - bib):
    errors.append(f'citation not in refs.bib: {c}')
print('\n'.join(errors) if errors else 'BUILD GATE OK')
sys.exit(1 if errors else 0)
