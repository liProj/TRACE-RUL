"""Parse the printed comparison tables (Tables II and III of the article, reproduced in the README of the
release) into data/printed.json, so that no printed number is typed by hand.  The README values were
checked against the PDF tables (pages 7-8) for the AIDGN rows and Table IV (ablations) is typed from
the PDF page 9 image once, below, and cross-checked against the text percentages."""
import json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
html = open(f'{ROOT}/repo/AIDGN-RUL/README.md', encoding='utf-8').read()
cm_part, n_part = html.split('## Performance on N-CMAPSS Dataset')
cell = re.compile(r'<td>(?:<[bu]>)?\s*([\d.]+)\s*±\s*([\d.]+)\s*(?:</[bu]>)?</td>')
out = {'CMAPSS': {}, 'NCMAPSS': {}}
for row in re.findall(r'<tr>\s*<td>(?:<b>)?([^<]+?)(?:</b>)?</td>(.*?)</tr>', cm_part, re.S):
    name, body = row
    vals = cell.findall(body)
    if len(vals) == 8:
        subs = ['FD001', 'FD002', 'FD003', 'FD004']
        out['CMAPSS'][name.replace('Proposed ', '').strip()] = {
            s: {'rmse': float(vals[i][0]), 'rmse_sd': float(vals[i][1]), 'score': float(vals[i + 4][0]),
                'score_sd': float(vals[i + 4][1])} for i, s in enumerate(subs)}
blocks = re.split(r'<td rowspan="2">', n_part)[1:]
for b in blocks:
    name = b.split('</td>')[0].replace('Proposed ', '').strip()
    vals = cell.findall(b)
    if len(vals) >= 16:
        dss = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07', 'Avg']
        out['NCMAPSS'][name] = {s: {'rmse': float(vals[i][0]), 'rmse_sd': float(vals[i][1]),
                                    'score': float(vals[8 + i][0]), 'score_sd': float(vals[8 + i][1])}
                                for i, s in enumerate(dss)}
# Table IV (ablation, CMAPSS) from the PDF, RMSE / Score means only
T4 = {'Proposed aging': [(11.46, 248.49), (12.63, 663.80), (10.52, 202.87), (13.03, 762.25)],
      'PCA prior': [(12.81, 272.44), (12.78, 682.79), (11.96, 211.07), (13.62, 826.19)],
      'Feature-mean prior': [(11.62, 221.91), (13.10, 691.36), (11.37, 238.08), (14.73, 997.20)],
      'Pos. encoding prior': [(12.57, 281.94), (13.27, 706.72), (12.01, 255.69), (14.58, 951.07)],
      'No residual': [(12.12, 269.53), (13.67, 725.74), (11.86, 235.89), (14.80, 954.66)],
      'Single-path ADGC': [(12.85, 324.95), (13.02, 715.40), (10.88, 232.87), (14.46, 949.83)],
      'Aging-free': [(12.08, 262.98), (13.11, 741.05), (12.09, 256.74), (14.55, 941.67)],
      'Aging as aux. input': [(12.67, 313.59), (13.17, 742.06), (11.92, 256.07), (14.92, 1019.28)],
      'Feat.-similarity encoder': [(12.67, 276.59), (13.18, 772.06), (12.22, 240.00), (14.65, 994.22)],
      'Transformer encoder': [(12.48, 255.89), (14.99, 833.77), (11.63, 254.53), (14.09, 1002.62)]}
out['CMAPSS_ablation'] = {k: {s: {'rmse': v[i][0], 'score': v[i][1]} for i, s in
                              enumerate(['FD001', 'FD002', 'FD003', 'FD004'])} for k, v in T4.items()}
# the text gives FD004 RMSE 12.93 while Table II / README give 13.03 -> we use the table value
assert abs(out['CMAPSS']['AIDGN']['FD003']['rmse'] - 10.52) < 1e-9
assert abs(out['NCMAPSS']['AIDGN']['Avg']['rmse'] - 7.44) < 1e-9
json.dump(out, open(f'{ROOT}/data/printed.json', 'w'), indent=1)
print({k: list(v) for k, v in out.items()})
