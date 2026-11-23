"""Build paper/refs.bib from the Crossref API (DOIs) and the arXiv API (preprints). Every entry carries the title we
expect; an entry whose fetched title does not match is dropped and reported, never written from memory."""
import os, re, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, difflib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
UA = {'User-Agent': 'Lunwen05-bib/1.0 (mailto:research@example.org)'}

DOIS = {  # key: (DOI or None -> Crossref title search, expected title)
    'abbas2026aidgn': ('10.1109/TII.2026.3694202', 'Aging-Informed Dynamic Graph Network for Aero-Engine RUL Prediction'),
    'saxena2008cmapss': ('10.1109/PHM.2008.4711414', 'Damage propagation modeling for aircraft engine run-to-failure simulation'),
    'chao2021ncmapss': ('10.3390/data6010005', 'Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics'),
    'li2018dcnn': ('10.1016/j.ress.2017.11.021', 'Remaining useful life estimation in prognostics using deep convolution neural networks'),
    'zheng2017lstm': ('10.1109/ICPHM.2017.7998311', 'Long Short-Term Memory Network for Remaining Useful Life estimation'),
    'hochreiter1997lstm': ('10.1162/neco.1997.9.8.1735', 'Long Short-Term Memory'),
    'cho2014gru': ('10.3115/v1/D14-1179', 'Learning Phrase Representations using RNN Encoder\u2013Decoder for Statistical Machine Translation'),
    'hollmann2025tabpfn': ('10.1038/s41586-024-08328-6', 'Accurate predictions on small data with a tabular foundation model'),
    'wolpert1992stacking': ('10.1016/S0893-6080(05)80023-1', 'Stacked generalization'),
    'breiman1996stacked': ('10.1007/BF00117832', 'Stacked regressions'),
    'wang2024fcstgnn': ('10.1609/aaai.v38i14.29500', 'Fully-Connected Spatial-Temporal Graph for Multivariate Time-Series Data'),
    'fu2024dualmixer': ('10.1016/j.ress.2024.110398', 'Supervised contrastive learning based dual-mixer model for Remaining Useful Life prediction'),
    'liu2021agcnn': ('10.1109/TII.2020.2983760', 'Remaining Useful Life Prediction Using a Novel Feature-Attention-Based End-to-End Approach'),
    'wang2023cdsg': (None, 'Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction'),
    'zhang2023sdagcn': (None, 'Spatial-temporal dual-channel adaptive graph convolutional network for remaining useful life prediction with multi-sensor information fusion'),
    'li2022datransformer': (None, 'Domain Adaptive Remaining Useful Life Prediction With Transformer'),
    'liao2023pinn': (None, 'Remaining useful life with self-attention assisted physics-informed neural network'),
    'kalman1960filter': ('10.1115/1.3662552', 'A New Approach to Linear Filtering and Prediction Problems'),
    'lei2018review': ('10.1016/j.ymssp.2017.11.016', 'Machinery health prognostics: A systematic review from data acquisition to RUL prediction'),
    'si2011review': ('10.1016/j.ejor.2010.11.018', 'Remaining useful life estimation \u2013 A review on the statistical data driven approaches'),
    'chernozhukov2018dml': ('10.1111/ectj.12097', 'Double/debiased machine learning for treatment and structural parameters'),
    'heimes2008rnn': ('10.1109/PHM.2008.4711422', 'Recurrent neural networks for remaining useful life estimation'),
    'wang2025gnnsurvey': (None, 'A survey on graph neural networks for remaining useful life prediction: Methodologies, evaluation and future trends'),
    'babu2016cnn': (None, 'Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life'),
    'kipf2017gcn': (None, 'Semi-Supervised Classification with Graph Convolutional Networks'),
    'efron1979bootstrap': ('10.1214/aos/1176344552', 'Bootstrap Methods: Another Look at the Jackknife'),
}
ARXIV = {
    'vaswani2017attention': ('1706.03762', 'Attention Is All You Need'),
    'hinton2015distilling': ('1503.02531', 'Distilling the Knowledge in a Neural Network'),
    'nie2023patchtst': ('2211.14730', 'A Time Series is Worth 64 Words: Long-term Forecasting with Transformers'),
    'wu2021autoformer': ('2106.13008', 'Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting'),
    'lakshminarayanan2017ensembles': ('1612.01474', 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles'),
    'kipf2016gcn_arxiv': ('1609.02907', 'Semi-Supervised Classification with Graph Convolutional Networks'),
}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(2 + 3 * k)
    raise err


def same(a, b):
    n = lambda s: re.sub(r'[^a-z0-9 ]', '', re.sub(r'<[^>]+>', '', s.lower()))
    a, b = n(a), n(b)
    return b in a or difflib.SequenceMatcher(None, a, b).ratio() > 0.85


def tex(s):
    s = re.sub(r'<[^>]+>', '', s)
    for c in '&%$#_':
        s = s.replace(c, '\\' + c)
    return s.replace('–', '--').replace('—', '---')


def names(auths):
    out = []
    for a in auths:
        if 'family' in a:
            out.append(f"{a['family']}, {a.get('given', '')}".strip(', '))
        elif 'name' in a:
            out.append('{' + a['name'] + '}')
    return ' and '.join(out)


def main():
    entries, report = [], []
    for key, (doi, want) in DOIS.items():
        if doi is None:
            try:
                q = json.loads(get('https://api.crossref.org/works?rows=5&query.bibliographic=' + urllib.parse.quote(want)))
                hits = [it for it in q['message']['items'] if same((it.get('title') or [''])[0], want)]
                if not hits:
                    report.append(f'NOTFOUND {key}: {want}'); continue
                doi = hits[0]['DOI']
            except Exception as e:
                report.append(f'FAIL search {key}: {e}'); continue
        try:
            m = json.loads(get('https://api.crossref.org/works/' + urllib.parse.quote(doi)))['message']
        except Exception as e:
            report.append(f'FAIL {key} {doi}: {e}'); continue
        title = (m.get('title') or [''])[0]
        if not same(title, want):
            report.append(f'MISMATCH {key} {doi}: got "{title}"'); continue
        year = m.get('issued', {}).get('date-parts', [[None]])[0][0]
        f = dict(author=names(m.get('author', [])), title='{' + tex(title) + '}', journal=tex((m.get('container-title') or [''])[0]),
                 year=str(year), volume=m.get('volume'), number=m.get('issue'), pages=(m.get('page') or '').replace('-', '--'),
                 doi=doi)
        if not f['pages'] and m.get('article-number'):
            f['pages'] = m['article-number']
        typ = 'inproceedings' if m.get('type') == 'proceedings-article' else 'article'
        if typ == 'inproceedings':
            f['booktitle'] = f.pop('journal')
        entries.append('@' + typ + '{' + key + ',\n' + ',\n'.join(f'  {k} = {{{v}}}' for k, v in f.items() if v) + '\n}')
        report.append(f'ok {key}: {title[:70]} ({year})')
        time.sleep(0.4)
    ids = ','.join(v[0] for v in ARXIV.values())
    xml = get('https://export.arxiv.org/api/query?max_results=50&id_list=' + ids)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    feed = ET.fromstring(xml); got = {}
    for e in feed.findall('a:entry', ns):
        aid = re.sub(r'v\d+$', '', e.find('a:id', ns).text.rsplit('/', 1)[-1])
        got[aid] = dict(title=' '.join(e.find('a:title', ns).text.split()),
                        authors=[x.find('a:name', ns).text for x in e.findall('a:author', ns)],
                        year=e.find('a:published', ns).text[:4])
    for key, (aid, want) in ARXIV.items():
        g = got.get(aid)
        if g is None or not same(g['title'], want):
            report.append(f'MISMATCH/FAIL arXiv {key} {aid}: {g and g["title"]}'); continue
        au = ' and '.join(', '.join(reversed(a.rsplit(' ', 1))) if ' ' in a else a for a in g['authors'])
        entries.append('@misc{' + key + ',\n' + ',\n'.join([
            f'  author = {{{au}}}', f'  title = {{{{{tex(g["title"])}}}}}', f'  year = {{{g["year"]}}}',
            f'  eprint = {{{aid}}}', '  archivePrefix = {arXiv}', f'  howpublished = {{arXiv:{aid}}}']) + '\n}')
        report.append(f'ok arXiv {key}: {g["title"][:70]} ({g["year"]})')
    open(os.path.join(ROOT, 'paper', 'refs.bib'), 'w').write('\n\n'.join(entries) + '\n')
    open(os.path.join(ROOT, 'logs', 'bib_check.log'), 'w').write('\n'.join(report) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
