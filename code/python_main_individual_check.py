#!/usr/bin/env python3
"""The comparison of the estimators repeated with simulated individual histories.

python_main_reproduction.py draws each survey from the stationary distribution of
Proposition 1. Here the same scenarios are simulated as individual histories (entry, latent
onset, latent emigration, own duration; python_individual_histories.main_design_surveys) and
the surveys are analysed by the same function. The two sets of results are then compared.
The article uses the population sizes 200, 1000 and 5000 (about three hours on two cores).

Run:  python code/python_main_individual_check.py --reps 10000 --out results/python
      (--scenario and --sizes select part of the design; --resume continues an interrupted run)
      python code/python_main_individual_check.py --compare --out results/python
      (comparison with the stationary-law results: table, numbers quoted in the text)
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from python_main_reproduction import get_scenarios, summarize, SHAPE, SIZES, ESTS, D0, I0, GAM
from python_individual_histories import main_design_surveys

SEED = 2026111700
L = 20000.0                     # days of entries before the survey
CAP = 4000                      # buffer for the times since onset of one group
ARTICLE_SIZES = (200, 1000, 5000)   # smallest, middle and largest population size of the main study


EST_LABEL = {'POR': 'POR', 'aPOR': 'aPOR', 'ROR30': 'ROR(30)', 'ROR60': 'ROR(60)', 'ROR90': 'ROR(90)',
             'EOR90': 'EOR(90)', 'EOR180': 'EOR(180)', 'EORInf': 'EOR($\\infty$)'}
TESTS = (('rej_wilcoxon', 'Wilcoxon'), ('rej_ks', 'Kolmogorov--Smirnov'), ('rej_recent30', 'Recent fraction, 30 days'),
         ('rej_shape', 'Scale model'), ('rej_linear', 'Linear logit'))


def compare(out):
    """Individual-history results against the stationary-law results of python_main_reproduction.py."""
    root = Path(__file__).resolve().parents[1]
    ind = pd.read_csv(out / 'main_individual_summary.csv'); law = pd.read_csv(out / 'main_summary_python.csv')
    ti = pd.read_csv(out / 'main_individual_tests.csv'); tl = pd.read_csv(out / 'main_tests_python.csv')
    done = ti.groupby('n').id.nunique(); sizes = sorted(int(n) for n in done.index if done[n] == 56)   # complete sizes only
    assert sizes, 'no population size is complete'
    if tuple(sizes) != ARTICLE_SIZES:
        print('NOTE: sizes compared', sizes, 'differ from those of the article', ARTICLE_SIZES)
    ind = ind[ind.n.isin(sizes)]; ti = ti[ti.n.isin(sizes)]
    assert (ind.reps == 10000).all() and len(ind) == 56 * 8 * len(sizes)
    m = ind.merge(law, on=['id', 'n', 'estimator'], suffixes=('_ind', '_law'), validate='one_to_one')
    m['d_log_gm'] = np.log(m.gmean_est_ind / m.gmean_est_law)
    m['z_gm'] = m.d_log_gm / np.sqrt(m.mcse_bias_ind ** 2 + m.mcse_bias_law ** 2)
    m['d_cov'] = 100 * (m.coverage_ind - m.coverage_law)
    se_cov = 100 * np.sqrt(m.mcse_cov_ind ** 2 + m.mcse_cov_law ** 2)
    m['z_cov'] = np.where(se_cov > 0, m.d_cov / se_cov.replace(0, np.nan), np.nan)
    m['se_ratio'] = m.emp_se_ind / m.emp_se_law
    m['d_undef'] = 100 * (m.undefined_ind - m.undefined_law)
    m.to_csv(out / 'main_individual_comparison.csv', index=False, float_format='%.10g')
    t = ti.merge(tl, on=['id', 'n'], suffixes=('_ind', '_law'), validate='one_to_one')
    rows = []; mac = {}

    def line(label, d, z, extra):
        zz = z[np.isfinite(z)]
        return ' & '.join([label, str(len(d)), f'{np.nanmax(np.abs(d)):.1f}', f'{100 * np.mean(np.abs(zz) > 1.959964):.1f}'] + extra) + r' \\'
    for e in list(EST_LABEL):
        x = m[m.estimator == e]
        zc = x.z_cov.to_numpy(float); zc = zc[np.isfinite(zc)]
        rows.append(' & '.join([EST_LABEL[e], str(len(x)), f'{100 * np.nanmax(np.abs(x.d_log_gm)):.1f}',
                                f'{100 * np.mean(np.abs(x.z_gm.dropna()) > 1.959964):.1f}', f'{np.nanmax(np.abs(x.d_cov)):.1f}',
                                f'{100 * np.mean(np.abs(zc) > 1.959964):.1f}', f'{x.se_ratio.min():.2f} to {x.se_ratio.max():.2f}']) + r' \\')
    zc = m.z_cov.to_numpy(float); zc = zc[np.isfinite(zc)]; zg = m.z_gm.dropna().to_numpy(float)
    rows.append(r'\addlinespace')
    rows.append(' & '.join(['All estimators', str(len(m)), f'{100 * np.nanmax(np.abs(m.d_log_gm)):.1f}', f'{100 * np.mean(np.abs(zg) > 1.959964):.1f}',
                            f'{np.nanmax(np.abs(m.d_cov)):.1f}', f'{100 * np.mean(np.abs(zc) > 1.959964):.1f}',
                            f'{m.se_ratio.min():.2f} to {m.se_ratio.max():.2f}']) + r' \\')
    trow = []; zt_all = []
    for col, lab in TESTS:
        d = 100 * (t[col + '_ind'] - t[col + '_law']); pbar = (t[col + '_ind'] + t[col + '_law']) / 2
        z = d / (100 * np.sqrt(2 * pbar * (1 - pbar) / 10000)).replace(0, np.nan); z = z.dropna().to_numpy(float); zt_all.append(z)
        trow.append(' & '.join([lab, str(int(d.notna().sum())), f'{np.nanmax(np.abs(d)):.1f}', f'{100 * np.mean(np.abs(z) > 1.959964):.1f}']) + r' \\')
    (root / 'tex/tab_supp_individual_body.tex').write_text('\n'.join(rows) + '\n')
    (root / 'tex/tab_supp_individual_tests_body.tex').write_text('\n'.join(trow) + '\n')
    zt = np.concatenate(zt_all)
    words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
    mac['nIndSizes'] = ', '.join(str(n) for n in sizes[:-1]) + (' and ' if len(sizes) > 1 else '') + str(sizes[-1])
    mac['nIndNsizes'] = words[len(sizes)]
    mac['nIndRows'] = f'{len(m):,}'.replace(',', '\\,')
    mac['nIndGmMax'] = f'{100 * np.nanmax(np.abs(m.d_log_gm)):.1f}'
    mac['nIndGmMaxFrom'] = f'{100 * np.nanmax(np.abs(m[m.n >= 1000].d_log_gm)):.1f}' if (m.n >= 1000).any() else ''
    mac['nIndGmBeyond'] = f'{100 * np.mean(np.abs(zg) > 1.959964):.1f}'
    mac['nIndCovMax'] = f'{np.nanmax(np.abs(m.d_cov)):.1f}'
    mac['nIndCovBeyond'] = f'{100 * np.mean(np.abs(zc) > 1.959964):.1f}'
    mac['nIndSeRatioMin'] = f'{m.se_ratio.min():.2f}'; mac['nIndSeRatioMax'] = f'{m.se_ratio.max():.2f}'
    mac['nIndTestBeyond'] = f'{100 * np.mean(np.abs(zt) > 1.959964):.1f}'
    mac['nIndUndefMax'] = f'{np.nanmax(np.abs(m.d_undef)):.1f}'
    mac['nIndLargestZ'] = f'{max(np.max(np.abs(zg)), np.max(np.abs(zc))):.1f}'
    (out / 'main_individual_macros.json').write_text(json.dumps(mac, indent=1))
    print(json.dumps(mac, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compare', action='store_true')
    p.add_argument('--reps', type=int, default=10000)
    p.add_argument('--scenario', nargs='*', type=int)
    p.add_argument('--sizes', nargs='*', type=int, default=list(ARTICLE_SIZES))
    p.add_argument('--out', type=Path, default=Path('results/python'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.compare:
        compare(a.out); return
    f_sum = a.out / 'main_individual_summary.csv'; f_test = a.out / 'main_individual_tests.csv'
    if a.resume and f_sum.exists():
        summaries = pd.read_csv(f_sum).to_dict('records'); tests = pd.read_csv(f_test).to_dict('records')
        seen = set((int(t['id']), int(t['n'])) for t in tests if int(t['reps']) == a.reps)
    else:
        summaries = []; tests = []; seen = set()
    start = time.perf_counter()
    for s in get_scenarios().itertuples(index=False):
        if a.scenario and s.id not in a.scenario:
            continue
        for n in a.sizes:
            if (s.id, n) in seen:
                continue
            i = SIZES.index(n)
            raw = main_design_surveys(a.reps, float(n), SHAPE[s.shape1], SHAPE[s.shape0], float(s.rho), float(s.idrr),
                                      float(s.beta), I0, D0, GAM, L, SEED + 1_000_000 * int(s.id) + 100_000 * i, CAP)
            if (raw[:, 23] < 0).any():
                raise SystemExit('buffer for the times since onset too small')
            rr, tt = summarize(raw, s, n, a.reps); summaries.extend(rr); tests.append(tt)
            pd.DataFrame(summaries).to_csv(f_sum, index=False, float_format='%.15g')
            pd.DataFrame(tests).to_csv(f_test, index=False, float_format='%.15g')
            print('scenario', s.id, 'n', n, 'elapsed', round(time.perf_counter() - start, 1), 'sec', flush=True)
    print('completed', len(summaries), 'estimator rows;', len(tests), 'diagnostic rows', flush=True)


if __name__ == '__main__':
    main()
