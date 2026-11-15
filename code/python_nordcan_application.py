#!/usr/bin/env python3
"""Open-population illustration: four cancers in Denmark, men compared with women.

Data: NORDCAN 2.0, version 9.6.1 (Association of the Nordic Cancer Registries,
https://nordcan.iarc.fr/). For each cancer and sex the registry publishes
  * the number of new cases and the person-years by calendar year (incidence), and
  * the number of persons alive on 31 December who were diagnosed within the previous
    1, 3, 5 and 10 years, and at any time (prevalence by time since diagnosis), and
  * relative survival one year after diagnosis, by period of diagnosis.
The prevalence counts on 31 December 2024 are treated as a prevalence survey in which
the time since diagnosis is recorded in the intervals (0,1], (1,3], (3,5], (5,10], >10 years.
The registry incidence of the same years is the reference for the incidence density
rate ratio (IDRR).

The published one-year relative survival of 2020-2024 supplies external information on
early exits. If the sex with the lower survival leaves the prevalence pool at least as fast
as the other sex throughout the year (ordered exit hazards), the ratio of the restricted mean
times R_men(1)/R_women(1) lies between S_men(1)/S_women(1) and 1, and the confidence limit of
the one-year recent-onset odds ratio on that side is divided by the ratio of the survival
figures (supplement, section S1). The constant Delta = |log{S_women(1)/S_men(1)}| per year is
also reported: it is the smallest constant for which the envelope of Proposition 7(b) can hold.

Age. The same series carry, for 2024, the new cases and person-years by five-year age group
and the age-specific prevalence proportions. The numbers of prevalent cases and of residents
on 31 December in each age group are recovered from the published proportions (they are the
integers consistent with all of them, and they reproduce every published total). The
comparison of men with women is then repeated within age groups (Mantel-Haenszel).

Default: read data/nordcan_denmark_extract.csv, data/nordcan_denmark_age.csv and
         data/nordcan_denmark_survival.csv (the series used in the article).
--download: fetch the same series from the public NORDCAN interface and check that they
            agree with the stored extract (--write-extract replaces the stored files instead;
            --cache DIR keeps the raw responses in DIR and reuses them).
"""
import argparse, json, math, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from python_interval_reproduction import fisher_ci, grouped_logistic

ROOT = Path(__file__).resolve().parents[1]
API = 'https://canapi.iarc.who.int/api/nordcan/v2/961'
CANCERS = {160: 'Lung', 130: 'Pancreas', 520: 'Colorectal', 290: 'Melanoma of skin'}
ORDER = (160, 130, 520, 290)
TAG = {160: 'Lung', 130: 'Panc', 520: 'Colo', 290: 'Mela'}
SURVEY = 2024
COUNTRY = 208                                 # NORDCAN population code: Denmark
WINDOWS = (1, 3, 5, 10)
EDGES = (0, 1, 3, 5, 10)                      # recorded intervals of time since diagnosis, years
YEARS_OF = {1: (2024,), 3: (2022, 2023, 2024), 5: tuple(range(2020, 2025)), 10: tuple(range(2015, 2025))}


def B(x):
    return 1.0 if x < 1e-12 else -math.expm1(-x) / x


AGES = ['%d-%d' % (5 * k, 5 * k + 4) for k in range(17)] + ['85+']
PREV_WINDOWS = (0, 1, 3, 5, 10)                # 0: all prevalent cases


def fetch(ep, cache=None):
    """One series of the public interface (optionally kept in, and read from, a local directory)."""
    f = None
    if cache is not None:
        f = Path(cache) / (ep.strip('/').replace('/', '_').replace('(', '').replace(')', '') + '.json')
        if f.exists():
            return json.loads(f.read_text())['dataset']
    raw = urllib.request.urlopen(API + ep, timeout=60).read()
    time.sleep(0.5)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(raw)
    return json.loads(raw)['dataset']


def recover_population(rates, guess, total):
    """Residents on 31 December by age group, from the published age-specific prevalence
    proportions (per 100 000) of several series of one sex.

    rates: array (series, 18); guess: approximate population by age group (person-years of the
    year, rescaled); total: published population. For each age group the population is the
    only integer near the guess for which every proportion corresponds to a whole number of
    persons. Age groups without prevalent cases share the remainder in proportion to the guess.
    """
    N = np.full(18, np.nan)
    for a in range(18):
        r = rates[:, a]
        if (r > 0).sum() < 2:
            continue
        cand = np.arange(int(guess[a] * .90), int(guess[a] * 1.10) + 1)
        prod = np.outer(cand, r) / 1e5
        err = np.abs(prod - np.round(prod)).max(axis=1)
        k = np.argsort(err)
        if not (err[k[0]] < 1e-8 and err[k[1]] > 1e-7):
            raise SystemExit('The population of an age group cannot be recovered from the published proportions.')
        N[a] = cand[k[0]]
    miss = np.isnan(N)
    if miss.any():
        N[miss] = (total - np.nansum(N)) * guess[miss] / guess[miss].sum()
    return N


def download(cache=None):
    rows = []; arows = []
    for sex, s in ((1, 'M'), (2, 'F')):
        prev = {}; guess = None; total = None
        for c in ORDER:
            ep = f'/data/population/0/{sex}/({COUNTRY})/({c})/'
            for d in fetch(ep, cache):
                if 2015 <= d['year'] <= SURVEY:
                    rows.append(dict(cancer_code=c, cancer=CANCERS[c], sex=s, measure='incidence', window_years='',
                                     year=d['year'], count=int(d['total']), population=float(d['total_pop']), endpoint=ep))
                if d['year'] == SURVEY:
                    py = np.array([d['populations'][str(k)] for k in range(1, 19)], float)
                    for k in range(18):
                        arows.append(dict(cancer_code=c, cancer=CANCERS[c], sex=s, measure='incidence', window_years='',
                                          age_group=k + 1, age=AGES[k], count=int(d['ages'][str(k + 1)]), denominator=py[k],
                                          proportion_per_100000='', endpoint=ep))
                    total_py = float(d['total_pop'])
            for sp in PREV_WINDOWS:
                ep = f'/prevalence/population/{sp}/{sex}/({COUNTRY})/({c})/'
                for d in fetch(ep, cache):
                    if d['year'] == SURVEY:
                        rows.append(dict(cancer_code=c, cancer=CANCERS[c], sex=s, measure='prevalence',
                                         window_years='all' if sp == 0 else str(sp), year=d['year'],
                                         count=int(d['total']), population=float(d['total_pop']), endpoint=ep))
                        prev[c, sp] = (np.array([d['age_specific_rate'][str(k)] for k in range(1, 19)], float), int(d['total']), ep)
                        total = float(d['total_pop']); guess = py * total / total_py
        N = recover_population(np.array([v[0] for v in prev.values()]), guess, total)
        for (c, sp), (r, tot, ep) in prev.items():
            cnt = np.round(r * N / 1e5)
            if int(cnt.sum()) != tot:
                raise SystemExit('Recovered age-specific counts do not add up to a published total.')
            for k in range(18):
                arows.append(dict(cancer_code=c, cancer=CANCERS[c], sex=s, measure='prevalence',
                                  window_years='all' if sp == 0 else str(sp), age_group=k + 1, age=AGES[k], count=int(cnt[k]),
                                  denominator=float(N[k]), proportion_per_100000=float(r[k]), endpoint=ep))
    return pd.DataFrame(rows), pd.DataFrame(arows)


def mh_odds_ratio(a, b, c, d):
    """Mantel-Haenszel odds ratio over strata, with the Robins-Breslow-Greenland interval."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    T = a + b + c + d; R = a * d / T; S = b * c / T; P = (a + d) / T; Q = (b + c) / T
    est = R.sum() / S.sum()
    v = (P * R).sum() / (2 * R.sum() ** 2) + (P * S + Q * R).sum() / (2 * R.sum() * S.sum()) + (Q * S).sum() / (2 * S.sum() ** 2)
    return est, est * math.exp(-1.959964 * math.sqrt(v)), est * math.exp(1.959964 * math.sqrt(v))


def mh_rate_ratio(x1, t1, x0, t0):
    """Mantel-Haenszel rate ratio over strata, with the Greenland-Robins interval."""
    x1, t1, x0, t0 = (np.asarray(x, float) for x in (x1, t1, x0, t0))
    T = t1 + t0; num = (x1 * t0 / T).sum(); den = (x0 * t1 / T).sum(); est = num / den
    v = ((x1 + x0) * t1 * t0 / T ** 2).sum() / (num * den)
    return est, est * math.exp(-1.959964 * math.sqrt(v)), est * math.exp(1.959964 * math.sqrt(v))


def analyse_age(age, res):
    """Comparison of men with women within five-year age groups (age at the survey)."""
    out = []
    for c in ORDER:
        z = age[age.cancer_code == c]

        def col(sex, measure, w, name):
            y = z[(z.sex == sex) & (z.measure == measure) & (z.window_years.astype(str) == str(w))].sort_values('age_group')
            assert len(y) == 18
            return y[name].to_numpy(float)
        r = dict(cancer_code=c, cancer=CANCERS[c])
        r['IDRR_age'], r['IDRR_age_lo'], r['IDRR_age_hi'] = mh_rate_ratio(col('M', 'incidence', '', 'count'), col('M', 'incidence', '', 'denominator'),
                                                                         col('F', 'incidence', '', 'count'), col('F', 'incidence', '', 'denominator'))
        H = {s: col(s, 'prevalence', 'all', 'denominator') - col(s, 'prevalence', 'all', 'count') for s in 'MF'}
        for w, tag in ((1, 'ROR1'), (3, 'ROR3'), (5, 'ROR5'), (10, 'ROR10'), ('all', 'POR')):
            m = col('M', 'prevalence', w, 'count'); f = col('F', 'prevalence', w, 'count')
            r[tag + '_age'], r[tag + '_age_lo'], r[tag + '_age_hi'] = mh_odds_ratio(m, H['M'], f, H['F'])
        base = res[res.cancer_code == c].iloc[0]; ratio = float(base['surv_ratio'])
        r['bound_age_lo'] = r['ROR1_age_lo'] / max(ratio, 1.); r['bound_age_hi'] = r['ROR1_age_hi'] / min(ratio, 1.)
        r['bound_age_covers'] = bool(r['bound_age_lo'] <= r['IDRR_age'] <= r['bound_age_hi'])
        # strata that carry the comparison: age groups with at least 10 new cases in each sex
        r['age_groups_10'] = int(((col('M', 'incidence', '', 'count') >= 10) & (col('F', 'incidence', '', 'count') >= 10)).sum())
        out.append(r)
    return pd.DataFrame(out)


def rate_ratio_ci(n1, py1, n0, py0, alpha=.05):
    """Exact interval for a ratio of two Poisson rates (binomial, conditional on n1 + n0)."""
    n = n1 + n0
    lo = stats.beta.ppf(alpha / 2, n1, n0 + 1) if n1 > 0 else 0.
    hi = stats.beta.ppf(1 - alpha / 2, n1 + 1, n0) if n0 > 0 else 1.
    f = lambda p: p / (1 - p) * py0 / py1
    return (n1 / py1) / (n0 / py0), f(lo), f(hi)


def analyse(x, surv):
    out = []; curves = {}
    for c in ORDER:
        d = x[x.cancer_code == c]
        sv = {s: float(surv[(surv.cancer_code == c) & (surv.sex == s)].relative_survival_percent.iloc[0]) / 100 for s in 'MF'}

        def inc(sex, years, col):
            z = d[(d.measure == 'incidence') & (d.sex == sex) & (d.year.isin(years))]
            assert len(z) == len(years)
            return float(z[col].sum())

        def prev(sex, w):
            z = d[(d.measure == 'prevalence') & (d.sex == sex) & (d.window_years.astype(str) == str(w))]
            assert len(z) == 1
            return int(z['count'].iloc[0]), float(z['population'].iloc[0])
        P = {s: {w: prev(s, w)[0] for w in WINDOWS + ('all',)} for s in 'MF'}
        H = {s: prev(s, 'all')[1] - P[s]['all'] for s in 'MF'}          # persons without the cancer
        r = dict(cancer_code=c, cancer=CANCERS[c], H_M=H['M'], H_F=H['F'])
        for s in 'MF':
            r[f'cases_{s}'] = inc(s, (SURVEY,), 'count'); r[f'P_all_{s}'] = P[s]['all']
            for w in WINDOWS:
                r[f'P{w}_{s}'] = P[s][w]; r[f'N{w}_{s}'] = inc(s, YEARS_OF[w], 'count')
                r[f'alive{w}_{s}'] = P[s][w] / r[f'N{w}_{s}']            # share of the window's cases still prevalent
        r['IDRR'], r['IDRR_lo'], r['IDRR_hi'] = rate_ratio_ci(r['cases_M'], inc('M', (SURVEY,), 'population'),
                                                               r['cases_F'], inc('F', (SURVEY,), 'population'))
        for w in WINDOWS:
            r[f'IDRR{w}'] = rate_ratio_ci(r[f'N{w}_M'], inc('M', YEARS_OF[w], 'population'),
                                           r[f'N{w}_F'], inc('F', YEARS_OF[w], 'population'))[0]
            r[f'ROR{w}'] = (P['M'][w] / H['M']) / (P['F'][w] / H['F'])
            lo, hi = fisher_ci(P['M'][w], int(round(H['M'])), P['F'][w], int(round(H['F'])), .05)
            r[f'ROR{w}_lo'], r[f'ROR{w}_hi'] = lo, hi
        r['POR'] = (P['M']['all'] / H['M']) / (P['F']['all'] / H['F'])
        # odds ratio in each recorded interval, and the registry rate ratio of the corresponding diagnosis years
        cm = np.array([P['M'][1], P['M'][3] - P['M'][1], P['M'][5] - P['M'][3], P['M'][10] - P['M'][5], P['M']['all'] - P['M'][10]], float)
        cf = np.array([P['F'][1], P['F'][3] - P['F'][1], P['F'][5] - P['F'][3], P['F'][10] - P['F'][5], P['F']['all'] - P['F'][10]], float)
        orr = (cm / H['M']) / (cf / H['F']); se = np.sqrt(1 / cm + 1 / cf + 1 / H['M'] + 1 / H['F'])
        yrs = [(2024,), (2022, 2023), (2020, 2021), tuple(range(2015, 2020))]
        irr = [rate_ratio_ci(inc('M', y, 'count'), inc('M', y, 'population'), inc('F', y, 'count'), inc('F', y, 'population'))[0] for y in yrs]
        curves[c] = dict(orr=orr, lo=orr * np.exp(-1.959964 * se), hi=orr * np.exp(1.959964 * se), irr=irr)
        for k, lab in enumerate(('0_1', '1_3', '3_5', '5_10', 'gt10')):
            r[f'OR_{lab}'] = orr[k]
            if k < 4: r[f'IRR_{lab}'] = irr[k]
        # extrapolated odds ratio from the three intervals within 5 years (cases placed at the midpoints)
        b0, var = grouped_logistic(np.array([.5, 2., 4.]), cm[:3], cf[:3])
        r['EOR5'] = math.exp(b0) * H['F'] / H['M']; se5 = math.sqrt(var + 1 / H['M'] + 1 / H['F'])
        r['EOR5_lo'], r['EOR5_hi'] = r['EOR5'] * math.exp(-1.959964 * se5), r['EOR5'] * math.exp(1.959964 * se5)
        # one-year window: smallest early-exit constant (per year) for which the interval of
        # Proposition 7(b) contains the registry IDRR, and the realized ratio of restricted means
        L, U = r['ROR1_lo'], r['ROR1_hi']
        if r['IDRR'] > U:
            r['delta_side'] = 1; r['delta_cover'] = optimize.brentq(lambda z: U / B(z) - r['IDRR'], 1e-9, 50.)
        elif r['IDRR'] < L:
            r['delta_side'] = 0; r['delta_cover'] = optimize.brentq(lambda z: L * B(z) - r['IDRR'], 1e-9, 50.)
        else:
            r['delta_side'] = -1; r['delta_cover'] = 0.
        r['R_ratio1'] = r['alive1_M'] / r['alive1_F']
        # external information: published one-year relative survival (window of one year)
        r['surv_M'], r['surv_F'] = sv['M'], sv['F']
        ratio = sv['M'] / sv['F']                                     # S_1(w) / S_0(w)
        r['surv_ratio'] = ratio
        # ordered exit hazards: R_1(w)/R_0(w) lies between S_1(w)/S_0(w) and 1
        r['bound_lo'] = L / max(ratio, 1.); r['bound_hi'] = U / min(ratio, 1.)
        r['bound_covers'] = bool(r['bound_lo'] <= r['IDRR'] <= r['bound_hi'])
        r['R_ratio_in_range'] = bool(min(ratio, 1.) <= r['R_ratio1'] <= max(ratio, 1.))
        # smallest admissible constant of the envelope, and the interval it would give
        dpub = -math.log(ratio)
        r['delta1_pub'], r['delta0_pub'] = max(dpub, 0.), max(-dpub, 0.)
        r['B_delta_pub'] = B(abs(dpub))
        r['const_lo'] = L * B(r['delta0_pub']); r['const_hi'] = U / B(r['delta1_pub'])
        out.append(r)
    return pd.DataFrame(out), curves


def f2(v): return f'{v:.2f}'
def f3(v): return f'{v:.3f}'


def write_tables(res, ag):
    rows = []; srows = []
    ci = lambda r, k: f'{f2(getattr(r, k))} ({f2(getattr(r, k + "_lo"))}, {f2(getattr(r, k + "_hi"))})'
    # main table: one column per cancer
    lines = [('HEAD', 'All ages combined'), ('New cases in 2024, M / W', lambda r: f'{r.cases_M:.0f} / {r.cases_F:.0f}'),
             ('Registry IDRR (95\\% CI)', lambda r: f'{f2(r.IDRR)} ({f2(r.IDRR_lo)}, {f2(r.IDRR_hi)})'),
             (None, None),
             ('POR, all prevalent cases', lambda r: f2(r.POR)),
             ('ROR(10)', lambda r: f2(r.ROR10)),
             ('ROR(1) (95\\% CI)', lambda r: f'{f2(r.ROR1)} ({f2(r.ROR1_lo)}, {f2(r.ROR1_hi)})'),
             ('EOR(5) (95\\% CI)', lambda r: f'{f2(r.EOR5)} ({f2(r.EOR5_lo)}, {f2(r.EOR5_hi)})'),
             (None, None),
             ('Still prevalent, \\%, M / W', lambda r: f'{100 * r.alive1_M:.0f} / {100 * r.alive1_F:.0f}'),
             ('One-year survival, \\%, M / W', lambda r: f'{100 * r.surv_M:.0f} / {100 * r.surv_F:.0f}'),
             ('Interval from survival', lambda r: f'{f2(r.bound_lo)} to {f2(r.bound_hi)}'),
             (None, None), ('HEAD', 'Within five-year age groups'),
             ('Registry IDRR (95\\% CI)', lambda r: ci(r, 'IDRR_age')),
             ('POR, all prevalent cases', lambda r: f2(r.POR_age)),
             ('ROR(10)', lambda r: f2(r.ROR10_age)),
             ('ROR(1) (95\\% CI)', lambda r: ci(r, 'ROR1_age')),
             ('Interval from survival', lambda r: f'{f2(r.bound_age_lo)} to {f2(r.bound_age_hi)}')]
    recs = list(res.merge(ag, on=['cancer_code', 'cancer']).itertuples(index=False))
    assert len(recs) == len(ORDER)
    for label, fn in lines:
        if label is None: rows.append('\\addlinespace')
        elif label == 'HEAD': rows.append('\\multicolumn{5}{@{}l}{\\emph{' + fn + '}} \\\\')
        else: rows.append('\\hspace{0.8em}' + ' & '.join([label] + [fn(r) for r in recs]) + r' \\')
    # supplementary table: age-stratified estimates for every window
    arows = []
    for label, k in (('Registry IDRR, 2024', 'IDRR_age'), ('ROR(1)', 'ROR1_age'), ('ROR(3)', 'ROR3_age'), ('ROR(5)', 'ROR5_age'),
                     ('ROR(10)', 'ROR10_age'), ('POR, all prevalent cases', 'POR_age')):
        arows.append(' & '.join([label] + [ci(r, k) for r in recs]) + r' \\')
        if k == 'IDRR_age': arows.append('\\addlinespace')
    arows.append('\\addlinespace')
    arows.append(' & '.join(['Interval from survival, ROR(1)'] + [f'{f2(r.bound_age_lo)} to {f2(r.bound_age_hi)}' for r in recs]) + r' \\')
    (ROOT / 'tex/tab_supp_nordcan_age_body.tex').write_text('\n'.join(arows) + '\n')
    for r in recs:
        d = r._asdict()
        for w in WINDOWS:
            srows.append(' & '.join([r.cancer if w == 1 else '', str(w), f"{d[f'P{w}_M']:.0f} / {d[f'P{w}_F']:.0f}",
                                     f"{d[f'N{w}_M']:.0f} / {d[f'N{w}_F']:.0f}", f"{100 * d[f'alive{w}_M']:.0f} / {100 * d[f'alive{w}_F']:.0f}",
                                     f3(d[f'ROR{w}']), f3(d[f'IDRR{w}']), f3(d[f'ROR{w}'] / d[f'IDRR{w}'])]) + r' \\')
        srows.append(' & '.join(['', 'All', f'{r.P_all_M:.0f} / {r.P_all_F:.0f}', '', '', f3(r.POR), '', '']) + r' \\')
        if r.cancer_code != ORDER[-1]: srows.append(r'\addlinespace')
    (ROOT / 'tex/tab_nordcan_body.tex').write_text('\n'.join(rows) + '\n')
    (ROOT / 'tex/tab_supp_nordcan_body.tex').write_text('\n'.join(srows) + '\n')


def macros(res, ag):
    m = {}
    for _, r in res.iterrows():
        t = 'nApp' + TAG[int(r['cancer_code'])]
        m[t + 'Idrr'] = f2(r['IDRR']); m[t + 'Por'] = f2(r['POR']); m[t + 'RorOne'] = f2(r['ROR1'])
        m[t + 'RorOneLo'] = f2(r['ROR1_lo']); m[t + 'RorOneHi'] = f2(r['ROR1_hi']); m[t + 'RorTen'] = f2(r['ROR10'])
        m[t + 'EorFive'] = f2(r['EOR5']); m[t + 'AliveM'] = f"{100 * r['alive1_M']:.0f}"; m[t + 'AliveF'] = f"{100 * r['alive1_F']:.0f}"
        m[t + 'DeltaCover'] = '0' if r['delta_cover'] == 0 else f2(r['delta_cover'])
        m[t + 'IdrrTen'] = f2(r['IDRR10']); m[t + 'RorOneOverIdrr'] = f2(r['ROR1'] / r['IDRR'])
        m[t + 'BoundLo'] = f2(r['bound_lo']); m[t + 'BoundHi'] = f2(r['bound_hi'])
        m[t + 'SurvM'] = f"{100 * r['surv_M']:.0f}"; m[t + 'SurvF'] = f"{100 * r['surv_F']:.0f}"
        m[t + 'DeltaPub'] = f2(max(r['delta1_pub'], r['delta0_pub']))
        m[t + 'SurvRatio'] = f2(r['surv_ratio']); m[t + 'Rratio'] = f2(r['R_ratio1']); m[t + 'BDeltaPub'] = f2(r['B_delta_pub'])
        m[t + 'ConstLo'] = f2(r['const_lo']); m[t + 'ConstHi'] = f2(r['const_hi'])
        m[t + 'OrLast'] = f2(r['OR_gt10']); m[t + 'IrrOld'] = f2(r['IRR_5_10'])
        # standard error of the log ratio of the shares still prevalent (binomial shares of the year's new cases)
        se = math.sqrt((1 - r['alive1_M']) / r['P1_M'] + (1 - r['alive1_F']) / r['P1_F'])
        m[t + 'RratioSe'] = f2(se)
    for _, r in ag.iterrows():
        t = 'nApp' + TAG[int(r['cancer_code'])]
        for tag, k in (('IdrrAge', 'IDRR_age'), ('RorOneAge', 'ROR1_age'), ('PorAge', 'POR_age'), ('RorTenAge', 'ROR10_age')):
            m[t + tag] = f2(r[k]); m[t + tag + 'Lo'] = f2(r[k + '_lo']); m[t + tag + 'Hi'] = f2(r[k + '_hi'])
        m[t + 'BoundAgeLo'] = f2(r['bound_age_lo']); m[t + 'BoundAgeHi'] = f2(r['bound_age_hi'])
        m[t + 'RorOneOverIdrrAge'] = f2(r['ROR1_age'] / r['IDRR_age'])
    other = res[res.cancer_code != 160]
    m['nAppMaxSurvDiff'] = f"{100 * (1 - np.minimum(other.surv_ratio, 1 / other.surv_ratio)).max():.1f}"
    return m


def figure(res, curves):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), layout='constrained')
    left = np.array([0, 1, 3, 5, 10.]); right = np.array([1, 3, 5, 10, 14.]); mid = (left + right) / 2
    for j, (ax, c) in enumerate(zip(axes.flat, ORDER)):
        r = res[res.cancer_code == c].iloc[0]; cv = curves[c]
        ax.axhline(r.IDRR, color='#D55E00', ls='--', lw=1.2, label='Registry IDRR, 2024')
        ax.axhline(r.POR, color='black', ls=':', lw=1.1, label='POR, all prevalent cases')
        for k in range(5):
            ax.plot([left[k], right[k]], [cv['orr'][k]] * 2, color='black', lw=1.8, label='Prevalence odds ratio in interval (95% CI)' if k == 0 else None)
            ax.plot([mid[k]] * 2, [cv['lo'][k], cv['hi'][k]], color='black', lw=.8)
        ax.scatter(mid[:4], cv['irr'], marker='D', s=22, facecolors='white', edgecolors='#0072B2', zorder=3,
                   label='Registry rate ratio in the years of diagnosis')
        ax.set(title=f'({chr(65 + j)}) {r.cancer}', xlabel='Time since diagnosis (years)', ylabel='Men / women', yscale='log', xlim=(0, 14))
        lo = min(cv['lo'].min(), r.POR, r.IDRR, min(cv['irr'])); hi = max(cv['hi'].max(), r.POR, r.IDRR, max(cv['irr']))
        ax.set_ylim(lo / 1.08, hi * 1.08)
        ticks = [t for t in (.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8) if lo / 1.08 <= t <= hi * 1.08]
        ax.set_yticks(ticks); ax.set_yticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
        ax.set_xticks([0, 1, 3, 5, 10, 14]); ax.set_xticklabels(['0', '1', '3', '5', '10', '>10'])
        if j == 0: handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.savefig(ROOT / 'figures/fig_nordcan.pdf', bbox_inches='tight', pad_inches=.11); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true', help='fetch the series again and compare them with the stored extract')
    p.add_argument('--write-extract', action='store_true', help='with --download: replace the stored extract by the fetched series')
    p.add_argument('--cache', type=Path, help='directory in which raw responses are kept and reused')
    a = p.parse_args()
    f_tot = ROOT / 'data/nordcan_denmark_extract.csv'; f_age = ROOT / 'data/nordcan_denmark_age.csv'
    if a.download:
        fresh, fresh_age = download(a.cache)
        if a.write_extract:
            fresh.to_csv(f_tot, index=False); fresh_age.to_csv(f_age, index=False, float_format='%.12g')
            print('Wrote', f_tot.name, len(fresh), 'rows and', f_age.name, len(fresh_age), 'rows')
    stored = pd.read_csv(f_tot, keep_default_na=False); stored['window_years'] = stored['window_years'].astype(str)
    age = pd.read_csv(f_age, keep_default_na=False); age['window_years'] = age['window_years'].astype(str)
    if a.download:
        key = ['cancer_code', 'sex', 'measure', 'window_years', 'year']
        fresh['window_years'] = fresh['window_years'].astype(str)
        z = stored.merge(fresh, on=key, suffixes=('', '_new'))
        assert len(z) == len(stored) == len(fresh)
        same = np.array_equal(z['count'].astype(int), z['count_new'].astype(int)) and np.allclose(z.population.astype(float), z.population_new)
        key = ['cancer_code', 'sex', 'measure', 'window_years', 'age_group']
        fresh_age['window_years'] = fresh_age['window_years'].astype(str)
        z = age.merge(fresh_age, on=key, suffixes=('', '_new'))
        assert len(z) == len(age) == len(fresh_age)
        same = same and np.array_equal(z['count'].astype(int), z['count_new'].astype(int)) and np.allclose(z.denominator.astype(float), z.denominator_new.astype(float))
        if not same:
            raise SystemExit('The NORDCAN interface now returns different numbers from the stored extract (new database version?).')
        print('Downloaded series agree with', f_tot.name, 'and', f_age.name)
    stored['population'] = stored['population'].astype(float); stored['count'] = stored['count'].astype(int)
    for c in ('count', 'denominator'): age[c] = age[c].astype(float)
    # the age-specific values add up to the totals used in the unstratified analysis
    for (c, s, m, w), g in age.groupby(['cancer_code', 'sex', 'measure', 'window_years']):
        t = stored[(stored.cancer_code == c) & (stored.sex == s) & (stored.measure == m) & (stored.window_years == w) & (stored.year == SURVEY)]
        assert len(g) == 18 and len(t) == 1 and int(g['count'].sum()) == int(t['count'].iloc[0]), (c, s, m, w)
        assert abs(g['denominator'].sum() - float(t['population'].iloc[0])) < 0.5, (c, s, m, w)
    surv = pd.read_csv(ROOT / 'data/nordcan_denmark_survival.csv')
    if a.download:
        for r in surv.itertuples(index=False):
            now = [float(d['survival']['value']) for d in fetch(r.endpoint, a.cache) if d.get('period') == r.period]
            if now != [r.relative_survival_percent]:
                raise SystemExit('The NORDCAN interface now returns a different survival figure from the stored one.')
        print('Downloaded survival figures agree with data/nordcan_denmark_survival.csv')
    res, curves = analyse(stored, surv)
    ag = analyse_age(age, res)
    res.to_csv(ROOT / 'results/python/nordcan_application.csv', index=False, float_format='%.10g')
    ag.to_csv(ROOT / 'results/python/nordcan_age_stratified.csv', index=False, float_format='%.10g')
    write_tables(res, ag); figure(res, curves)
    (ROOT / 'results/python/nordcan_macros.json').write_text(json.dumps(macros(res, ag), indent=1))
    cols = ['cancer', 'IDRR', 'POR', 'ROR1', 'ROR1_lo', 'ROR1_hi', 'ROR10', 'EOR5', 'alive1_M', 'alive1_F', 'surv_ratio', 'R_ratio1', 'bound_lo', 'bound_hi', 'bound_covers']
    print(res[cols].round(3).to_string(index=False))
    print(ag.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
