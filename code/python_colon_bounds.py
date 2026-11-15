#!/usr/bin/env python3
"""Colon-cancer trial: the bounds of Proposition 7(b) applied to a reconstructed prevalence snapshot.

Snapshot one year (365 days) after randomization. Exposed: levamisole plus fluorouracil;
unexposed: observation. 'Disease' is recurrence; a case leaves the prevalence pool by death
or loss to follow-up. For onset windows of 90, 180 and 365 days the script computes
  * the recent-onset odds ratio with its exact conditional 95% confidence interval,
  * the interval for the IDRR from the bounds of Proposition 7(b) for several values of the early-exit constant,
  * the tipping point: the excess exit hazard of exposed cases at which the upper limit reaches 1,
and, from the longitudinal records that a prevalence survey would not have,
  * the recurrence rate ratio during the window (first recurrences per recurrence-free patient-day),
  * survival after recurrence in each arm (Kaplan-Meier), the realized early-exit envelope
    max over a of -log{S1(a)/S0(a)}/a, and the ratio of restricted mean times R1(w)/R0(w).

--input colon.csv or --download : analyse the public patient-level file (not redistributed here)
--from-results                 : rebuild the table, figure and text numbers from the stored
                                 aggregate results in results/python/
"""
import argparse, json, math, io, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from python_interval_reproduction import fisher_ci

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/colon.csv'
T = 365; WINDOWS = (90, 180, 365); DELTAS = (0., .001, .002, .003)
EXPOSED, UNEXPOSED = 'Lev+5FU', 'Obs'
GRID = np.arange(0, 366, 5)


def B(x):
    return 1.0 if x < 1e-12 else -math.expm1(-x) / x


def records(raw):
    assert len(raw) == 1858 and raw.id.nunique() == 929
    rec = raw[raw.etype == 1].set_index('id'); dth = raw[raw.etype == 2].set_index('id')
    d = rec[['rx', 'time', 'status']].rename(columns={'time': 'rec_time', 'status': 'rec'}).join(
        dth[['time', 'status']].rename(columns={'time': 'end_time', 'status': 'dead'})).reset_index()
    return d[d.rx.isin([EXPOSED, UNEXPOSED])]


def km(dur, event):
    """Kaplan-Meier estimate as a step function: arrays of event times and survival just after them."""
    ts = np.sort(np.unique(dur[event == 1])); s = 1.0; out = []
    for u in ts:
        s *= 1 - ((dur == u) & (event == 1)).sum() / (dur >= u).sum(); out.append(s)
    return ts, np.array(out)


def step(ts, ss, a):
    k = np.searchsorted(ts, a, side='right')
    return 1.0 if k == 0 else float(ss[k - 1])


def restricted_mean(ts, ss, w):
    pts = np.r_[0., ts[ts < w], w]; vals = np.r_[1., ss[ts < w]]
    return float((np.diff(pts) * vals).sum())


def analyse(d):
    rows = []; curve = {}
    post = {}
    for g in (EXPOSED, UNEXPOSED):                      # survival after recurrence, recurrences of the first year
        z = d[(d.rx == g) & (d.rec == 1) & (d.rec_time <= T)]
        post[g] = km((z.end_time - z.rec_time).to_numpy(float), z.dead.to_numpy(int)) + (len(z),)
        curve[g] = [step(post[g][0], post[g][1], a) for a in GRID]
    for w in WINDOWS:
        c = {}
        for g in (EXPOSED, UNEXPOSED):
            z = d[d.rx == g]
            followed = z.end_time >= T
            prevalent = followed & (z.rec == 1) & (z.rec_time <= T)
            healthy = followed & ~prevalent & (z.rec_time >= T)
            recent = prevalent & ((T - z.rec_time) <= w)
            pt = np.maximum(0, np.minimum(np.minimum(z.rec_time, z.end_time), T) - max(0, T - w)).sum()
            ev = ((z.rec == 1) & (z.rec_time > T - w) & (z.rec_time <= T)).sum()
            c[g] = dict(H=int(healthy.sum()), R=int(recent.sum()), E=int(ev), PT=float(pt))
        a, b = c[EXPOSED], c[UNEXPOSED]
        ror = (a['R'] / a['H']) / (b['R'] / b['H'])
        lo, hi = fisher_ci(a['R'], a['H'], b['R'], b['H'], .05)
        r = dict(t=T, w=w, recent1=a['R'], recent0=b['R'], healthy1=a['H'], healthy0=b['H'], events1=a['E'], events0=b['E'],
                 ROR=ror, ROR_lo=lo, ROR_hi=hi, direct=(a['E'] / a['PT']) / (b['E'] / b['PT']))
        for dl in DELTAS:
            r[f'lo_{dl:g}'] = lo * B(dl * w); r[f'hi_{dl:g}'] = hi / B(dl * w)
        r['tipping'] = optimize.brentq(lambda x: B(x) - hi, 1e-9, 60.) / w if hi < 1 else 0.
        s1 = np.array([step(post[EXPOSED][0], post[EXPOSED][1], x) for x in range(30, w + 1, 5)])
        s0 = np.array([step(post[UNEXPOSED][0], post[UNEXPOSED][1], x) for x in range(30, w + 1, 5)])
        r['envelope'] = float(np.max(-np.log(s1 / s0) / np.arange(30, w + 1, 5)))
        r['S1_w'] = step(post[EXPOSED][0], post[EXPOSED][1], w); r['S0_w'] = step(post[UNEXPOSED][0], post[UNEXPOSED][1], w)
        r['R_ratio'] = restricted_mean(post[EXPOSED][0], post[EXPOSED][1], w) / restricted_mean(post[UNEXPOSED][0], post[UNEXPOSED][1], w)
        r['ROR_corrected'] = ror / r['R_ratio']
        r['n_post1'] = post[EXPOSED][2]; r['n_post0'] = post[UNEXPOSED][2]
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(dict(a=GRID, S1=curve[EXPOSED], S0=curve[UNEXPOSED]))


def f2(v): return f'{v:.2f}'
def f3(v): return f'{v:.3f}'
def perk(v): return f'{1000 * v:.1f}'          # per 1000 days


def outputs(res, kmg):
    rows = []
    for _, r in res.iterrows():
        cells = [f"{int(r['w'])}", f"{int(r['recent1'])} / {int(r['recent0'])}", f"{int(r['events1'])} / {int(r['events0'])}",
                 f"{f2(r['ROR'])} ({f2(r['ROR_lo'])}, {f2(r['ROR_hi'])})"]
        cells += [f"{f2(r[f'lo_{dl:g}'])} to {f2(r[f'hi_{dl:g}'])}" for dl in DELTAS[1:]]
        cells += [perk(r['tipping']) if r['tipping'] > 0 else '--', perk(r['envelope']), f2(r['R_ratio']), f2(r['ROR_corrected']), f2(r['direct'])]
        rows.append(' & '.join(cells) + r' \\')
    (ROOT / 'tex/tab_supp_colon_bounds_body.tex').write_text('\n'.join(rows) + '\n')
    m = {}
    for _, r in res.iterrows():
        t = 'nAppColon' + {90: 'A', 180: 'B', 365: 'C'}[int(r['w'])]
        m[t + 'Ror'] = f2(r['ROR']); m[t + 'Lo'] = f2(r['ROR_lo']); m[t + 'Hi'] = f2(r['ROR_hi']); m[t + 'Direct'] = f2(r['direct'])
        m[t + 'Tip'] = perk(r['tipping']) if r['tipping'] > 0 else '--'; m[t + 'Env'] = perk(r['envelope'])
        m[t + 'Corrected'] = f2(r['ROR_corrected']); m[t + 'Rratio'] = f2(r['R_ratio'])
        m[t + 'Recent'] = f"{int(r['recent1'])} and {int(r['recent0'])}"; m[t + 'Events'] = f"{int(r['events1'])} and {int(r['events0'])}"
        m[t + 'BoundTwoLo'] = f2(r['lo_0.002']); m[t + 'BoundTwoHi'] = f2(r['hi_0.002'])
    last = res[res.w == 365].iloc[0]
    m['nAppColonSurvOne'] = f2(last['S1_w']); m['nAppColonSurvZero'] = f2(last['S0_w'])
    m['nAppColonNpostOne'] = str(int(last['n_post1'])); m['nAppColonNpostZero'] = str(int(last['n_post0']))
    (ROOT / 'results/python/colon_bounds_macros.json').write_text(json.dumps(m, indent=1))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), layout='constrained')
    x = np.linspace(0, .004, 161); w = 365.
    lo = np.array([last['ROR_lo'] * B(v * w) for v in x]); hi = np.array([last['ROR_hi'] / B(v * w) for v in x])
    ax[0].fill_between(1000 * x, lo, hi, color='0.82', label='Interval for the IDRR from the bounds')
    ax[0].axhline(last['ROR'], color='black', lw=1.2, label='Recent-onset odds ratio')
    ax[0].axhline(last['direct'], color='#D55E00', ls='--', lw=1.2, label='Recurrence rate ratio from follow-up')
    ax[0].axhline(1, color='gray', ls=':', lw=.8)
    ax[0].axvline(1000 * last['tipping'], color='black', ls=':', lw=1.0)
    ax[0].axvline(1000 * last['envelope'], color='#0072B2', ls='-.', lw=1.0)
    ax[0].annotate('tipping point', (1000 * last['tipping'] + .07, .135), fontsize=7, rotation=90, ha='left', va='bottom')
    ax[0].annotate('observed in follow-up', (1000 * last['envelope'] - .07, .135), fontsize=7, rotation=90, ha='right', va='bottom', color='#0072B2')
    ax[0].set(yscale='log', ylim=(.12, 1.6), xlim=(0, 4), xlabel='Assumed limit on the excess exit hazard (per 1000 days)',
              ylabel='Combination treatment / observation', title='(A) 365-day window, snapshot at day 365')
    ax[0].set_yticks([.15, .2, .3, .5, .7, 1., 1.5]); ax[0].set_yticklabels(['0.15', '0.2', '0.3', '0.5', '0.7', '1', '1.5']); ax[0].minorticks_off()
    ax[0].legend(frameon=False, loc='upper left')
    ax[1].step(kmg.a, kmg.S1, where='post', color='black', lw=1.4, label='Combination treatment')
    ax[1].step(kmg.a, kmg.S0, where='post', color='black', lw=1.4, ls='--', label='Observation')
    ax[1].set(xlim=(0, 365), ylim=(0, 1), xlabel='Days since recurrence', ylabel='Proportion still alive and followed',
              title='(B) Not seen by the snapshot: survival after recurrence')
    ax[1].legend(frameon=False, loc='lower left')
    fig.savefig(ROOT / 'figures/fig_colon_bounds.pdf', bbox_inches='tight', pad_inches=.11); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input', type=Path); g.add_argument('--download', action='store_true'); g.add_argument('--from-results', action='store_true')
    a = p.parse_args(); R = ROOT / 'results/python'
    if a.from_results:
        res = pd.read_csv(R / 'colon_bounds.csv'); kmg = pd.read_csv(R / 'colon_postrecurrence_km.csv')
    else:
        raw = pd.read_csv(a.input) if a.input else pd.read_csv(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        res, kmg = analyse(records(raw))
        res.to_csv(R / 'colon_bounds.csv', index=False, float_format='%.10g'); kmg.to_csv(R / 'colon_postrecurrence_km.csv', index=False, float_format='%.6g')
    outputs(res, kmg)
    print(res[['w', 'recent1', 'recent0', 'ROR', 'ROR_lo', 'ROR_hi', 'direct', 'tipping', 'envelope', 'R_ratio', 'ROR_corrected', 'S1_w', 'S0_w']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
