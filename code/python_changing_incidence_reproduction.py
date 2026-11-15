#!/usr/bin/env python3
"""Incidence rates that change over calendar time (Proposition 6).

Simulation: individual histories forward in time (python_individual_histories.py). Each
population starts empty 12 000 days before the survey; healthy people enter at a constant
rate, fall ill at a rate that depends on calendar time, or emigrate; durations are exponential.

Limits: computed separately from the intensity I(t-a) h(t-a) S(a) of Proposition 6, with
h obtained by integrating the entry, incidence and emigration rates on a 0.25-day grid.
The two are compared in the supplementary table on changing incidence rates.
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from python_individual_histories import changing_incidence_surveys, MODE_CODE, Z975

I0 = .001; GAM = .0005; D0 = 180.; DT = .25; L = 12000.; N = 5000; SEED = 2026111600
ESTS = ('POR', 'aPOR', 'ROR30', 'ROR90', 'EOR90', 'EOR180', 'EORInf')
# name, change in the exposed group, change in the unexposed group, duration ratio
SC = [('rise', 'rise', 'none', 1.), ('fall', 'fall', 'none', 1.), ('both', 'rise', 'rise', 1.),
      ('step', 'step', 'none', 1.), ('rise6', 'rise', 'none', .6)]


def factor(u, mode):
    """Incidence rate at calendar time u <= 0 relative to the rate at the survey (u = 0)."""
    a = np.minimum(-u, 365.)
    if mode == 'none': return np.ones_like(u)
    if mode == 'rise': return 2 ** (-a / 365.)
    if mode == 'fall': return 2 ** (a / 365.)
    if mode == 'step': return np.where(u < -180., .5, 1.)
    raise ValueError(mode)


def entry_rate(inow, D):
    return (inow + GAM) * N / (1 + inow * D)


def group(inow, kind, D):
    """Expected number healthy at the survey and the intensity of the times since onset."""
    mid = -L + (np.arange(int(L / DT)) + .5) * DT
    inc = inow * factor(mid, kind); rate = inc + GAM
    lam = entry_rate(inow, D)
    H = np.zeros(len(inc) + 1)
    for k in range(len(inc)):
        e = math.exp(-rate[k] * DT)
        H[k + 1] = H[k] * e + lam * (-math.expm1(-rate[k] * DT)) / rate[k]
    age = -mid[::-1]
    weights = (inc * ((H[:-1] + H[1:]) / 2))[::-1] * np.exp(-age / D) * DT
    return dict(H=H[-1], P=weights.sum(), age=age, w=weights)


def pop_fit(a, n1, n0):
    nn = n1 + n0; good = nn > 1e-15; a = a[good]; n1 = n1[good]; n0 = n0[good]; nn = nn[good]
    scale = (a * nn).sum() / nn.sum(); x = a / scale
    theta = np.array([np.log(n1.sum() / n0.sum()), 0.])
    for _ in range(80):
        prob = expit(theta[0] + theta[1] * x); wt = nn * prob * (1 - prob)
        grad = np.array([(n1 - nn * prob).sum(), ((n1 - nn * prob) * x).sum()])
        Hm = np.array([[wt.sum(), (wt * x).sum()], [(wt * x).sum(), (wt * x * x).sum()]])
        step = np.linalg.solve(Hm, grad); theta += step
        if np.max(np.abs(step)) < 1e-10: break
    return float(math.exp(theta[0]))


def limits(g1, g0):
    a = g1['age']; w1 = g1['w']; w0 = g0['w']; h1 = g1['H']; h0 = g0['H']

    def oratio(mask):
        return (w1[mask].sum() / h1) / (w0[mask].sum() / h0)
    full = np.ones(len(a), bool)
    d = dict(POR=oratio(full), aPOR=oratio(full) * (a * w0).sum() / w0.sum() / ((a * w1).sum() / w1.sum()),
             ROR30=oratio(a <= 30), ROR90=oratio(a <= 90))
    for label, window in [('EOR90', 90), ('EOR180', 180), ('EORInf', 1e100)]:
        ix = a <= window; d[label] = pop_fit(a[ix], w1[ix], w0[ix]) * h0 / h1
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=5000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(exist_ok=True, parents=True)
    rows = []
    for k, (name, typ1, typ0, rho) in enumerate(SC, 1):
        D1 = D0 * rho
        g1 = group(2 * I0, typ1, D1); g0 = group(I0, typ0, D0)
        lim = limits(g1, g0)
        x = changing_incidence_surveys(a.reps, entry_rate(2 * I0, D1), 2 * I0, MODE_CODE[typ1], D1,
                                       entry_rate(I0, D0), I0, MODE_CODE[typ0], D0, GAM, L, SEED + 100_000 * k, 6000)
        for ix, e in enumerate(ESTS):
            v = x[:, ix]; se = x[:, ix + 7]; ok = (v > 0) & np.isfinite(v) & np.isfinite(se)
            log = np.log(v[ok]); err = log - math.log(2)
            rows.append(dict(id=k, scenario=name, rho=rho, estimator=e, reps=a.reps, undefined=1 - ok.mean(),
                             limit=lim[e] / 2, gmr=np.exp(log.mean()) / 2, mcse_log=log.std(ddof=1) / math.sqrt(len(log)),
                             emp_se=log.std(ddof=1), mean_se=se[ok].mean(), coverage=np.mean(abs(err) <= Z975 * se[ok]),
                             EH1=g1['H'], EH0=g0['H'], EP1=g1['P'], EP0=g0['P'], mean_H1=x[:, 14].mean(),
                             mean_H0=x[:, 15].mean(), mean_P1=x[:, 16].mean(), mean_P0=x[:, 17].mean()))
        print('changing-incidence', k, name, flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'changing_incidence_python.csv', index=False, float_format='%.15g')


if __name__ == '__main__':
    main()
