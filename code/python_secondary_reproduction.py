#!/usr/bin/env python3
"""Checks of Propositions 1, 2 and S1 by individual-history simulation.

validation    one stationary group; counts and times since onset compared with
              Propositions 1 and 2 (supplementary table of simulated and theoretical moments)
trajectories  approach to stationarity after cases start to be counted (Proposition S1(a),
              supplementary figure); the surveys at successive days are of the same populations
immigration   prevalent cases that enter the population (Proposition S1(c), supplementary table)

All three use python_individual_histories.py: people enter, fall ill or emigrate, and
leave the prevalence pool after their own duration. No count or time since onset is
drawn from the distributions that the propositions derive.
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from python_individual_histories import (stationary_surveys, recent_start_paths, immigration_surveys, SHAPE_CODE)

SEED = 2026111400
I0 = .001; GAM = .0005; D0 = 180.
SHAPES = ('constant', 'uniform', 'exponential', 'lognormal')


def R(w, shape, D):
    """Restricted mean duration up to w."""
    w = np.asarray(w, dtype=float)
    if shape == 'constant':
        return np.minimum(w, D)
    if shape == 'uniform':
        a, b = .5 * D, 1.5 * D
        return np.where(w <= a, w, np.where(w >= b, D, a + ((b - a) ** 2 - (b - w) ** 2) / (2 * (b - a))))
    if shape == 'exponential':
        return -D * np.expm1(-w / D)
    z = np.maximum(w, 1.e-20); mu = np.log(D) - .5
    return w * stats.norm.sf(np.log(z) - mu) + D * stats.norm.cdf(np.log(z) - mu - 1)


def validation(reps, L=20000.):
    out = []; n = 2000
    cv2 = {'constant': 0, 'uniform': 1 / 12, 'exponential': 1, 'lognormal': math.e - 1}
    hbar = n / (1 + I0 * D0); lam = (I0 + GAM) * hbar
    for k, shape in enumerate(SHAPES):
        H, P, sumA, n30, kept = stationary_surveys(reps, lam, I0, GAM, D0, SHAPE_CODE[shape], D0 / 2, L,
                                                   SEED + 100_000 * k, 200, 1200)
        pooled = kept[kept >= 0]
        ks = stats.kstest(pooled, lambda x: R(x, shape, D0) / D0, method='asymp')
        out.append(dict(shape=shape, reps=reps, EH_theory=hbar, EH_sim=H.mean(), varH_sim=H.var(ddof=1),
                        EP_theory=hbar * I0 * D0, EP_sim=P.mean(), varP_sim=P.var(ddof=1),
                        cor_HP=np.corrcoef(H, P)[0, 1], PO_theory=I0 * D0, PO_sim=P.mean() / H.mean(),
                        EA_theory=D0 * (1 + cv2[shape]) / 2, EA_sim=sumA.sum() / P.sum(),
                        F30_theory=float(R(30, shape, D0) / D0), F30_sim=n30.sum() / P.sum(),
                        ks_stat=ks.statistic, ks_p=ks.pvalue, ks_n=len(pooled)))
        print('validation', shape, flush=True)
    return pd.DataFrame(out)


def trajectories(reps, L=12000.):
    out = []; step = 10.; nt = 90; w = 30.; n = 2000; idrr = 2; rho = .6
    tgrid = step * np.arange(1, nt + 1)
    for k, s in enumerate(SHAPES):
        vals = []
        for g, (I, D) in enumerate([(I0 * idrr, D0 * rho), (I0, D0)]):
            lam = (I + GAM) * n / (1 + I * D)
            vals.append(recent_start_paths(reps, lam, I, GAM, D, SHAPE_CODE[s], D / 2, L, step, nt, w,
                                           SEED + 10_000_000 + 100_000 * (2 * k + g)))
        (h1, p1, r1), (h0, p0, r0) = vals
        mH1, mP1, mR1 = h1.mean(axis=0), p1.mean(axis=0), r1.mean(axis=0)
        mH0, mP0, mR0 = h0.mean(axis=0), p0.mean(axis=0), r0.mean(axis=0)
        N0 = h0 + p0
        out.append(pd.DataFrame(dict(shape=s, t=tgrid, PO1=mP1 / mH1, PO0=mP0 / mH0, POR=(mP1 / mH1) / (mP0 / mH0),
                                     ROR=(mR1 / mH1) / (mR0 / mH0), PO1_theory=I0 * idrr * R(tgrid, s, D0 * rho),
                                     PO0_theory=I0 * R(tgrid, s, D0),
                                     POR_theory=idrr * R(tgrid, s, D0 * rho) / R(tgrid, s, D0),
                                     ROR_theory=idrr * R(np.minimum(tgrid, w), s, D0 * rho) / R(np.minimum(tgrid, w), s, D0),
                                     N0_mean=N0.mean(axis=0), N0_sd=N0.std(axis=0), H0_mean=mH0, P0_mean=mP0)))
        print('trajectory', s, flush=True)
    return pd.concat(out, ignore_index=True)


def immigration(reps, L=9000.):
    out = []; n = 5000; call = 0

    def gmean(x):
        ok = np.isfinite(x) & (x > 0)
        return float(np.exp(np.log(x[ok]).mean()))
    for rho in (1., .6):
        for mrel in (0., .25, .5, 1.):
            call += 1
            I1, D1 = 2 * I0, D0 * rho
            h1 = n / (1 + I1 * D1); h0 = n / (1 + I0 * D0)
            x = immigration_surveys(reps, (I1 + GAM) * h1, I1, D1, mrel * I0 * h1, (I0 + GAM) * h0, I0, D0,
                                    mrel * I0 * h0, GAM, L, SEED + 30_000_000 + 100_000 * call, 4000)
            m = mrel * I0
            out.append(dict(mrel=mrel, rho=rho, reps=reps, por_all_theory=(I1 * D1 + m * D1) / (I0 * D0 + m * D0),
                            por_all=gmean(x[:, 0]), por_local=gmean(x[:, 1]), apor_local=gmean(x[:, 2]),
                            ror30_local=gmean(x[:, 3])))
            print('immigration', rho, mrel, flush=True)
    return pd.DataFrame(out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--validation', type=int, default=5000)
    a.add_argument('--trajectories', type=int, default=2000)
    a.add_argument('--immigration', type=int, default=5000)
    a.add_argument('--out', type=Path, required=True)
    z = a.parse_args(); z.out.mkdir(parents=True, exist_ok=True)
    for name, fn, reps in [('validation', validation, z.validation), ('trajectories', trajectories, z.trajectories),
                           ('immigration', immigration, z.immigration)]:
        x = fn(reps)
        x.to_csv(z.out / (name + '_python.csv'), index=False, float_format='%.15g')
        print('wrote', name, len(x), flush=True)


if __name__ == '__main__':
    main()
