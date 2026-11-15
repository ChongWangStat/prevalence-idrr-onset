#!/usr/bin/env python3
"""Growth of the exposed population (Proposition S1(b) and (d)), by individual-history simulation.

Healthy people enter the exposed group at a rate that grows exponentially, lam1 * exp(kappa * s)
at time s before the survey; the unexposed group has a constant entry rate. Every person has a
latent onset time, a latent emigration time and a duration (python_individual_histories.py).
The large-sample limits are computed separately, by numerical integration of Proposition S1.
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import integrate
from python_individual_histories import two_group_surveys, SHAPE_CODE, Z975

SEED = 2026111500
I0 = .001; GAM = .0005; D0 = 180.
EST = ('POR', 'aPOR', 'ROR30', 'EOR90', 'EORInf')
COL = {'POR': 0, 'aPOR': 1, 'ROR30': 2, 'EOR90': 4, 'EORInf': 6}


def theoretical(shape, kap):
    """Limits of the estimators divided by the IDRR when the exposed entry rate grows at rate kap."""
    if shape == 'exponential':
        i0 = lambda p, w: ((1 / D0 + kap) ** (-p - 1) * math.gamma(p + 1) if w == math.inf else
                           integrate.quad(lambda a: a ** p * math.exp(-(kap + 1 / D0) * a), 0, w)[0])
        j0 = lambda p, w: ((1 / D0) ** (-p - 1) * math.gamma(p + 1) if w == math.inf else
                           integrate.quad(lambda a: a ** p * math.exp(-a / D0), 0, w)[0])
    else:
        def s(a):
            return 1. if a <= D0 * .5 else max(0., (1.5 * D0 - a) / D0)
        i0 = lambda p, w: integrate.quad(lambda a: a ** p * math.exp(-kap * a) * s(a), 0, min(1.5 * D0, w),
                                         points=[D0 / 2] if w > D0 / 2 else None)[0]
        j0 = lambda p, w: integrate.quad(lambda a: a ** p * s(a), 0, min(1.5 * D0, w),
                                         points=[D0 / 2] if w > D0 / 2 else None)[0]
    a = i0(0, math.inf); b = j0(0, math.inf)
    return dict(POR=a / b, aPOR=(a / b) * (j0(1, math.inf) / b) / (i0(1, math.inf) / a),
                ROR30=i0(0, 30) / j0(0, 30), EOR90=1., EORInf=1.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=5000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = []; n = 5000; L = 12000.; I1 = 2 * I0
    h1 = n / (1 + I1 * D0); h0 = n / (1 + I0 * D0)      # expected numbers healthy at the survey
    for k, (shape, kap) in enumerate([(sh, kp) for kp in (.001, .002) for sh in ('exponential', 'uniform')], 1):
        lim = theoretical(shape, kap)
        raw = two_group_surveys(a.reps, h1 * (I1 + GAM + kap), kap, I1, D0, h0 * (I0 + GAM), I0, D0, GAM,
                                SHAPE_CODE[shape], D0 / 2, D0 / 2, L, SEED + 100_000 * k, 6000)
        for e in EST:
            v = raw[:, COL[e]]; se = raw[:, COL[e] + 7]
            ok = (v > 0) & np.isfinite(v) & np.isfinite(se)
            le = np.log(v[ok]); err = le - math.log(2)
            rows.append(dict(shape=shape, kappa=kap, estimator=e, reps=a.reps, limit=lim[e], gmr=np.exp(le.mean()) / 2,
                             emp_se=le.std(ddof=1), coverage=np.mean(abs(err) <= Z975 * se[ok])))
        print('growth done', shape, kap, 'mean H1 %.1f (expected %.1f)' % (raw[:, 14].mean(), h1), flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'growth_python.csv', index=False, float_format='%.15g')


if __name__ == '__main__':
    main()
