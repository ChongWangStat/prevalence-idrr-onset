#!/usr/bin/env python3
"""Aims 1 to 3 and the stationarity check, by individual-history simulation.

Each replicate is one run of Algorithm S2 (python_individual_histories.algorithm_s2):
N0 healthy subjects on day 0, healthy immigrants at a constant rate, a latent emigration
time, a latent disease incidence time and a disease duration for every subject, and a
prevalence study on day 2000. The duration distribution (constant, uniform, exponential)
is drawn subject by subject.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from python_individual_histories import algorithm_s2, algorithm_s2_monitor, SHAPE_CODE

SEED = 2026111300
TMAX = 2000.0
# incidence rate, emigration rate, mean duration (per day, days): Groups 1, 2 and 3
GROUPS = ((.036, .004, 7.), (.018, .002, 7.), (.018, .002, 10.))
HALF = 3.0   # uniform durations on (mean - 3, mean + 3) days


def study(reps, monitoring):
    rows = []
    call = 0
    for shape in ('constant', 'uniform', 'exponential'):
        for n in (100, 200, 500, 1000):
            counts = []
            for inc, emig, D in GROUPS:
                call += 1
                counts.append(algorithm_s2(reps, n, inc, emig, D, SHAPE_CODE[shape], HALF, TMAX, SEED + 100_000 * call))
            h1, p1 = counts[0]
            for label, ix in (('1-2', 1), ('1-3', 2)):
                h0, p0 = counts[ix]
                with np.errstate(divide='ignore', invalid='ignore'):
                    po = (p1 / h1) / (p0 / h0)
                good = (po > 0) & np.isfinite(po)
                v = po[good]
                rows.append(dict(comparison=label, shape=shape, N0=n, reps=reps, undefined=1 - good.mean(),
                                 theory=2 * GROUPS[0][2] / GROUPS[ix][2], mean=v.mean(), se=v.std(ddof=1),
                                 bias=v.mean() - 2, rmse=np.sqrt(((v - 2) ** 2).mean()), median=np.median(v),
                                 gmean=np.exp(np.log(v).mean()), mean_H1=h1.mean(), mean_P1=p1.mean(),
                                 mean_H0=h0.mean(), mean_P0=p0.mean()))
            print('Aims 1-3', shape, n, flush=True)
    stat = []
    n = 500
    step, nday = 50.0, 41
    for gi, (inc, emig, D) in enumerate(GROUPS, 1):
        N, P = algorithm_s2_monitor(monitoring, n, inc, emig, D, SHAPE_CODE['constant'], HALF, step, nday,
                                    SEED + 50_000_000 + 100_000 * gi)
        pbar = n * inc * D / (1 + inc * D)
        for i in range(nday):
            stat.append(dict(group=gi, day=int(i * step), reps=monitoring, mean_N=N[:, i].mean(), sd_N=N[:, i].std(ddof=1),
                             mean_P=P[:, i].mean(), sd_P=P[:, i].std(ddof=1), expected_N=n, expected_P=pbar))
    return pd.DataFrame(rows), pd.DataFrame(stat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=10000)
    p.add_argument('--monitoring', type=int, default=1000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    x, y = study(a.reps, a.monitoring)
    x.to_csv(a.out / 'original_summary_python.csv', index=False, float_format='%.15g')
    y.to_csv(a.out / 'original_stationarity_python.csv', index=False, float_format='%.15g')
    print(x[['comparison', 'shape', 'N0', 'mean', 'theory']].to_string(index=False))


if __name__ == '__main__':
    main()
