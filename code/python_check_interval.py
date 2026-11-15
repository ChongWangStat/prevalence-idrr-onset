#!/usr/bin/env python3
"""Numerical checks of the statements that accompany Proposition 7 in the supplement.

(a) Non-identification with interval-recorded onset: the alternative (I', S') gives the same
    expected count in every recording interval.
(b) Bounds: B(D1 w) <= R1(w)/R0(w) <= 1/B(D0 w) for exponential durations, and sharpness.
(c) Exact onset times: the total variation bound  c * Hbar * I * b.
(d) Window w* = (z^2 v / Dbar^2)^(1/3) against the window that minimizes the expected length
    of the interval with Wald limits, computed without the approximations.
(e) Extrapolated odds ratio from recorded intervals: factor exp{(l1^2 - l0^2) b^2 / 24}.
"""
import math, json
import numpy as np
from scipy import integrate, optimize, stats
from python_interval_reproduction import grouped_logistic

def B(x): return 1. if x < 1e-12 else -math.expm1(-x) / x
out = {}

# (a)
b = 30.; I = .001; D = 180.; edges = np.arange(0, 361, b)
S = lambda a: math.exp(-a / D); R = lambda w: D * (1 - math.exp(-w / D))
worst = 0.
for c in (1.5, 4.):
    eta = (R(b) - b * S(b)) / (c - S(b))
    m1 = I * np.diff([R(e) for e in edges])
    m2 = [c * I * (eta + (b - eta) * S(b) / c)] + [c * I * (R(edges[k + 1]) - R(edges[k])) / c for k in range(1, len(edges) - 1)]
    assert 0 < eta < b
    worst = max(worst, float(np.max(np.abs(m1 - np.array(m2)))))
out['a_largest_difference_of_interval_means'] = worst
assert worst < 1e-12

# (b)
for D1 in (15., 50., 108., 180., 400.):
    for w in (30., 60., 90.):
        d = 1 / D1 - 1 / 180.; r = (D1 * (1 - math.exp(-w / D1))) / (180. * (1 - math.exp(-w / 180.)))
        assert B(max(d, 0) * w) - 1e-12 <= r <= 1 / B(max(-d, 0) * w) + 1e-12
out['b_sharpness'] = [integrate.quad(lambda a: math.exp(-.01 * a), 0, 30)[0] / 30, B(.3)]

# (c)
tv = {}
for n in (1000, 5000):
    for bb in (.1, .01):
        I = .002; D = 108.; c = 2.; Hbar = n / (1 + I * D)
        S2 = lambda a: 1 - (a / bb) * (1 - math.exp(-bb / D) / c)
        l1 = Hbar * integrate.quad(lambda a: abs(I * math.exp(-a / D) - c * I * S2(a)), 0, bb, epsabs=1e-13)[0]
        assert l1 <= c * Hbar * I * bb
        tv[f'n={n}, b={bb}'] = round(l1, 4)
out['c_total_variation'] = tv

# (d)
I0, I1, D0, D1 = .001, .002, 180., 108.; z = stats.norm.ppf(.975); dd = 1 / D1 - 1 / D0
win = {}
for n in (1000, 5000, 50000):
    H1 = n / (1 + I1 * D1); H0 = n / (1 + I0 * D0)
    v = 1 / (H1 * I1) + 1 / (H0 * I0)
    wstar = (z * z * v / dd ** 2) ** (1 / 3)
    def length(w):
        p1 = H1 * I1 * D1 * (1 - math.exp(-w / D1)); p0 = H0 * I0 * D0 * (1 - math.exp(-w / D0))
        return 2 * z * math.sqrt(1 / p1 + 1 / p0 + 1 / H1 + 1 / H0) - 2 * math.log(B(dd * w))
    best = optimize.minimize_scalar(length, bounds=(1, 400), method='bounded')
    win[f'n={n}'] = dict(w_star=round(wstar, 1), exact_minimizer=round(best.x, 1), excess_length_percent=round(100 * (length(wstar) / best.fun - 1), 3))
out['d_window'] = win

# (e)
def interval_limit(l1, l0, b, nint):
    edges = b * np.arange(nint + 1)
    f = lambda l, x0, x1: (math.exp(-l * x0) - math.exp(-l * x1)) / l if l > 0 else x1 - x0
    c1 = np.array([f(l1, edges[k], edges[k + 1]) for k in range(nint)]) * 1e6
    c0 = np.array([f(l0, edges[k], edges[k + 1]) for k in range(nint)]) * 1e6
    b0, _ = grouped_logistic((edges[1:] + edges[:-1]) / 2, c1, c0)
    return math.exp(b0)
out['e_interval_factor'] = {
    'means 108 and 180 days, 30-day intervals': dict(limit=round(interval_limit(1 / 108, 1 / 180, 30., 3), 4), approximation=round(math.exp((1 / 108 ** 2 - 1 / 180 ** 2) * 900 / 24), 4)),
    'mean 15 days and no exits, 30-day intervals': dict(limit=round(interval_limit(1 / 15, 0., 30., 3), 3), approximation=round(math.exp((1 / 15 ** 2) * 900 / 24), 3))}
print(json.dumps(out, indent=1))
