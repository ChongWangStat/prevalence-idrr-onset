#!/usr/bin/env python3
"""Attempts to falsify the propositions numerically.

Each block takes one statement of the article and looks for a counterexample among duration
distributions that are NOT the four families of the simulation study: random piecewise-constant
hazards, random discrete distributions (durations with atoms), and mixtures with early exits.
A block fails if a statement is violated beyond numerical tolerance (or, for the Monte Carlo
blocks, beyond five standard errors). The script exits with a non-zero status on any failure.

These checks do not replace a proof. They are listed, statement by statement, in
PROOF_AUDIT_CHECKLIST.md, which also says what each one cannot detect.

Run: python code/python_falsify_propositions.py        (about one minute)
"""
import json, math, sys
import numpy as np
from scipy import optimize

RNG = np.random.default_rng(2026111301)       # replaced block by block in the main program
OUT = {}
FAIL = []


def check(name, ok, detail):
    OUT[name] = detail
    if not ok:
        FAIL.append(name)
    print(('PASS  ' if ok else 'FAIL  ') + name + ': ' + json.dumps(detail), flush=True)


def B(x):
    return 1.0 if x < 1e-12 else -math.expm1(-x) / x


# ------------------------------------------------------------------ duration distributions
class PiecewiseHazard:
    """Hazard constant on intervals [t_k, t_{k+1}); last interval extends to infinity."""
    def __init__(self, knots, rates):
        self.t = np.asarray(knots, float); self.r = np.asarray(rates, float)      # t[0] = 0
        self.cum = np.concatenate([[0.], np.cumsum(self.r[:-1] * np.diff(self.t))])

    def S(self, a):
        a = np.asarray(a, float); k = np.searchsorted(self.t, a, side='right') - 1
        return np.exp(-(self.cum[k] + self.r[k] * (a - self.t[k])))

    def R(self, w):
        """Restricted mean, exact."""
        tot = 0.0
        for k in range(len(self.t)):
            lo = self.t[k]; hi = self.t[k + 1] if k + 1 < len(self.t) else math.inf
            if w <= lo:
                break
            up = min(w, hi); s0 = math.exp(-self.cum[k]); r = self.r[k]
            tot += s0 * (up - lo) if r < 1e-14 else s0 * (-math.expm1(-r * (up - lo))) / r
        return tot

    def mean(self):
        return self.R(math.inf)

    def draw(self, size, rng):
        e = rng.exponential(size=size)                 # cumulative hazard at the event
        k = np.searchsorted(self.cum, e, side='right') - 1
        return self.t[k] + (e - self.cum[k]) / self.r[k]


def random_hazard(rng, scale=180., k=None, floor=0.0):
    k = k or int(rng.integers(2, 7))
    knots = np.concatenate([[0.], np.sort(rng.uniform(0, 3 * scale, k - 1))])
    rates = floor + rng.gamma(1.0, 1.0 / scale, k)
    rates[-1] = max(rates[-1], 0.2 / scale)            # finite mean
    return PiecewiseHazard(knots, rates)


class Discrete:
    """Durations with atoms: value t_k with probability p_k."""
    def __init__(self, t, p):
        o = np.argsort(t); self.t = np.asarray(t, float)[o]; self.p = np.asarray(p, float)[o] / np.sum(p)

    def S(self, a):
        a = np.atleast_1d(np.asarray(a, float))
        return np.array([self.p[self.t > x].sum() for x in a])       # right-continuous

    def R(self, w):
        return float((self.p * np.minimum(self.t, w)).sum())

    def mean(self):
        return float((self.p * self.t).sum())


def random_discrete(rng, scale=180.):
    k = int(rng.integers(2, 6))
    return Discrete(rng.uniform(0.02 * scale, 3 * scale, k), rng.dirichlet(np.ones(k)))


# ------------------------------------------------------------------ Proposition 1 and Proposition 3
def histories(rng, lam, I, gam, dur, L):
    """One survey of one group simulated as individual histories (numpy)."""
    m = rng.poisson(lam * L)
    u = L * rng.random(m); x = rng.exponential(1 / I, m); y = rng.exponential(1 / gam, m)
    healthy = int(((x > u) & (y > u)).sum())
    case = (x < y) & (x <= u)
    a = (u - x)[case]
    a = a[dur.draw(a.size, rng) > a]
    return healthy, a


def block_prop1_prop3():
    # exposed: 35% of cases leave within days, the others slowly; unexposed: piecewise hazard
    class Mix:
        def draw(self, size, rng):
            early = rng.random(size) < .35
            return np.where(early, rng.exponential(4.0, size), 40.0 + rng.gamma(2.0, 90.0, size))
        def S(self, a):
            from scipy.stats import gamma
            a = np.asarray(a, float)
            return .35 * np.exp(-a / 4.0) + .65 * np.where(a < 40, 1.0, gamma.sf(np.maximum(a - 40, 0), 2.0, scale=90.0))
    d1 = Mix(); d0 = PiecewiseHazard([0., 30., 200.], [0.001, 0.012, 0.004])
    grid = np.linspace(0, 20000, 400001)
    R1 = np.concatenate([[0.], np.cumsum((d1.S(grid[1:]) + d1.S(grid[:-1])) / 2 * np.diff(grid))])
    D1 = R1[-1]; D0 = d0.mean()
    I1, I0, gam, n, L, reps = .002, .001, .0005, 2000., 20000., 1500
    h1 = n / (1 + I1 * D1); h0 = n / (1 + I0 * D0)
    edges = np.array([0., 7., 30., 90., 365., 20000.])
    H = np.zeros((reps, 2)); P = np.zeros((reps, 2)); bins = np.zeros((reps, 2, len(edges) - 1)); ages1 = []
    for r in range(reps):
        for g, (hbar, I, d) in enumerate(((h1, I1, d1), (h0, I0, d0))):
            hh, a = histories(RNG, (I + gam) * hbar, I, gam, d, L)
            H[r, g] = hh; P[r, g] = a.size; bins[r, g] = np.histogram(a, edges)[0]
            if g == 0 and r < 150: ages1.append(a)
    se = lambda v: v.std(ddof=1) / math.sqrt(reps)
    z = []
    # Proposition 1(a), (b): means, Poisson variance, independence
    for g, (hbar, I, D) in enumerate(((h1, I1, D1), (h0, I0, D0))):
        z += [(H[:, g].mean() - hbar) / se(H[:, g]), (P[:, g].mean() - hbar * I * D) / se(P[:, g])]
        for v in (H[:, g], P[:, g]):                                   # variance / mean = 1
            z.append((v.var(ddof=1) / v.mean() - 1) / math.sqrt(2 / (reps - 1)))
        z.append(np.corrcoef(H[:, g], P[:, g])[0, 1] * math.sqrt(reps))
    # Proposition 3(a): expected cases per onset interval = Hbar * I * integral of S
    r1 = np.interp(edges, grid, R1); exp1 = h1 * I1 * np.diff(r1)
    exp0 = h0 * I0 * np.diff([d0.R(e) for e in edges])
    for k in range(len(edges) - 1):
        z += [(bins[:, 0, k].mean() - exp1[k]) / se(bins[:, 0, k]), (bins[:, 1, k].mean() - exp0[k]) / se(bins[:, 1, k])]
        z.append((bins[:, 0, k].var(ddof=1) / bins[:, 0, k].mean() - 1) / math.sqrt(2 / (reps - 1)))
    # disjoint onset intervals are independent (Proposition 1, used in Proposition 7)
    z.append(np.corrcoef(bins[:, 0, 0], bins[:, 0, 2])[0, 1] * math.sqrt(reps))
    # Proposition 1(c): distribution of the time since onset (Kolmogorov-Smirnov distance)
    a = np.sort(np.concatenate(ages1)); F = np.interp(a, grid, R1) / D1
    ks = max(np.max(np.arange(1, a.size + 1) / a.size - F), np.max(F - np.arange(a.size) / a.size))
    check('Prop 1 and Prop 3(a): individual histories, durations outside the four families',
          max(abs(v) for v in z) < 5 and ks * math.sqrt(a.size) < 2.0,
          dict(largest_abs_z=round(float(max(abs(v) for v in z)), 2), n_z=len(z), KS_times_sqrt_n=round(float(ks * math.sqrt(a.size)), 2),
               share_of_exposed_cases_leaving_within_7_days=round(float(1 - d1.S(7.0)), 3)))
    # Proposition 3(b): exposure odds of the cases in an onset interval / odds of the healthy
    ors = (bins[:, 0].sum(0) / H[:, 0].sum()) / (bins[:, 1].sum(0) / H[:, 1].sum())
    target = (I1 / I0) * np.diff(r1) / np.diff([d0.R(e) for e in edges])
    selog = np.sqrt(1 / bins[:, 0].sum(0) + 1 / bins[:, 1].sum(0) + 1 / H[:, 0].sum() + 1 / H[:, 1].sum())
    zz = np.log(ors / target) / selog
    check('Prop 3(a)-(b): odds ratio by onset interval = IDRR x ratio of integrals of S', np.max(np.abs(zz)) < 5,
          dict(odds_ratio=[round(float(v), 3) for v in ors], theory=[round(float(v), 3) for v in target], largest_abs_z=round(float(np.max(np.abs(zz))), 2)))


# ------------------------------------------------------------------ Proposition 2
def block_prop2():
    worst_b = 0.0; worst_d = 0.0
    for _ in range(400):
        d = random_discrete(RNG)
        # (b) E(A) = E(T^2) / (2 Dbar), computed from the density S(a)/Dbar of a step function
        t = np.concatenate([[0.], d.t]); s = np.array([d.p[d.t > x].sum() for x in t[:-1]])
        EA = float((s * (t[1:] ** 2 - t[:-1] ** 2) / 2).sum() / d.mean())
        worst_b = max(worst_b, abs(EA - (d.p * d.t ** 2).sum() / (2 * d.mean())))
        # (d) scale model: Pr(A1 <= w) = Pr(rho A0 <= w)
        rho = RNG.uniform(.3, 3.); d1 = Discrete(rho * d.t, d.p); w = RNG.uniform(0, 3 * 180.)
        worst_d = max(worst_d, abs(d1.R(w) / d1.mean() - d.R(w / rho) / d.mean()))
    check('Prop 2(b),(d): mean time since onset; scale model (discrete durations)', worst_b < 1e-9 and worst_d < 1e-12,
          dict(largest_error_b=worst_b, largest_error_d=worst_d, distributions=400))


# ------------------------------------------------------------------ Proposition 4
def trapezoid_weights(a):
    w = np.empty_like(a); w[1:-1] = (a[2:] - a[:-2]) / 2; w[0] = (a[1] - a[0]) / 2; w[-1] = (a[-1] - a[-2]) / 2
    return w


def logistic_limit(m1, m0, a):
    """Maximiser of the limiting log-likelihood on a grid a with intensities m1, m0 (trapezoid weights)."""
    w = trapezoid_weights(a); x = a / a[-1]
    th = np.array([math.log(np.sum(m1 * w) / np.sum(m0 * w)), 0.])
    for _ in range(200):
        p = 1 / (1 + np.exp(-(th[0] + th[1] * x))); q = (m1 + m0) * w
        g = np.array([np.sum((m1 - (m1 + m0) * p) * w), np.sum((m1 - (m1 + m0) * p) * w * x)])
        v = q * p * (1 - p); Hm = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
        step = np.linalg.solve(Hm, g); th = th + step
        if np.max(np.abs(step)) < 1e-13:
            break
    negdef = np.all(np.linalg.eigvalsh(Hm) > 0)
    return th[0], th[1] / a[-1], negdef


def block_prop4():
    # (c) scale model: min(1, rho) <= R1(w)/R0(w) <= max(1, rho), durations with atoms and without
    worst = 0.0
    for _ in range(3000):
        d0 = random_discrete(RNG) if RNG.random() < .5 else random_hazard(RNG)
        rho = RNG.uniform(.2, 4.); w = RNG.uniform(1., 800.)
        ratio = rho * d0.R(w / rho) / d0.R(w)                         # R1(w) = rho R0(w / rho)
        worst = max(worst, min(1, rho) - ratio, ratio - max(1, rho))
    check('Prop 4(c): limits of the recent-onset odds ratio under the scale model', worst < 1e-12, dict(largest_violation=worst, cases=3000))
    # (d) constant excess hazard on the window: the limit of the extrapolated odds ratio is the IDRR,
    #     whatever the hazard of the unexposed, the group sizes and the durations beyond the window
    worst = 0.0; ok = True
    for _ in range(300):
        d0 = random_hazard(RNG); w = RNG.uniform(20., 400.); beta = RNG.uniform(-.004, .02)
        idrr = RNG.uniform(.3, 4.); hr = RNG.uniform(.3, 3.)           # h1 / h0
        a = np.linspace(0, w, 20001); s0 = d0.S(a)
        if beta < 0 and np.any(np.diff(s0 * np.exp(-beta * a)) > 1e-15):
            continue                                                  # S1 must be a survival function
        th0, th1, negdef = logistic_limit(hr * idrr * s0 * np.exp(-beta * a), s0, a)
        worst = max(worst, abs(math.exp(th0) / hr / idrr - 1), abs(th1 + beta)); ok = ok and negdef
    check('Prop 4(d): extrapolated odds ratio under a constant excess hazard on the window', worst < 1e-7 and ok,
          dict(largest_relative_error_of_limit=worst, unique_maximiser=bool(ok)))
    # (d) converse direction: without the condition the limit differs from the IDRR (the statement is not vacuous)
    d0 = PiecewiseHazard([0., 60.], [0.0, 0.02]); d1 = PiecewiseHazard([0., 20.], [0.0, 0.03]); a = np.linspace(0, 180., 20001)
    th0, _, _ = logistic_limit(2.0 * d1.S(a), d0.S(a), a)
    check('Prop 4(d): limit differs from the IDRR when the logit is not linear', abs(math.exp(th0) / 2.0 - 1) > .02, dict(limit_over_IDRR=round(math.exp(th0) / 2.0, 4)))


# ------------------------------------------------------------------ Proposition 5
def block_prop5():
    # smooth hazards xi_j(a) = c0 + c1 a + c2 a^2 near onset. Each expansion is tested through its
    # remainder: the statement "leading term + o(w^k)" holds here with a remainder of order w^(k+1),
    # so halving the window must divide the remainder by about 2^(k+1).
    ws = (8.0, 4.0, 2.0, 1.0); rate = {'a_first_order': [], 'a_second_order': [], 'b_extrapolated': []}
    for _ in range(40):
        x0 = RNG.uniform(.001, .01); c = [(x0, RNG.uniform(-2e-5, 6e-5), RNG.uniform(0, 2e-7)) for _ in (0, 1)]
        equal = RNG.random() < .5
        if not equal:                                                 # unequal hazards at onset
            c[1] = (c[1][0] + RNG.uniform(.001, .01), c[1][1], c[1][2])
        d1 = c[1][0] - c[0][0]; d2 = c[1][1] - c[0][1]
        rem = {k: [] for k in rate}
        for w in ws:
            a = np.linspace(0, w, 8001)
            S = [np.exp(-(k[0] * a + k[1] * a ** 2 / 2 + k[2] * a ** 3 / 3)) for k in c]
            R = [np.trapezoid(s, a) for s in S]
            lr = math.log(R[1] / R[0])
            if not equal:
                rem['a_first_order'].append(abs(lr + d1 * w / 2))                       # o(w)
            else:
                rem['a_second_order'].append(abs(lr + d2 * w * w / 6))                  # o(w^2)
                th0, _, _ = logistic_limit(1.7 * S[1], S[0], a)
                rem['b_extrapolated'].append(abs(th0 - math.log(1.7) - d2 * w * w / 12))  # o(w^2)
        for k, v in rem.items():
            if v:
                rate[k] += [math.log2(v[i] / v[i + 1]) for i in range(len(v) - 1)]        # observed order of the remainder
    need = {'a_first_order': 2, 'a_second_order': 3, 'b_extrapolated': 3}
    ok = all(min(rate[k]) > need[k] - .25 for k in need)
    check('Prop 5(a),(b): expansions for short windows (order of the remainder)', ok,
          dict(smallest_observed_order={k: round(float(min(v)), 2) for k, v in rate.items()}, order_required_to_exceed={'a_first_order': 1, 'a_second_order': 2, 'b_extrapolated': 2}))
    # (c) variance of the intercept: 4 times that of the recent-onset odds ratio as w -> 0
    worst_c = 0.0
    for w in (8.0, 2.0, 0.5):
        a = np.linspace(0, w, 4001); x = a / w; m1 = 2.0 * np.exp(-.004 * a); m0 = np.exp(-.002 * a); wt = trapezoid_weights(a)
        th0, th1, _ = logistic_limit(m1, m0, a); p = 1 / (1 + np.exp(-(th0 + th1 * a))); v = (m1 + m0) * wt * p * (1 - p)
        info = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
        factor = np.linalg.inv(info)[0, 0] / (1 / np.sum(m1 * wt) + 1 / np.sum(m0 * wt))
        worst_c = max(worst_c, abs(factor - 4) / w)
    check('Prop 5(c): variance factor of the extrapolated odds ratio tends to 4', worst_c < .05, dict(largest_error_per_unit_window=worst_c))


# ------------------------------------------------------------------ Proposition 7 and the remark on ordered hazards
def block_prop7():
    # (a) the alternative (I', S') is a survival function and gives the same mean in every recording interval
    worst = 0.0; valid = True
    for _ in range(500):
        d = random_discrete(RNG) if RNG.random() < .4 else random_hazard(RNG)
        ed = np.concatenate([[0.], np.cumsum(RNG.uniform(5., 120., int(RNG.integers(1, 6)))), [1e6]]); b1 = ed[1]
        Sb = float(np.atleast_1d(d.S(b1))[0]); c = RNG.uniform(1.01, 50.)
        if Sb >= 1 - 1e-12:
            continue                                                  # the statement requires S(b1) < 1
        eta = (d.R(b1) - b1 * Sb) / (c - Sb); valid = valid and (0 < eta < b1) and (Sb / c <= 1)
        m = np.diff([d.R(e) for e in ed])                              # I = 1
        m2 = np.concatenate([[c * (eta + (b1 - eta) * Sb / c)], m[1:]])   # c * integral of S' = c * S / c beyond b1
        worst = max(worst, float(np.max(np.abs(m - m2) / np.maximum(m, 1e-300))))
    check('Prop 7(a): observationally equivalent incidence rates with interval-recorded onset', worst < 1e-10 and valid,
          dict(largest_relative_difference_of_interval_means=worst, alternative_is_a_survival_function=bool(valid)))
    # (b) bounds from Equation 13, three ways of generating pairs inside the envelope
    worst = 0.0; tight = math.inf; n = 0
    for _ in range(6000):
        w = RNG.uniform(5., 400.); D1 = RNG.uniform(0, .02); D0 = RNG.uniform(0, .02); a = np.linspace(0, w, 2001)
        kind = RNG.integers(0, 3)
        if kind == 0:                                                 # hazards that differ by at most Delta_1, Delta_0
            k = int(RNG.integers(1, 6)); kn = np.concatenate([[0.], np.sort(RNG.uniform(0, w, k - 1))])
            r0 = RNG.gamma(1., .01, k); r1 = np.maximum(r0 + RNG.uniform(-D0, D1, k), 0.)
            s0 = PiecewiseHazard(kn, r0).S(a); s1 = PiecewiseHazard(kn, r1).S(a)
        elif kind == 1:                                               # ratio anywhere in the cone, steep locally
            base = RNG.uniform(.03, .08); s0 = np.exp(-base * a)
            u = np.cumsum(RNG.uniform(-base, base, a.size) * np.gradient(a)); u = np.clip(u, -D1 * a, D0 * a)
            s1 = np.minimum.accumulate(np.minimum(s0 * np.exp(u), 1.0))
            s1 = np.clip(s1, s0 * np.exp(-D1 * a), s0 * np.exp(D0 * a))
            if np.any(np.diff(s1) > 1e-15):
                continue
        else:                                                         # a jump of the exposed survival inside the cone
            s0 = np.exp(-RNG.uniform(0, .01) * a); j = RNG.uniform(0, w)
            s1 = s0 * np.where(a < j, 1.0, np.exp(-D1 * j))
        if np.any(s1 < s0 * np.exp(-D1 * a) - 1e-12) or np.any(s1 > s0 * np.exp(D0 * a) + 1e-12):
            continue
        ratio = np.trapezoid(s1, a) / np.trapezoid(s0, a); n += 1
        worst = max(worst, B(D1 * w) - ratio, ratio - 1 / B(D0 * w)); tight = min(tight, ratio - B(D1 * w))
    a = np.linspace(0, 90., 200001)
    sharp = abs(np.trapezoid(np.exp(-.01 * a), a) / 90. - B(.9))
    check('Prop 7(b): bounds B(D1 w) <= R1/R0 <= 1/B(D0 w) inside the envelope; sharpness', worst < 1e-6 and sharp < 1e-9,
          dict(pairs=n, largest_violation=worst, closest_approach_to_lower_bound=tight, sharpness_error=sharp))
    # (b) the envelope is needed: just outside it the bound can fail
    w = 60.; a = np.linspace(0, w, 20001); s1 = np.where(a < 1., 1., .5); ratio = np.trapezoid(s1, a) / w
    check('Prop 7(b): bound fails outside the envelope (statement is not vacuous)', ratio < B(-math.log(.5) / w * w),
          dict(R_ratio=round(float(ratio), 4), bound_with_end_of_window_constant=round(B(-math.log(.5)), 4)))
    # ordered exit hazards: S1(w)/S0(w) <= R1(w)/R0(w) <= 1
    worst = 0.0; n = 0
    for _ in range(6000):
        w = RNG.uniform(5., 400.); a = np.linspace(0, w, 2001)
        if RNG.random() < .5:
            k = int(RNG.integers(1, 6)); kn = np.concatenate([[0.], np.sort(RNG.uniform(0, w, k - 1))])
            r0 = RNG.gamma(1., .01, k); s0 = PiecewiseHazard(kn, r0).S(a); s1 = PiecewiseHazard(kn, r0 + RNG.gamma(1., .01, k)).S(a)
        else:                                                         # with atoms: ratio decreasing by jumps
            s0 = np.exp(-RNG.uniform(0, .01) * a); g = np.ones_like(a)
            for j in RNG.uniform(0, w, int(RNG.integers(1, 5))):
                g = g * np.where(a < j, 1.0, RNG.uniform(.3, 1.))
            s1 = s0 * g
        ratio = np.trapezoid(s1, a) / np.trapezoid(s0, a); n += 1
        worst = max(worst, s1[-1] / s0[-1] - ratio, ratio - 1.0)
    check('Remark (ordered exit hazards): S1(w)/S0(w) <= R1(w)/R0(w) <= 1', worst < 1e-6, dict(pairs=n, largest_violation=worst))
    # (c) total variation bound c * Hbar * I * b for the alternative with exact onset times
    worst = -math.inf; valid = True
    for _ in range(500):
        d = random_hazard(RNG); c = RNG.uniform(1.01, 20.); b = RNG.uniform(.001, 30.); a = np.linspace(0, b, 4001)
        s = d.S(a); sp = 1 - (a / b) * (1 - s[-1] / c)
        valid = valid and np.all(np.diff(sp) <= 1e-15) and abs(sp[-1] - s[-1] / c) < 1e-12
        worst = max(worst, np.trapezoid(np.abs(s - c * sp), a) - c * b)   # Hbar * I = 1
    check('Prop 7(c): total variation bound for the alternative with exact onset times', worst <= 1e-9 and valid,
          dict(largest_excess_over_bound=worst, alternative_is_a_survival_function=bool(valid)))
    # tipping point: B(Delta* w) = 1/L (L > 1) and B(Delta* w) = U (U < 1)
    err = 0.0
    for _ in range(200):
        w = RNG.uniform(5., 400.); L = RNG.uniform(1.01, 5.); U = RNG.uniform(.05, .99)
        dl = optimize.brentq(lambda x: B(x * w) - 1 / L, 1e-12, 100. / w); du = optimize.brentq(lambda x: B(x * w) - U, 1e-12, 100. / w)
        err = max(err, abs(L * B(dl * w) - 1), abs(U / B(du * w) - 1))
    check('Tipping point: limit reaches 1 at the solution of B(Delta* w) = 1/L or U', err < 1e-9, dict(largest_error=err))


# ------------------------------------------------------------------ smaller statements of the supplement
def block_remarks():
    # sampling remark: multinomial variance of a log ratio of two cells
    err = 0.0
    for _ in range(200):
        th = RNG.dirichlet(np.ones(3)); m = RNG.uniform(50, 5000)
        lhs = (1 - th[0]) / (m * th[0]) + (1 - th[1]) / (m * th[1]) + 2 / m
        err = max(err, abs(lhs / (1 / (m * th[0]) + 1 / (m * th[1])) - 1))
    # two-window correction: variance of 4 N(w/2) - N(w) with independent Poisson increments
    a = RNG.poisson(40., 400000); b = RNG.poisson(55., 400000); c = 4 * a - (a + b)
    zvar = (c.var() / (8 * 40. + (40. + 55.)) - 1) * math.sqrt(400000 / 2)
    # two-window correction: {4 R(w/2) - R(w)} / w = 1 - S''(0) w^2 / 12 + o(w^2)
    worst = 0.0
    for w in (4., 2., 1.):
        x0, x1 = .006, 3e-5; S = lambda t: np.exp(-(x0 * t + x1 * t * t / 2))
        g = np.linspace(0, w, 8001); Rw = np.trapezoid(S(g), g); Rh = np.trapezoid(S(g[:4001]), g[:4001])
        s2 = x0 * x0 - x1                                              # S''(0)
        worst = max(worst, abs(((4 * Rh - Rw) / w - 1) / (-s2 * w * w / 12) - 1) / w)
    check('Remarks: multinomial variance identity; two-window correction', err < 1e-12 and abs(zvar) < 5 and worst < .05,
          dict(identity_error=err, z_of_variance=round(float(zvar), 2), expansion_error_per_unit_window=worst))


if __name__ == '__main__':
    for k, block in enumerate((block_prop2, block_prop4, block_prop5, block_prop7, block_remarks, block_prop1_prop3)):
        RNG = np.random.default_rng(2026111301 + k)      # each block has its own stream
        block()
    print(json.dumps({'status': 'FAIL' if FAIL else 'PASS', 'failed': FAIL, 'blocks': len(OUT)}))
    sys.exit(1 if FAIL else 0)
