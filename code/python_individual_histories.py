#!/usr/bin/env python3
"""Individual-history simulators of the dynamic population.

Every function in this module follows individuals: each person enters the
population, has a latent time to disease onset and a latent time to emigration,
and, if the disease starts first, stays in the prevalence pool for a duration
drawn from the stated distribution. A prevalence survey then counts who is
healthy and who is diseased on the survey day and records the time since onset
of each prevalent case.

Nothing here samples from the distributions derived in Propositions 1, 2 or 6.
These simulators are therefore used for Aims 1 to 3 and for the checks of those
propositions. (The 56-scenario comparison of estimators samples the stationary
survey distribution of Proposition 1 directly; see python_main_reproduction.py.
main_design_surveys below repeats that comparison with individual histories.)

Replicate r of a call uses the random seed  seed + r,  so results do not depend
on the number of threads.
"""
import math
import numpy as np
from numba import njit, prange
from python_main_reproduction import linear_logistic, analyse_survey

CONSTANT, UNIFORM, EXPONENTIAL, LOGNORMAL = 0, 1, 2, 3
SHAPE_CODE = {'constant': CONSTANT, 'uniform': UNIFORM,
              'exponential': EXPONENTIAL, 'lognormal': LOGNORMAL}
Z975 = 1.959963984540054


@njit(cache=True)
def draw_duration(code, D, half):
    """Disease duration with mean D. Uniform durations lie in (D - half, D + half);
    log-normal durations have standard deviation 1 on the log scale."""
    if code == CONSTANT:
        return D
    if code == UNIFORM:
        return D - half + 2.0 * half * np.random.random()
    if code == EXPONENTIAL:
        return -D * math.log(1.0 - np.random.random())
    return math.exp(math.log(D) - 0.5 + np.random.normal())


# ---------------------------------------------------------------------------
# Aims 1 to 3: Algorithm S2 (population followed forward from day 0)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _s2_population(N0, I, gam, D, code, half, tmax):
    """One run of Algorithm S2: numbers healthy and diseased on day tmax."""
    a = N0 / (1.0 + I * D) * (I + gam)        # immigration rate, H * alpha
    h = 0
    p = 0
    t0 = 0.0
    k = 0
    while True:
        if k >= N0:                           # after the initial population: next immigrant
            t0 += np.random.exponential(1.0 / a)
            if t0 >= tmax:
                break
        k += 1
        t_emig = t0 + np.random.exponential(1.0 / gam)
        td_start = t0 + np.random.exponential(1.0 / I)
        if td_start < t_emig:                 # the subject gets the disease
            if td_start > tmax:
                h += 1
            elif td_start + draw_duration(code, D, half) > tmax:
                p += 1
        elif t_emig > tmax:
            h += 1
    return h, p


@njit(cache=True, parallel=True)
def algorithm_s2(reps, N0, I, gam, D, code, half, tmax, seed):
    """Algorithm S2. N0 healthy subjects at day 0; healthy immigrants arrive at the
    constant rate a = H * alpha with alpha = I + gam and H = N0 / (1 + I * D);
    survey on day tmax. Returns the numbers healthy and diseased on that day."""
    H = np.zeros(reps, np.int64)
    P = np.zeros(reps, np.int64)
    for r in prange(reps):
        np.random.seed(seed + r)
        h, p = _s2_population(N0, I, gam, D, code, half, tmax)
        H[r] = h
        P[r] = p
    return H, P


@njit(cache=True)
def _s2_monitored(N0, I, gam, D, code, half, step, nday, Nrow, Prow):
    tmax = step * (nday - 1)
    a = N0 / (1.0 + I * D) * (I + gam)
    dH = np.zeros(nday + 1, np.int64)         # difference arrays over the monitoring days
    dP = np.zeros(nday + 1, np.int64)
    t0 = 0.0
    k = 0
    while True:
        if k >= N0:
            t0 += np.random.exponential(1.0 / a)
            if t0 > tmax:
                break
        k += 1
        t_emig = t0 + np.random.exponential(1.0 / gam)
        td_start = t0 + np.random.exponential(1.0 / I)
        h_end = min(t_emig, td_start)
        i0 = int(math.ceil(t0 / step))            # healthy on days in [t0, h_end)
        i1 = min(int(math.ceil(h_end / step)), nday)
        if i1 > i0:
            dH[i0] += 1
            dH[i1] -= 1
        if td_start < t_emig and td_start <= tmax:
            td_end = td_start + draw_duration(code, D, half)
            j0 = int(math.ceil(td_start / step))  # diseased on days in [td_start, td_end)
            j1 = min(int(math.ceil(td_end / step)), nday)
            if j1 > j0:
                dP[j0] += 1
                dP[j1] -= 1
    ch = 0
    cp = 0
    for i in range(nday):
        ch += dH[i]
        cp += dP[i]
        Nrow[i] = ch + cp
        Prow[i] = cp


@njit(cache=True, parallel=True)
def algorithm_s2_monitor(reps, N0, I, gam, D, code, half, step, nday, seed):
    """Algorithm S2 with the population size and the prevalence pool recorded on
    days 0, step, 2*step, ..., (nday-1)*step in each simulated population."""
    N = np.zeros((reps, nday), np.int64)
    P = np.zeros((reps, nday), np.int64)
    for r in prange(reps):
        np.random.seed(seed + r)
        _s2_monitored(N0, I, gam, D, code, half, step, nday, N[r], P[r])
    return N, P


# ---------------------------------------------------------------------------
# One group observed at a survey on day 0; the population started empty L days earlier
# ---------------------------------------------------------------------------
@njit(cache=True)
def group_history(lam, kappa, I, gam, D, code, half, L, ages):
    """Entries are a Poisson process on (-L, 0] with rate lam * exp(kappa * s) at time s
    (kappa = 0: constant rate). Returns the number healthy at day 0 and the number of
    prevalent cases, whose times since onset are written to ages[:p]."""
    low = math.exp(-kappa * L)
    if kappa == 0.0:
        m = np.random.poisson(lam * L)
    else:
        m = np.random.poisson(lam * (1.0 - low) / kappa)
    h = 0
    p = 0
    for _ in range(m):
        if kappa == 0.0:
            u = L * np.random.random()                                  # time since entry
        else:
            u = -math.log(low + np.random.random() * (1.0 - low)) / kappa
        x = np.random.exponential(1.0 / I)                              # entry to onset
        y = np.random.exponential(1.0 / gam)                            # entry to emigration
        if x > u and y > u:
            h += 1
        elif x < y and x <= u:
            a = u - x                                                   # time since onset
            if draw_duration(code, D, half) > a:
                if p < ages.shape[0]:
                    ages[p] = a
                p += 1
    return h, p


@njit(cache=True, parallel=True)
def stationary_surveys(reps, lam, I, gam, D, code, half, L, seed, keep, cap):
    """Surveys of one group with constant rates. Returns H, P, the sum of the times since
    onset, the number of times of at most 30 days, and the times of the first `keep`
    replicates (padded with -1)."""
    H = np.zeros(reps, np.int64)
    P = np.zeros(reps, np.int64)
    sumA = np.zeros(reps)
    n30 = np.zeros(reps, np.int64)
    kept = np.full((keep, cap), -1.0)
    for r in prange(reps):
        np.random.seed(seed + r)
        ages = np.empty(cap)
        h, p = group_history(lam, 0.0, I, gam, D, code, half, L, ages)
        H[r] = h
        P[r] = p
        s = 0.0
        c = 0
        for i in range(min(p, cap)):
            s += ages[i]
            if ages[i] <= 30.0:
                c += 1
            if r < keep:
                kept[r, i] = ages[i]
        sumA[r] = s
        n30[r] = c
    return H, P, sumA, n30, kept


# ---------------------------------------------------------------------------
# Estimators from the data of one survey (same formulas as python_main_reproduction.py)
# ---------------------------------------------------------------------------
@njit(cache=True)
def survey_estimates(h1, h0, a1, p1, a0, p0, out):
    """out[0:7]  = POR, aPOR, ROR(30), ROR(90), EOR(90), EOR(180), EOR(all cases)
       out[7:14] = their estimated standard errors on the log scale (nan if undefined)."""
    for i in range(14):
        out[i] = np.nan
    if h1 < 1 or h0 < 1 or p1 < 1 or p0 < 1:
        return
    por = (p1 / h1) / (p0 / h0)
    v = 1.0 / p1 + 1.0 / h1 + 1.0 / p0 + 1.0 / h0
    out[0] = por
    out[7] = math.sqrt(v)
    if p1 > 1 and p0 > 1:
        m1 = np.mean(a1[:p1])
        m0 = np.mean(a0[:p0])
        s1 = np.var(a1[:p1]) * p1 / (p1 - 1)
        s0 = np.var(a0[:p0]) * p0 / (p0 - 1)
        out[1] = por * m0 / m1
        out[8] = math.sqrt(v + s1 / (p1 * m1 * m1) + s0 / (p0 * m0 * m0))
    for k in range(2):
        w = 30.0 if k == 0 else 90.0
        r1 = 0
        r0 = 0
        for i in range(p1):
            if a1[i] <= w:
                r1 += 1
        for i in range(p0):
            if a0[i] <= w:
                r0 += 1
        if r1 > 0 and r0 > 0:
            out[2 + k] = (r1 / h1) / (r0 / h0)
            out[9 + k] = math.sqrt(1.0 / r1 + 1.0 / h1 + 1.0 / r0 + 1.0 / h0)
    for k in range(3):
        w = 90.0 if k == 0 else (180.0 if k == 1 else 1e100)
        n1 = 0
        n0 = 0
        for i in range(p1):
            if a1[i] <= w:
                n1 += 1
        for i in range(p0):
            if a0[i] <= w:
                n0 += 1
        if n1 < 2 or n0 < 2:
            continue
        x = np.empty(n1 + n0)
        y = np.empty(n1 + n0, np.int64)
        j = 0
        for i in range(p1):
            if a1[i] <= w:
                x[j] = a1[i]
                y[j] = 1
                j += 1
        for i in range(p0):
            if a0[i] <= w:
                x[j] = a0[i]
                y[j] = 0
                j += 1
        b0, var0, _ = linear_logistic(x, y)
        if math.isfinite(var0):
            out[4 + k] = math.exp(b0) * h0 / h1
            out[11 + k] = math.sqrt(var0 + 1.0 / h1 + 1.0 / h0)


@njit(cache=True, parallel=True)
def two_group_surveys(reps, lam1, kappa1, I1, D1, lam0, I0, D0, gam, code, half1, half0, L, seed, cap):
    """Surveys of an exposed group (entry rate lam1 * exp(kappa1 * s)) and an unexposed
    group (constant entry rate lam0). Returns a (reps, 18) array: 14 columns of
    survey_estimates, then H1, H0, P1, P0."""
    out = np.full((reps, 18), np.nan)
    for r in prange(reps):
        np.random.seed(seed + r)
        a1 = np.empty(cap)
        a0 = np.empty(cap)
        h1, p1 = group_history(lam1, kappa1, I1, gam, D1, code, half1, L, a1)
        h0, p0 = group_history(lam0, 0.0, I0, gam, D0, code, half0, L, a0)
        out[r, 14] = h1
        out[r, 15] = h0
        out[r, 16] = p1
        out[r, 17] = p0
        if p1 <= cap and p0 <= cap:
            survey_estimates(h1, h0, a1, p1, a0, p0, out[r, :14])
    return out


# ---------------------------------------------------------------------------
# Immigration of prevalent cases (Proposition S1(c))
# ---------------------------------------------------------------------------
@njit(cache=True)
def _immigration_one(lam1, I1, D1, mu1, lam0, I0, D0, mu0, gam, L, cap, row):
    a1 = np.empty(cap)
    a0 = np.empty(cap)
    h1, p1 = group_history(lam1, 0.0, I1, gam, D1, EXPONENTIAL, 0.0, L, a1)
    h0, p0 = group_history(lam0, 0.0, I0, gam, D0, EXPONENTIAL, 0.0, L, a0)
    m1 = 0
    for _ in range(np.random.poisson(mu1 * L)):       # immigrant cases of the exposed group
        if np.random.exponential(D1) > L * np.random.random():
            m1 += 1
    m0 = 0
    for _ in range(np.random.poisson(mu0 * L)):
        if np.random.exponential(D0) > L * np.random.random():
            m0 += 1
    if h1 < 1 or h0 < 1 or p1 < 1 or p0 < 1 or p1 > cap or p0 > cap:
        return
    local = (p1 / h1) / (p0 / h0)
    row[0] = ((p1 + m1) / h1) / ((p0 + m0) / h0)
    row[1] = local
    row[2] = local * np.mean(a0[:p0]) / np.mean(a1[:p1])
    r1 = 0
    r0 = 0
    for i in range(p1):
        if a1[i] <= 30.0:
            r1 += 1
    for i in range(p0):
        if a0[i] <= 30.0:
            r0 += 1
    if r1 > 0 and r0 > 0:
        row[3] = (r1 / h1) / (r0 / h0)


@njit(cache=True, parallel=True)
def immigration_surveys(reps, lam1, I1, D1, mu1, lam0, I0, D0, mu0, gam, L, seed, cap):
    """Exponential durations. In group j prevalent cases also arrive as a Poisson stream
    with rate mu_j and stay for an exponential time with mean D_j.
    Columns: POR from all prevalent cases; POR, aPOR and ROR(30) from the cases whose
    onset occurred in the study population."""
    out = np.full((reps, 4), np.nan)
    for r in prange(reps):
        np.random.seed(seed + r)
        _immigration_one(lam1, I1, D1, mu1, lam0, I0, D0, mu0, gam, L, cap, out[r])
    return out


# ---------------------------------------------------------------------------
# Approach to stationarity (Proposition S1(a))
# ---------------------------------------------------------------------------
@njit(cache=True)
def _recent_start_one(lam, I, gam, D, code, half, L, step, nt, w, Hrow, Prow, Rrow):
    tmax = step * nt
    dH = np.zeros(nt + 2, np.int64)
    dP = np.zeros(nt + 2, np.int64)
    dR = np.zeros(nt + 2, np.int64)
    for _ in range(np.random.poisson(lam * (L + tmax))):
        s = -L + (L + tmax) * np.random.random()          # entry time
        onset = s + np.random.exponential(1.0 / I)
        emig = s + np.random.exponential(1.0 / gam)
        h_end = min(onset, emig)
        if h_end > 0.0:
            i0 = max(1, int(math.ceil(s / step)))         # healthy on survey days in [s, h_end)
            i1 = min(int(math.ceil(h_end / step)), nt + 1)
            if i1 > i0:
                dH[i0] += 1
                dH[i1] -= 1
        if onset < emig and onset >= 0.0 and onset < tmax:
            end = onset + draw_duration(code, D, half)
            j0 = max(1, int(math.ceil(onset / step)))     # diseased on days in [onset, end)
            j1 = min(int(math.ceil(end / step)), nt + 1)
            if j1 > j0:
                dP[j0] += 1
                dP[j1] -= 1
                k1 = min(j1, int(math.floor((onset + w) / step)) + 1)   # onset within w days
                if k1 > j0:
                    dR[j0] += 1
                    dR[k1] -= 1
    ch = 0
    cp = 0
    cr = 0
    for i in range(1, nt + 1):
        ch += dH[i]
        cp += dP[i]
        cr += dR[i]
        Hrow[i - 1] = ch
        Prow[i - 1] = cp
        Rrow[i - 1] = cr


@njit(cache=True, parallel=True)
def recent_start_paths(reps, lam, I, gam, D, code, half, L, step, nt, w, seed):
    """The healthy pool has been running for L days when cases start to be counted on day 0:
    a person whose disease began before day 0 has left the healthy pool and is not a
    prevalent case. Surveys on days step, 2*step, ..., nt*step of the same population.
    Returns H, P and the number of cases with onset in the last w days, each (reps, nt)."""
    H = np.zeros((reps, nt), np.int64)
    P = np.zeros((reps, nt), np.int64)
    R = np.zeros((reps, nt), np.int64)
    for r in prange(reps):
        np.random.seed(seed + r)
        _recent_start_one(lam, I, gam, D, code, half, L, step, nt, w, H[r], P[r], R[r])
    return H, P, R


# ---------------------------------------------------------------------------
# Incidence rates that change over calendar time (Proposition 6)
# ---------------------------------------------------------------------------
NONE, RISE, FALL, STEP = 0, 1, 2, 3
MODE_CODE = {'none': NONE, 'rise': RISE, 'fall': FALL, 'step': STEP}
G_YEAR = math.log(2.0) / 365.0


@njit(cache=True)
def onset_time(s, e, inow, mode):
    """Calendar time of disease onset for a person who enters at time s < 0 and whose
    cumulative incidence hazard at onset is e (a unit exponential draw). The incidence
    rate at the survey (day 0) is inow. RISE: the rate doubled during the last 365 days at
    a constant relative rate; FALL: it halved; STEP: it doubled 180 days before the survey.
    Before the period of change the rate was constant. Returns +inf if onset is after day 0."""
    if mode == NONE:
        t = s + e / inow
        return t if t < 0.0 else np.inf
    if mode == STEP:
        c = 0.5 * inow
        if s < -180.0:
            t = s + e / c
            if t < -180.0:
                return t
            e -= c * (-180.0 - s)
            s = -180.0
        t = s + e / inow
        return t if t < 0.0 else np.inf
    g = G_YEAR if mode == RISE else -G_YEAR      # rate = inow * exp(g * t) on [-365, 0]
    c = inow * math.exp(-g * 365.0)              # constant rate before day -365
    if s < -365.0:
        t = s + e / c
        if t < -365.0:
            return t
        e -= c * (-365.0 - s)
        s = -365.0
    z = math.exp(g * s) + g * e / inow           # exp(g * t) at onset
    if z <= 0.0:
        return np.inf
    t = math.log(z) / g
    return t if t < 0.0 else np.inf


@njit(cache=True)
def changing_group(lam, inow, mode, gam, D, L, ages):
    """One group with a time-dependent incidence rate and exponential durations with mean D."""
    h = 0
    p = 0
    for _ in range(np.random.poisson(lam * L)):
        s = -L * np.random.random()
        onset = onset_time(s, np.random.exponential(1.0), inow, mode)
        emig = s + np.random.exponential(1.0 / gam)
        if onset > 0.0 and emig > 0.0:
            h += 1
        elif onset < emig and onset <= 0.0:
            a = -onset
            if np.random.exponential(D) > a:
                if p < ages.shape[0]:
                    ages[p] = a
                p += 1
    return h, p


@njit(cache=True, parallel=True)
def changing_incidence_surveys(reps, lam1, inow1, mode1, D1, lam0, inow0, mode0, D0, gam, L, seed, cap):
    """Returns a (reps, 18) array laid out as in two_group_surveys."""
    out = np.full((reps, 18), np.nan)
    for r in prange(reps):
        np.random.seed(seed + r)
        a1 = np.empty(cap)
        a0 = np.empty(cap)
        h1, p1 = changing_group(lam1, inow1, mode1, gam, D1, L, a1)
        h0, p0 = changing_group(lam0, inow0, mode0, gam, D0, L, a0)
        out[r, 14] = h1
        out[r, 15] = h0
        out[r, 16] = p1
        out[r, 17] = p0
        if p1 <= cap and p0 <= cap:
            survey_estimates(h1, h0, a1, p1, a0, p0, out[r, :14])
    return out


# ---------------------------------------------------------------------------
# The design of the comparison of the estimators (Aim 4), by individual histories
# ---------------------------------------------------------------------------
@njit(cache=True)
def group_history_excess(lam, I, gam, D, code, half, beta, L, ages):
    """Stationary group as in group_history, with an optional constant excess hazard beta of
    leaving the prevalence pool (duration = minimum of a duration from the stated distribution
    and an independent exponential time with rate beta; scenario set C). Returns the number
    healthy at day 0 and the number of prevalent cases, whose times since onset are written
    to ages[:p]."""
    h = 0
    p = 0
    for _ in range(np.random.poisson(lam * L)):
        u = L * np.random.random()                      # time since entry
        x = np.random.exponential(1.0 / I)              # entry to onset
        y = np.random.exponential(1.0 / gam)            # entry to emigration
        if x > u and y > u:
            h += 1
        elif x < y and x <= u:
            a = u - x                                   # time since onset
            t = draw_duration(code, D, half)
            if beta > 0.0:
                t = min(t, np.random.exponential(1.0 / beta))
            if t > a:
                if p < ages.shape[0]:
                    ages[p] = a
                p += 1
    return h, p


@njit(cache=True)
def _main_design_one(n, code1, code0, rho, idrr, beta, I0, D0, gam, L, cap, row):
    i1 = I0 * idrr
    d1 = D0 if beta > 0.0 else D0 * rho                 # set C: same distribution, excess hazard beta
    h1bar = n / (1.0 + i1 * D0 * rho)
    h0bar = n / (1.0 + I0 * D0)
    a1 = np.empty(cap)
    a0 = np.empty(cap)
    h1, p1 = group_history_excess((i1 + gam) * h1bar, i1, gam, d1, code1, d1 / 2.0, beta, L, a1)
    h0, p0 = group_history_excess((I0 + gam) * h0bar, I0, gam, D0, code0, D0 / 2.0, 0.0, L, a0)
    if p1 > cap or p0 > cap:
        row[23] = -1.0                                  # flag: more prevalent cases than the buffer holds
        return
    analyse_survey(h1, h0, a1[:p1].copy(), a0[:p0].copy(), row)


@njit(cache=True, parallel=True)
def main_design_surveys(reps, n, code1, code0, rho, idrr, beta, I0, D0, gam, L, seed, cap):
    """Surveys of two stationary groups simulated as individual histories, analysed by the
    function that analyses the surveys of python_main_reproduction.sim_batch.
    Same parameterization as sim_batch; returns the same 27 columns."""
    out = np.full((reps, 27), np.nan)
    for r in prange(reps):
        np.random.seed(seed + r)
        _main_design_one(n, code1, code0, rho, idrr, beta, I0, D0, gam, L, cap, out[r])
    return out

