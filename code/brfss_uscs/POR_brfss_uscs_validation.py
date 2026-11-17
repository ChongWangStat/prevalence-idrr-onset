"""Recent-onset odds ratio from BRFSS 2022 against the registry incidence rate ratio from USCS.

Follows BRFSS_USCS_analysis_plan.md, which was fixed before any sex-specific result was computed.

Inputs (public CDC files):
  LLCP2022.ASC   BRFSS 2022 combined landline and cell phone data, fixed width
                 (https://www.cdc.gov/brfss/annual_data/annual_2022.html; column positions from the 2022 variable layout)
  BYAGE.TXT, BYAREA.TXT   U.S. Cancer Statistics 1999-2023 ASCII files

Usage: python3 POR_brfss_uscs_validation.py <LLCP2022.ASC> <BYAGE.TXT> <BYAREA.TXT>      (needs numpy, scipy)
"""
import sys, csv, math, collections
import numpy as np
from scipy.special import gammaln

BRFSS, BYAGE, BYAREA = sys.argv[1:4]
COLS = {"_STATE": (1, 2), "DISPCODE": (32, 4), "CHCOCNC1": (124, 1), "CNCRDIFF": (310, 1), "CNCRAGE": (311, 2),
        "CNCRTYP2": (313, 2), "QSTVER": (665, 2), "_STSTR": (1410, 6), "_LLCPWT": (1751, 10), "_SEX": (1980, 1), "_AGE80": (1984, 2)}
MODULE_STATES = {"05", "08", "09", "10", "15", "16", "17", "18", "26", "29", "31", "32", "34", "35", "37", "41", "44", "46", "49", "50", "51", "55"}
NO_MODULE_CELLS = {("08", "20"), ("09", "20"), ("26", "20"), ("31", "20"), ("34", "20"), ("49", "20")}
STATE_NAMES = {"05": "Arkansas", "08": "Colorado", "09": "Connecticut", "10": "Delaware", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
               "18": "Indiana", "26": "Michigan", "29": "Missouri", "31": "Nebraska", "32": "Nevada", "34": "New Jersey", "35": "New Mexico",
               "37": "North Carolina", "41": "Oregon", "44": "Rhode Island", "46": "South Dakota", "49": "Utah", "50": "Vermont",
               "51": "Virginia", "55": "Wisconsin"}

# ---------------------------------------------------------------- survey records
recs = []
with open(BRFSS, "r", encoding="latin-1", newline="") as f:
    for line in f:
        st = line[0:2]
        if st not in MODULE_STATES:
            continue
        r = {k: line[s - 1:s - 1 + l].strip() for k, (s, l) in COLS.items()}
        recs.append(r)
print("records in the 22 module states:", len(recs))

n = len(recs)
w = np.array([float(r["_LLCPWT"]) for r in recs])
strata = np.array([r["_STSTR"] for r in recs])
age = np.array([int(r["_AGE80"]) for r in recs])
male = np.array([r["_SEX"] == "1" for r in recs]); female = np.array([r["_SEX"] == "2" for r in recs])
eligible = np.array([r["DISPCODE"] == "1100" and (r["_STATE"], r["QSTVER"]) not in NO_MODULE_CELLS for r in recs])
healthy = np.array([r["CHCOCNC1"] == "2" for r in recs])
case_raw = np.array([r["CHCOCNC1"] == "1" for r in recs])
dxage = np.array([int(r["CNCRAGE"]) if r["CNCRAGE"].isdigit() and 1 <= int(r["CNCRAGE"]) <= 97 else -1 for r in recs])
onetype = np.array([r["CNCRDIFF"] == "1" for r in recs])
ctype = np.array([int(r["CNCRTYP2"]) if r["CNCRTYP2"].isdigit() else -1 for r in recs])
skin_only = onetype & np.isin(ctype, (22, 23))
# Exclude skin-only histories consistently from the case count used in POR and ROR.
# This is a post-analysis correction to the initial denominator, not a prespecified change.
case = case_raw & ~skin_only
healthy = healthy | (case_raw & skin_only) # Eligible-cancer-free denominator includes skin-only histories.
d = np.where(dxage > 0, age - dxage, -99)
recent0 = case & (d == 0)                                       # primary
recentS1 = case & (d == 0) & onetype & ~np.isin(ctype, (22, 23, 77, 99, -1))  # one known type
recentS2 = case & np.isin(d, (0, 1))                             # under two years

_, sidx = np.unique(strata, return_inverse=True)
nstr = sidx.max() + 1
nh = np.bincount(sidx, minlength=nstr).astype(float)

def design_ratio(num, band, weighted=True):
    """log odds ratio men/women of num against healthy, within the age band, with linearisation SE."""
    dom = eligible & band
    ww = w if weighted else np.ones(n)
    RM = (ww * (num & male & dom)).sum(); RF = (ww * (num & female & dom)).sum()
    TM = (ww * (healthy & male & dom)).sum(); TF = (ww * (healthy & female & dom)).sum()
    theta = math.log(RM / TM) - math.log(RF / TF)
    z = ww * dom * ((num & male) / RM - (healthy & male) / TM - (num & female) / RF + (healthy & female) / TF)
    zbar = np.bincount(sidx, weights=z, minlength=nstr) / nh
    ss = np.bincount(sidx, weights=(z - zbar[sidx]) ** 2, minlength=nstr)
    var = float((np.where(nh > 1, nh / np.maximum(nh - 1, 1), 0.0) * ss).sum())
    return theta, math.sqrt(var)

def exact_or_ci(a, b, c, dd, alpha=0.05):
    n1, n2, m = a + b, c + dd, a + c
    lo, hi = max(0, m - n2), min(n1, m)
    xs = np.arange(lo, hi + 1)
    lw = (gammaln(n1 + 1) - gammaln(xs + 1) - gammaln(n1 - xs + 1) + gammaln(n2 + 1) - gammaln(m - xs + 1) - gammaln(n2 - m + xs + 1))
    def tails(lp):
        l = lw + xs * lp; l = l - l.max(); p = np.exp(l); p /= p.sum()
        return p[xs >= a].sum(), p[xs <= a].sum()
    def solve(fn):
        a_, b_ = -30.0, 30.0
        for _ in range(100):
            mid = (a_ + b_) / 2
            if fn(mid) > 0: b_ = mid
            else: a_ = mid
        return (a_ + b_) / 2
    return math.exp(solve(lambda lp: tails(lp)[0] - alpha / 2)), math.exp(solve(lambda lp: alpha / 2 - tails(lp)[1]))

# ---------------------------------------------------------------- registry
AGE_GROUPS = ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79"]
reg = collections.defaultdict(lambda: [0.0, 0.0])
with open(BYAGE, newline="", encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter="|"):
        if (row["EVENT_TYPE"] == "Incidence" and row["SITE"] == "All Cancer Sites Combined" and row["RACE"] == "All Races"
                and row["SEX"] in ("Male", "Female") and row["YEAR"] in ("2021", "2022", "2023") and row["AGE"] in AGE_GROUPS):
            reg[(row["YEAR"], row["SEX"], row["AGE"])] = [float(row["COUNT"]), float(row["POPULATION"])]

def registry_rr(year, groups):
    cm = sum(reg[(year, "Male", g)][0] for g in groups); pm = sum(reg[(year, "Male", g)][1] for g in groups)
    cf = sum(reg[(year, "Female", g)][0] for g in groups); pf = sum(reg[(year, "Female", g)][1] for g in groups)
    rr = (cm / pm) / (cf / pf); se = math.sqrt(1 / cm + 1 / cf)
    return rr, rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se), cm, cf, pm, pf

# ---------------------------------------------------------------- results
BANDS = [("Ages 20-79 (primary)", (age >= 20) & (age <= 79), AGE_GROUPS),
         ("Ages 20-49", (age >= 20) & (age <= 49), AGE_GROUPS[:6]),
         ("Ages 50-79", (age >= 50) & (age <= 79), AGE_GROUPS[6:])]
fmt = lambda th, se: f"{math.exp(th):.2f} ({math.exp(th - 1.96 * se):.2f}, {math.exp(th + 1.96 * se):.2f})"
for label, band, groups in BANDS:
    dom = eligible & band
    print(f"\n=== {label}: men against women")
    for yr in ("2022", "2021", "2023"):
        rr, lo, hi, cm, cf, pm, pf = registry_rr(yr, groups)
        print(f"  registry IDRR {yr}: {rr:.3f} ({lo:.3f}, {hi:.3f})   cases M/F {cm:.0f}/{cf:.0f}; rates per 100,000 {1e5*cm/pm:.0f}/{1e5*cf/pf:.0f}")
    rr22 = registry_rr("2022", groups)[0]
    cnt = lambda x: (int((x & male & dom).sum()), int((x & female & dom).sum()))
    hM, hF = cnt(healthy); pM, pF = cnt(case)
    print(f"  survey, unweighted: healthy M/F {hM}/{hF}; cases M/F {pM}/{pF}; cases with unknown age at diagnosis M/F "
          f"{int((case & (dxage < 0) & male & dom).sum())}/{int((case & (dxage < 0) & female & dom).sum())}")
    th, se = design_ratio(case, band)
    print(f"  POR  weighted {fmt(th, se)}")
    for name, rec in (("ROR primary (told since last birthday)", recent0), ("ROR S1 (one known type)", recentS1), ("ROR S2 (under two years)", recentS2)):
        rM, rF = cnt(rec)
        th, se = design_ratio(rec, band)
        inside = math.exp(th - 1.96 * se) <= rr22 <= math.exp(th + 1.96 * se)
        tu, _ = design_ratio(rec, band, weighted=False)
        L, U = exact_or_ci(rM, hM, rF, hF)
        print(f"  {name}: recent M/F {rM}/{rF}; weighted {fmt(th, se)} -> registry IDRR {'inside' if inside else 'OUTSIDE'};"
              f" unweighted {math.exp(tu):.2f} exact ({L:.2f}, {U:.2f})")

# ---------------------------------------------------------------- check on the choice of states (all ages, crude)
tot = collections.defaultdict(lambda: [0.0, 0.0]); us = collections.defaultdict(lambda: [0.0, 0.0])
with open(BYAREA, newline="", encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter="|"):
        if (row["EVENT_TYPE"] == "Incidence" and row["SITE"] == "All Cancer Sites Combined" and row["RACE"] == "All Races"
                and row["SEX"] in ("Male", "Female") and row["YEAR"] == "2022"):
            try:
                c, p_ = float(row["COUNT"]), float(row["POPULATION"])
            except ValueError:
                continue
            us[row["SEX"]][0] += c; us[row["SEX"]][1] += p_          # 50 states and the District of Columbia
            if row["AREA"] in STATE_NAMES.values():
                tot[row["SEX"]][0] += c; tot[row["SEX"]][1] += p_
rr_us = (us["Male"][0] / us["Male"][1]) / (us["Female"][0] / us["Female"][1])
rr_22 = (tot["Male"][0] / tot["Male"][1]) / (tot["Female"][0] / tot["Female"][1])
print(f"\nall-ages crude rate ratio M/F, 2022: all states {rr_us:.3f}; the 22 module states {rr_22:.3f}")

# ================================================================ NOT in the plan: added after the planned results were seen
# Age-stratified comparison on a common footing. Within five-year age groups the survey gives a Mantel-Haenszel odds
# ratio (unweighted; Robins-Breslow-Greenland interval). The registry figure it should be compared with is the ratio
# of the age-specific rates averaged with the same Mantel-Haenszel weights, healthy_M * healthy_F / n in each age group.
print("\n=== EXPLORATORY (not in the plan): age-stratified Mantel-Haenszel, survey unweighted, registry averaged with the survey's MH weights")
def mh(rec, groups):
    num = den = 0.0; sPR = sQS = sPSQR = 0.0; wr_m = wr_f = 0.0; cases = [0, 0]
    for g in groups:
        lo, hi = (int(x) for x in g.split("-"))
        dom = eligible & (age >= lo) & (age <= hi)
        a = float((rec & male & dom).sum()); b = float((healthy & male & dom).sum())
        c = float((rec & female & dom).sum()); dd = float((healthy & female & dom).sum())
        nn = a + b + c + dd
        R, S = a * dd / nn, b * c / nn; P, Q = (a + dd) / nn, (b + c) / nn
        num += R; den += S; sPR += P * R; sQS += Q * S; sPSQR += P * S + Q * R
        om = b * dd / nn
        wr_m += om * reg[("2022", "Male", g)][0] / reg[("2022", "Male", g)][1]
        wr_f += om * reg[("2022", "Female", g)][0] / reg[("2022", "Female", g)][1]
        cases[0] += int(a); cases[1] += int(c)
    orr = num / den
    se = math.sqrt(sPR / (2 * num ** 2) + sPSQR / (2 * num * den) + sQS / (2 * den ** 2))
    return orr, orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se), wr_m / wr_f, cases
for label, band, groups in BANDS:
    for name, rec in (("POR", case), ("ROR, told since last birthday", recent0), ("ROR, under two years", recentS2)):
        orr, lo, hi, target, cs = mh(rec, groups)
        print(f"  {label:22s} {name:30s} cases M/F {cs[0]}/{cs[1]}: MH {orr:.2f} ({lo:.2f}, {hi:.2f}); registry with the same weights {target:.2f}"
              f" -> {'inside' if lo <= target <= hi else 'OUTSIDE'}")
