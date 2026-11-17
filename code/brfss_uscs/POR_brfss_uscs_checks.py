"""Three checks on the BRFSS 2022 - USCS comparison (run after POR_brfss_uscs_validation.py; same inputs and definitions).

  1. Estimand: the registry benchmark recomputed for the quantity the survey-weighted odds ratio estimates
     (registry age-specific rates applied to the survey's own never-diagnosed population, by sex), with the
     effect of second and later primaries bounded.
  2. Design: sampling units, strata, and whether keeping completed interviews only changes the sex contrast.
  3. Window: counts by (current age - age at diagnosis) against what completed-year ages imply, and a
     design-based interval for ROR/POR, which takes the covariance of the two estimators into account.

These checks were specified after the planned results had been seen (16 November 2026). They change no planned
estimate; they change the benchmark the estimates are compared with and add one interval.

Usage: python3 POR_brfss_uscs_checks.py <LLCP2022.ASC> <BYAGE.TXT>      (needs numpy)
"""
import sys, csv, math, collections
import numpy as np

BRFSS, BYAGE = sys.argv[1:3]
COLS = {"_STATE": (1, 2), "DISPCODE": (32, 4), "_PSU": (36, 10), "CHCOCNC1": (124, 1), "CNCRDIFF": (310, 1), "CNCRAGE": (311, 2),
        "CNCRTYP2": (313, 2), "QSTVER": (665, 2), "_STSTR": (1410, 6), "_LLCPWT": (1751, 10), "_SEX": (1980, 1), "_AGE80": (1984, 2)}
MODULE_STATES = {"05", "08", "09", "10", "15", "16", "17", "18", "26", "29", "31", "32", "34", "35", "37", "41", "44", "46", "49", "50", "51", "55"}
NO_MODULE_CELLS = {("08", "20"), ("09", "20"), ("26", "20"), ("31", "20"), ("34", "20"), ("49", "20")}
AGE_GROUPS = ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79"]

recs = []
with open(BRFSS, "r", encoding="latin-1", newline="") as f:
    for line in f:
        if line[0:2] in MODULE_STATES:
            recs.append({k: line[s - 1:s - 1 + l].strip() for k, (s, l) in COLS.items()})
n = len(recs)
w = np.array([float(r["_LLCPWT"]) for r in recs])
strata = np.array([r["_STSTR"] for r in recs])
age = np.array([int(r["_AGE80"]) for r in recs])
male = np.array([r["_SEX"] == "1" for r in recs]); female = np.array([r["_SEX"] == "2" for r in recs])
complete = np.array([r["DISPCODE"] == "1100" for r in recs])
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
recent0 = case & (d == 0)
recentS2 = case & np.isin(d, (0, 1))
_, sidx = np.unique(strata, return_inverse=True)
nstr = sidx.max() + 1
nh = np.bincount(sidx, minlength=nstr).astype(float)
agegrp = np.clip((age - 20) // 5, -1, 12)          # 0..11 for 20-24 .. 75-79

def design_ratio(num, den, band):
    """log{(num_M/den_M)/(num_F/den_F)} among eligible records of the age band, weighted, with linearisation SE
    (strata _STSTR, each respondent a sampling unit, all records of a stratum kept)."""
    dom = eligible & band
    RM = (w * (num & male & dom)).sum(); RF = (w * (num & female & dom)).sum()
    TM = (w * (den & male & dom)).sum(); TF = (w * (den & female & dom)).sum()
    theta = math.log(RM / TM) - math.log(RF / TF)
    z = w * dom * ((num & male) / RM - (den & male) / TM - (num & female) / RF + (den & female) / TF)
    zbar = np.bincount(sidx, weights=z, minlength=nstr) / nh
    ss = np.bincount(sidx, weights=(z - zbar[sidx]) ** 2, minlength=nstr)
    return theta, math.sqrt(float((np.where(nh > 1, nh / np.maximum(nh - 1, 1), 0.0) * ss).sum()))

reg = {}
with open(BYAGE, newline="", encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter="|"):
        if (row["EVENT_TYPE"] == "Incidence" and row["SITE"] == "All Cancer Sites Combined" and row["RACE"] == "All Races"
                and row["SEX"] in ("Male", "Female") and row["YEAR"] in ("2021", "2022", "2023") and row["AGE"] in AGE_GROUPS):
            reg[(row["YEAR"], row["SEX"], row["AGE"])] = (float(row["COUNT"]), float(row["POPULATION"]))

BANDS = [("20-79", range(0, 12)), ("20-49", range(0, 6)), ("50-79", range(6, 12))]
ci = lambda th, se: f"{math.exp(th):.2f} ({math.exp(th - 1.96 * se):.2f}, {math.exp(th + 1.96 * se):.2f})"

# ------------------------------------------------------------------ 1. estimand
print("=== 1. Registry benchmark for the quantity the survey-weighted odds ratio estimates")
print("    H = weighted never-diagnosed, p = weighted share ever diagnosed, r = registry rate (all primaries / whole population).")
print("    First-cancer rate per never-diagnosed person: r / (1 + p (SIR - 1)), SIR = rate of a new primary in people already")
print("    diagnosed relative to people never diagnosed. Benchmark = [sum_a H_Ma I_Ma / sum_a H_Ma] / [same for women].")
H = np.zeros((2, 12)); P = np.zeros((2, 12))
for s, sexmask in enumerate((male, female)):
    for a in range(12):
        m = eligible & sexmask & (agegrp == a)
        H[s, a] = (w * (healthy & m)).sum(); P[s, a] = (w * (case & m)).sum()
p = P / (H + P)
print("    share ever diagnosed by age group (%), men:  ", " ".join(f"{100 * x:4.1f}" for x in p[0]))
print("    share ever diagnosed by age group (%), women:", " ".join(f"{100 * x:4.1f}" for x in p[1]))
def benchmark(year, groups, sir_m=1.0, sir_f=1.0):
    out = []
    for s, (sex, sir) in enumerate((("Male", sir_m), ("Female", sir_f))):
        num = sum(H[s, a] * (reg[(year, sex, AGE_GROUPS[a])][0] / reg[(year, sex, AGE_GROUPS[a])][1]) / (1 + p[s, a] * (sir - 1)) for a in groups)
        out.append(num / sum(H[s, a] for a in groups))
    return out[0] / out[1]
def planned(year, groups):
    c = [sum(reg[(year, sex, AGE_GROUPS[a])][0] for a in groups) / sum(reg[(year, sex, AGE_GROUPS[a])][1] for a in groups) for sex in ("Male", "Female")]
    return c[0] / c[1]
bench = {}
for label, groups in BANDS:
    b = benchmark("2022", groups); bench[label] = b
    print(f"    ages {label}: planned benchmark (national counts / population) {planned('2022', groups):.3f};"
          f" aligned to the survey's never-diagnosed age structure {b:.3f} (2021: {benchmark('2021', groups):.3f}, 2023: {benchmark('2023', groups):.3f});"
          f" SIR 1.1 in both sexes {benchmark('2022', groups, 1.1, 1.1):.3f};"
          f" extremes SIR 1.5 men / 1.0 women {benchmark('2022', groups, 1.5, 1.0):.3f}, 1.0 men / 1.5 women {benchmark('2022', groups, 1.0, 1.5):.3f}")

print("\n    planned estimates against the aligned benchmark (survey-weighted, 95% interval from stratified linearisation)")
for label, groups in BANDS:
    band = np.isin(agegrp, list(groups))
    for name, rec in (("POR", case), ("ROR, told since last birthday", recent0), ("ROR, under two years", recentS2)):
        th, se = design_ratio(rec, healthy, band)
        lo, hi = math.exp(th - 1.96 * se), math.exp(th + 1.96 * se)
        print(f"    ages {label}  {name:30s} {ci(th, se)}   aligned benchmark {bench[label]:.2f} -> {'inside' if lo <= bench[label] <= hi else 'OUTSIDE'};"
              f" estimate / benchmark {math.exp(th) / bench[label]:.2f}")

# ------------------------------------------------------------------ 2. design
print("\n=== 2. Design")
keys = collections.Counter((r["_STATE"], r["_PSU"]) for r in recs)
print(f"    records {n}; strata (_STSTR) {nstr}; strata with one record {int((nh == 1).sum())};"
      f" (_STATE, _PSU) pairs occurring more than once: {sum(1 for v in keys.values() if v > 1)}")
band = (age >= 20) & (age <= 79)
print("    weighted share of interviews that are completed and in a questionnaire version with the module, ages 20-79:")
c = {}
for gname, g in (("never diagnosed", healthy), ("ever diagnosed", case)):
    for sname, sm in (("men", male), ("women", female)):
        m = band & g & sm
        c[(gname, sname)] = (w * (m & eligible)).sum() / (w * m).sum()
        print(f"      {gname:16s} {sname:6s} {100 * c[(gname, sname)]:.1f}%   (records {int(m.sum())}, kept {int((m & eligible).sum())})")
k = (c[("ever diagnosed", "men")] / c[("never diagnosed", "men")]) / (c[("ever diagnosed", "women")] / c[("never diagnosed", "women")])
print(f"    factor by which keeping completed interviews only multiplies the sex odds ratio of ever-diagnosed against never-diagnosed: {k:.3f}")
th_all, _ = design_ratio(case, healthy, band)
RM = (w * (case & male & band)).sum(); RF = (w * (case & female & band)).sum(); TM = (w * (healthy & male & band)).sum(); TF = (w * (healthy & female & band)).sum()
print(f"    POR ages 20-79: completed interviews {math.exp(th_all):.3f}; all interviews with an answer to the core cancer question {(RM / TM) / (RF / TF):.3f}")

# ------------------------------------------------------------------ 3. window
print("\n=== 3. Window implied by completed-year ages")
print("    A case told a years ago has (current age - age at diagnosis) = 0 with probability 1 - a (a < 1), and = 0 or 1 with")
print("    probability 1 (a < 1) or 2 - a (1 <= a < 2). Without loss of cases the expected counts at differences 0, 1, 2 are 1 : 2 : 2.")
dom = eligible & band & case
for sname, sm in (("men", male), ("women", female), ("both", male | female)):
    cnt = [int((dom & sm & (d == j)).sum()) for j in range(6)]
    wt = [(w * (dom & sm & (d == j))).sum() for j in range(6)]
    print(f"    {sname:6s} unweighted counts at differences 0..5: {cnt};  0 : 1 : 2 = 1 : {cnt[1] / cnt[0]:.2f} : {cnt[2] / cnt[0]:.2f}"
          f"   weighted 1 : {wt[1] / wt[0]:.2f} : {wt[2] / wt[0]:.2f}")
print("    mean time since diagnosis under the two weights (no loss): 1/3 year (difference 0) and 7/9 year (difference 0 or 1)")

print("\n    ROR / POR = (recent_M / cases_M) / (recent_F / cases_F): ratio of the recent fractions; the never-diagnosed cancel,")
print("    so the interval below accounts for the dependence of the two estimators (survey-weighted, stratified linearisation)")
for label, groups in BANDS:
    b = np.isin(agegrp, list(groups))
    for name, rec in (("told since last birthday", recent0), ("under two years", recentS2)):
        th, se = design_ratio(rec, case, b)
        print(f"    ages {label}  {name:26s} ROR/POR {ci(th, se)}   z = {th / se:.2f}")
