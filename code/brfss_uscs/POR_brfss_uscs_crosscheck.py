"""Second implementation of the BRFSS 2022 estimates, for checking POR_brfss_uscs_validation.py and POR_brfss_uscs_checks.py.

It shares no code with those scripts and differs from them in two ways:
  - the survey file is read as a fixed-length byte table (every record must be 2052 characters plus CR LF),
    not line by line;
  - standard errors come from the stratified delete-one jackknife (strata _STSTR, each respondent a sampling unit),
    not from linearisation.
Point estimates must agree exactly with the other scripts; intervals should agree to about two decimals.

Usage: python3 POR_brfss_uscs_crosscheck.py <LLCP2022.ASC>      (needs numpy)
"""
import sys, os, math
import numpy as np

path = sys.argv[1]
REC = 2054
size = os.path.getsize(path)
assert size % REC == 0, "file length is not a multiple of 2054 bytes"
tab = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, REC)
assert (tab[:, 2052] == 13).all() and (tab[:, 2053] == 10).all(), "a record does not end with CR LF at byte 2053"
print("records in the file:", tab.shape[0])

def field(start, length, rows):
    """Text of a field (1-based start) for the given rows, as a bytes array."""
    return np.ascontiguousarray(tab[rows, start - 1:start - 1 + length]).view(f"S{length}").ravel()

state_all = field(1, 2, slice(None))
MODULE = [b"05", b"08", b"09", b"10", b"15", b"16", b"17", b"18", b"26", b"29", b"31", b"32", b"34", b"35", b"37", b"41",
          b"44", b"46", b"49", b"50", b"51", b"55"]
rows = np.flatnonzero(np.isin(state_all, MODULE))
print("records in the 22 states:", rows.size)
state = state_all[rows]
qstver = field(665, 2, rows)
disp = field(32, 4, rows)
wt = np.char.strip(field(1751, 10, rows)).astype(float)
stratum = field(1410, 6, rows)
sex = field(1980, 1, rows)
age = field(1984, 2, rows).astype(int)
other_cancer = field(124, 1, rows)
ntypes = field(310, 1, rows)
dx = np.char.strip(field(311, 2, rows)); kind = np.char.strip(field(313, 2, rows))
dx_age = np.array([int(x) if x.isdigit() else 0 for x in dx]); dx_age[(dx_age < 1) | (dx_age > 97)] = 0
kind_code = np.array([int(x) if x.isdigit() else -1 for x in kind])

no_module = (qstver == b"20") & np.isin(state, [b"08", b"09", b"26", b"31", b"34", b"49"])
kept = (disp == b"1100") & ~no_module
man, woman = sex == b"1", sex == b"2"
never, ever_raw = other_cancer == b"2", other_cancer == b"1"
skin = (ntypes == b"1") & ((kind_code == 22) | (kind_code == 23))
# Post-analysis correction: exclude skin-only history consistently from POR case count.
ever = ever_raw & ~skin
never = never | (ever_raw & skin) # No eligible cancer, so include in comparison denominator.
years = np.where(dx_age > 0, age - dx_age, -1)
recent1 = ever & (years == 0) & ~skin
recent2 = ever & ((years == 0) | (years == 1)) & ~skin

_, h = np.unique(stratum, return_inverse=True)
H = h.max() + 1
nh = np.bincount(h, minlength=H).astype(float)

def jackknife(numer, denom, lo, hi):
    """log{(numer_M/denom_M)/(numer_F/denom_F)} for ages lo..hi with a delete-one jackknife standard error."""
    dom = kept & (age >= lo) & (age <= hi)
    ys = [wt * (numer & man & dom), wt * (denom & man & dom), wt * (numer & woman & dom), wt * (denom & woman & dom)]
    tot = [y.sum() for y in ys]
    est = math.log(tot[0] / tot[1]) - math.log(tot[2] / tot[3])
    ok = nh[h] > 1
    rep = []
    for y, t in zip(ys, tot):
        yh = np.bincount(h, weights=y, minlength=H)
        rep.append(t + (yh[h] - nh[h] * y) / np.maximum(nh[h] - 1, 1))
    th = np.log(rep[0] / rep[1]) - np.log(rep[2] / rep[3])
    var = float((((nh[h] - 1) / nh[h]) * (th - est) ** 2)[ok].sum())
    return est, math.sqrt(var)

show = lambda e, s: f"{math.exp(e):.2f} ({math.exp(e - 1.96 * s):.2f}, {math.exp(e + 1.96 * s):.2f})   log estimate {e:+.5f}, SE {s:.4f}"
for lo, hi in ((20, 79), (20, 49), (50, 79)):
    dom = kept & (age >= lo) & (age <= hi)
    print(f"\nages {lo}-{hi}: never diagnosed M/F {int((never & man & dom).sum())}/{int((never & woman & dom).sum())};"
          f" recent (difference 0) M/F {int((recent1 & man & dom).sum())}/{int((recent1 & woman & dom).sum())};"
          f" recent (difference 0 or 1) M/F {int((recent2 & man & dom).sum())}/{int((recent2 & woman & dom).sum())}")
    print("  POR                            ", show(*jackknife(ever, never, lo, hi)))
    print("  ROR, told since last birthday  ", show(*jackknife(recent1, never, lo, hi)))
    print("  ROR, under two years           ", show(*jackknife(recent2, never, lo, hi)))
    print("  ROR/POR, told since last birthday", show(*jackknife(recent1, ever, lo, hi)))
    print("  ROR/POR, under two years       ", show(*jackknife(recent2, ever, lo, hi)))
