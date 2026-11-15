#!/usr/bin/env python3
"""Seeded, exact-model illustration of simultaneous IDRR sensitivity intervals.

The paper's stationary survey law gives independent Poisson H_j and counts of
prevalent cases in disjoint onset-time bins. We simulate those counts directly;
we do not reuse individual-level historical R random streams.

For each prespecified window w, an exact (conservative) 2-by-2 odds-ratio
interval obtained by binomial Clopper-Pearson bounds is transformed using Proposition 7(b)'s sharp early-exit envelope.
Bonferroni adjustment ensures at least 95% simultaneous coverage across all
three windows. Intersecting the intervals remains valid after data-based window
selection, if the envelope holds over the longest prespecified window.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats.contingency import odds_ratio
from scipy.stats import beta

WINDOWS = (30, 60, 90)
ALPHA = 0.05
DELTA = 0.005
BASE_SEED = 20261111
D0 = 180.0
D1_RATIO = 0.6
I0 = 0.001


def B(x):
    if x == 0:
        return 1.0
    return -math.expm1(-x)/x


def rest_mean_exp(w, duration):
    return -duration*math.expm1(-w/duration)


def parameters(total_size, idrr):
    i1 = I0*idrr
    d1 = D0*D1_RATIO
    h0 = total_size/(1+I0*D0)
    h1 = total_size/(1+i1*d1)
    return np.array([h1,h0]), np.array([i1,I0]), np.array([d1,D0])


def simulate_survey(rng, total_size, idrr):
    h,i,d=parameters(total_size,idrr)
    healthy=rng.poisson(h)
    cumul=np.array([[hj*ij*rest_mean_exp(w,dj) for w in WINDOWS]
                  for hj,ij,dj in zip(h,i,d)])
    increments=np.column_stack([cumul[:,0],np.diff(cumul,axis=1)])
    # independent Poisson onset-bin counts for exposed and unexposed
    bins=rng.poisson(increments)
    return healthy,np.cumsum(bins,axis=1)


def clopper_pearson(p,h,alpha):
    n=p+h
    if n==0: return 0.,1.
    lower=0. if p==0 else float(beta.ppf(alpha/2,p,h+1))
    upper=1. if h==0 else float(beta.ppf(1-alpha/2,p+1,h))
    return lower,upper


def cp_exact_bounds(p1,p0,h1,h0,w,delta,alpha):
    # Conditional on each group's Poisson total, its recent-case count is
    # binomial. CP intervals for the two group proportions at level
    # 1-alpha/2 each give a (conservative) exact confidence interval for OR.
    a1,b1=clopper_pearson(int(p1),int(h1),alpha/2)
    a0,b0=clopper_pearson(int(p0),int(h0),alpha/2)
    odds=lambda x: x/(1-x) if x<1 else float('inf')
    lo=0. if a1==0 or b0==1 else odds(a1)/odds(b0)
    hi=float('inf') if b1==1 or a0==0 else odds(b1)/odds(a0)
    return lo*B(delta*w),hi/B(delta*w)


def exact_bounds(p1,p0,h1,h0,w,delta,alpha):
    return cp_exact_bounds(p1,p0,h1,h0,w,delta,alpha)


def fisher_exact_bounds(p1,p0,h1,h0,w,delta,alpha):
    # Noncentral-hypergeometric inversion is finite-sample valid, including zeros.
    table=[[int(p1),int(h1)],[int(p0),int(h0)]]
    if sum(sum(r) for r in table)==0:
        orlo,orhi=0.0,float('inf')
    else:
        ci=odds_ratio(table,kind='conditional').confidence_interval(
            confidence_level=1-alpha)
        orlo,orhi=ci.low,ci.high
    return orlo*B(delta*w),orhi/B(delta*w)


def interval_length(lo,hi):
    if lo>0 and math.isfinite(hi) and hi>=lo:
        return math.log(hi/lo)
    return math.inf


def trial(healthy,counts,idrr,delta):
    h1,h0=healthy
    individual=[]
    adjusted=[]
    for k,w in enumerate(WINDOWS):
        p1,p0=counts[:,k]
        individual.append(exact_bounds(p1,p0,h1,h0,w,delta,ALPHA))
        adjusted.append(exact_bounds(p1,p0,h1,h0,w,delta,ALPHA/len(WINDOWS)))
    # Data-dependent choice is permitted only from the multiplicity-adjusted CIs
    choice=min(range(len(WINDOWS)),key=lambda k:interval_length(*adjusted[k]))
    lo=max(x[0] for x in adjusted)
    hi=min(x[1] for x in adjusted)
    return {"fixed60":individual[1],
            "best_bonf":adjusted[choice],
            "intersection":(lo,hi),
            "window":WINDOWS[choice],
            "per_windows":adjusted}


def stats(records,idrr):
    out=[]
    for method in ('fixed60','best_bonf','intersection'):
        iv=[r[method] for r in records]
        cover=np.mean([lo<=idrr<=hi for lo,hi in iv])
        invalid=np.mean([lo>hi for lo,hi in iv])
        finite=np.mean([lo>0 and math.isfinite(hi) and lo<=hi for lo,hi in iv])
        median=np.median([interval_length(lo,hi) for lo,hi in iv if math.isfinite(interval_length(lo,hi))])
        reject=np.mean([lo>1 or hi<1 for lo,hi in iv])
        out.append({"method":method,"coverage":cover,"empty":invalid,"finite":finite,
                    "median_log_width":median,"exclude_one":reject})
    return out


def write_csv(path,rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def run_condition(args):
    n,idrr,delta,reps=args
    key=f'{BASE_SEED}:{n}:{idrr}:{delta}'.encode()
    seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    records=[];ex=[]
    for r in range(reps):
        h,c=simulate_survey(rng,n,idrr)
        t=trial(h,c,idrr,delta)
        records.append(t)
        if r==0:
            for w,(lo,hi) in zip(WINDOWS,t['per_windows']):
                k=WINDOWS.index(w)
                ex.append({"n":n,"idrr":idrr,"delta":delta,"seed":seed,
                           "window":w,"healthy1":int(h[0]),"healthy0":int(h[1]),
                           "recent1":int(c[0,k]),"recent0":int(c[1,k]),
                           "simult_lower":lo,"simult_upper":hi})
    rows=[]
    for s in stats(records,idrr):
        rows.append({"n":n,"idrr":idrr,"delta":delta,"seed":seed,
                     "reps":reps,**s})
    print(f'done n={n} idrr={idrr} delta={delta}',flush=True)
    return rows,ex


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=800)
    ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--output-dir',default='results/simultaneous_windows')
    args=ap.parse_args()
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    rows=[];ex=[]
    conditions=[(1000,1.0,DELTA),(1000,2.0,DELTA),
                (5000,1.0,DELTA),(5000,2.0,DELTA),
                (50000,2.0,0.001)]
    tasks=[(*c,args.reps) for c in conditions]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rr,ee in pool.map(run_condition,tasks):
            rows.extend(rr);ex.extend(ee)
    write_csv(outdir/'summary.csv',rows)
    write_csv(outdir/'worked_example.csv',ex)
    (outdir/'analysis_metadata.txt').write_text(
      f'Stationary exact independent Poisson model; replicate seed per setting SHA256\n'
      f'Base seed: {BASE_SEED}\nReplicates per setting: {args.reps}\n'
      f'Windows in days: {WINDOWS}\nDelta units: per day\n'
      f'All intervals use conservative exact Bonferroni-Clopper-Pearson inversion.\n'
      f'Single fixed w=60 uses alpha=.05. Bonferroni and intersection use alpha=.05/3 per w.\n'
      f'In the delta=.001 setting with D1=108 and D0=180 days the envelope fails,\n'
      f'because the differential hazard is {1/108-1/180:.8f} per day.\n',encoding='utf-8')
    print('wrote',outdir,flush=True)
if __name__=='__main__':
    main()
