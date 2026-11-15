#!/usr/bin/env python3
"""Reproduce the worked simulated survey and sensitivity curve from saved CSV."""
import argparse
import csv
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('multiwindow',Path(__file__).with_name('11_simultaneous_windows.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result-dir',default='results/simultaneous_windows')
    ap.add_argument('--figure-dir',default='figures')
    args=ap.parse_args()
    p=Path(args.result_dir);figdir=Path(args.figure_dir);figdir.mkdir(parents=True,exist_ok=True)
    with open(p/'worked_example.csv',newline='',encoding='utf-8') as f:
        ex=[r for r in csv.DictReader(f) if float(r['n'])==5000 and float(r['idrr'])==2 and float(r['delta'])==0.005]
    assert len(ex)==3
    def get_bounds(delta):
        indiv=[]
        for r in ex:
            w=int(r['window'])
            lo,hi=m.exact_bounds(int(r['recent1']),int(r['recent0']),int(r['healthy1']),int(r['healthy0']),w,delta,0.05/3)
            indiv.append((lo,hi))
        return max(x[0] for x in indiv),min(x[1] for x in indiv)
    deltas=np.linspace(0,0.012,81)
    grid=[]
    for d in deltas:
        lo,hi=get_bounds(float(d))
        grid.append({'delta_per_day':float(d),'lower':lo,'upper':hi,'null_excluded':int(lo>1 or hi<1)})
    with open(p/'sensitivity_curve.csv','w',newline='',encoding='utf-8') as f:
        ww=csv.DictWriter(f,fieldnames=grid[0].keys());ww.writeheader();ww.writerows(grid)
    f=lambda d:get_bounds(d)[0]-1
    tip=0. if f(0)<=0 else brentq(f,0,1.)
    print('Example tipping delta per day',tip,'actual hazard difference',1/108-1/180)
    (p/'example_tipping.txt').write_text(
        'This is a simulated, not observed, survey.\n'
        f'Expected population size 5000 per group, IDRR 2, durations exponential 108 vs 180 days.\n'
        f'Bonferroni-adjusted 95% simultaneous envelope; null tipping delta = {tip:.8f}/day.\n'
        f'Actual differential exit hazard = {1/108-1/180:.8f}/day.\n'
        'A tipping delta is NOT a bound inferred from the prevalence survey.\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7.3,4.5))
    ax.plot(deltas,[r['lower'] for r in grid],label='Simultaneous lower limit')
    ax.plot(deltas,[r['upper'] for r in grid],label='Simultaneous upper limit')
    ax.axhline(1,linestyle='--',linewidth=.9,label='No incidence association')
    ax.axvline(1/108-1/180,linestyle=':',linewidth=1,label='Simulated hazard difference')
    ax.set(xlabel='Assumed maximum differential early-exit hazard (per day)',ylabel='IDRR confidence limits',title='Simulated cross-sectional survey (not observational data)')
    ax.set_ylim(0.5,4.5)
    ax.grid(alpha=.15)
    ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/'figS2_multiwindow_sensitivity.pdf')
    fig.savefig(figdir/'figS2_multiwindow_sensitivity.png',dpi=240)
    plt.close(fig)

if __name__=='__main__': main()
