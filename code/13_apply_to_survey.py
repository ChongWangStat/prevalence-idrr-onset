#!/usr/bin/env python3
"""Apply simultaneous early-exit sensitivity intervals to grouped onset counts.

Input CSV has two rows (exposed and unexposed), columns group, healthy,
onset_0_30,onset_30_60,onset_60_90 (or other contiguous onset bins).
A confidence interval is guaranteed only under independent sampling by group
and a credible envelope on differential early-exit hazards. No causality claim.
"""
import argparse
import csv
import importlib.util
import math
from pathlib import Path
from scipy.optimize import brentq

spec=importlib.util.spec_from_file_location('multiwindow',Path(__file__).with_name('11_simultaneous_windows.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def parse_table(path):
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        headers=reader.fieldnames
        rows={r['group'].strip().lower():r for r in reader}
    if set(rows)!={'exposed','unexposed'}:
        raise ValueError('Required group labels: exposed, unexposed')
    if 'healthy' not in headers:
        raise ValueError('Missing healthy column')
    bins=[]
    for name in headers:
        if not name.startswith('onset_'): continue
        try:lo,hi=[int(x) for x in name[6:].split('_')]
        except Exception as e:raise ValueError('Onset fields must be onset_LO_HI') from e
        if hi<=lo: raise ValueError('Onset intervals must increase')
        bins.append((lo,hi,name))
    bins.sort()
    if not bins or bins[0][0]!=0 or any(bins[i][1]!=bins[i+1][0] for i in range(len(bins)-1)):
        raise ValueError('Onset intervals must start at zero and be contiguous')
    counts={}
    for group,row in rows.items():
        vals={k:int(row[k]) for k in ['healthy']+[x[2] for x in bins]}
        if any(v<0 for v in vals.values()): raise ValueError('Counts must be nonnegative')
        counts[group]=vals
    return bins,counts

def calculate(bins,counts,delta0,delta1,alpha):
    K=len(bins)
    wlist=[];los=[];his=[]
    for lo,hi,col in bins:
        p1=sum(counts['exposed'][c] for l,h,c in bins if h<=hi)
        p0=sum(counts['unexposed'][c] for l,h,c in bins if h<=hi)
        h1=counts['exposed']['healthy'];h0=counts['unexposed']['healthy']
        # symmetric envelope factor applied after exact interval for ROR*(w)
        rawlo,rawhi=m.exact_bounds(p1,p0,h1,h0,hi,0,alpha/K)
        adjlo=rawlo*m.B(delta0*hi)
        adjhi=rawhi/m.B(delta1*hi)
        wlist.append({'window':hi,'exposed_recent':p1,'unexposed_recent':p0,
                     'raw_lower':rawlo,'raw_upper':rawhi,
                     'lower':adjlo,'upper':adjhi})
        los.append(adjlo);his.append(adjhi)
    lo=max(los);hi=min(his)
    return wlist,(lo,hi)

def tipping(bins,counts,alpha):
    def f(d):
        _,(lo,hi)=calculate(bins,counts,d,d,alpha)
        return max(lo-1,1-hi)
    if f(0)<=0: return 0.0
    if f(100)>0: return math.inf
    return brentq(f,0,100)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--delta0',type=float,required=True)
    p.add_argument('--delta1',type=float,required=True)
    p.add_argument('--alpha',type=float,default=0.05)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.delta0<0 or args.delta1<0:raise ValueError('Hazard difference bounds cannot be negative')
    bins,c=parse_table(args.input)
    windows,(lo,hi)=calculate(bins,c,args.delta0,args.delta1,args.alpha)
    res=[{**r,'simult_lower':lo,'simult_upper':hi,'delta0':args.delta0,'delta1':args.delta1,'alpha':args.alpha} for r in windows]
    dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
    with open(dst,'w',newline='',encoding='utf-8') as f:
        ww=csv.DictWriter(f,fieldnames=list(res[0]));ww.writeheader();ww.writerows(res)
    print(f'Simultaneous interval: [{lo:.6g}, {hi:.6g}]; alpha={args.alpha}; K={len(bins)}')
    print(f'Symmetric-envelope tipping delta for null IDRR=1: {tipping(bins,c,args.alpha):.7f} / day')
    print('WARNING: Delta is an externally justified sensitivity parameter, not an estimate.')
if __name__=='__main__':main()
