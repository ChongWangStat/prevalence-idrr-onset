#!/usr/bin/env python3
"""Public randomized colon-cancer trial: independent longitudinal recurrence benchmark.

R survival::colon records 929 patients (2 records/patient: recurrence and death).
This script reconstructs a pseudo-cross-sectional prevalence survey at followup
age t, compares POR / mean-backward-time-adjusted POR / recent-onset OR with
an **independent calculation** of recurrence events / event-free person-time
from the longitudinal cohort within (t-w,t]. The numerical estimators are
not statistically independent: the SAME cohort contributes to both methods.

The cohort has one enrollment time, and later censored/deceased patients
leave observation; stationarity assumptions of the main paper DO NOT hold.
Competing death, time-changing incidence, and small recent-case counts mean
this is an informative stress test, not formal validation of stationarity,
confidence-interval calibration, or clinical treatment efficacy.

Download from https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv
when networked; no raw patient files are redistributed.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
URL='https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv'
BASE=Path(__file__).resolve().parents[1]
DAYS=(365,730,1095,1460,1825);WINDOWS=(90,180,365)


def validate(raw):
    must={'id','rx','time','status','etype'}
    if not must<=set(raw):raise ValueError('Input missing required fields '+str(must-set(raw)))
    if len(raw)!=1858 or raw.id.nunique()!=929:raise ValueError('Expected 929 patients, two rows per patient')
    if set(raw.rx)!=set(['Obs','Lev','Lev+5FU']):raise ValueError('Unexpected treatment arms')
    if set(raw.etype)!={1,2}:raise ValueError('Expected recurrence etype=1, death etype=2')
    if (raw.groupby('id').etype.nunique()!=2).any() or (raw.groupby('id').size()!=2).any():raise ValueError('Invalid per-patient structure')
    events=raw[(raw.etype==1)&(raw.status==1)]
    srv=raw[raw.etype==2].set_index('id');rec=events.set_index('id')
    if np.any(rec.time>srv.loc[rec.index,'time']):raise ValueError('Recurrence after death/censoring')
    return raw

def person_records(raw):
    rec=raw[raw.etype==1].set_index('id');srv=raw[raw.etype==2].set_index('id')
    r=rec[['rx','time','status']].rename(columns={'time':'rec_time','status':'rec_status'})
    r=r.join(srv[['time','status']].rename(columns={'time':'death_time','status':'death_status'}))
    return r.reset_index()

def compute(raw):
    recs=person_records(validate(raw));rows=[]
    for t in DAYS:
        for w in WINDOWS:
            z={'t':t,'w':w}
            for g in ['Obs','Lev+5FU']:
                d=recs[recs.rx==g]
                alive=d.death_time>=t
                preval=alive&(d.rec_status==1)&(d.rec_time<=t)
                healthy=alive&~preval&(d.rec_time>=t)
                ages=(t-d.loc[preval,'rec_time']).to_numpy(float)
                exit_time=np.minimum(d.rec_time.to_numpy(float),d.death_time.to_numpy(float))
                pt=np.maximum(0,np.minimum(t,exit_time)-max(0,t-w)).sum()
                events=((d.rec_status==1)&(d.rec_time>t-w)&(d.rec_time<=t)&(d.rec_time<=d.death_time)).sum()
                z[g+'H']=int(healthy.sum());z[g+'P']=int(preval.sum());z[g+'R']=int((ages<=w).sum());z[g+'A']=float(ages.mean()) if len(ages) else np.nan
                z[g+'PT']=float(pt);z[g+'E']=int(events)
            control='Obs';active='Lev+5FU'
            z['POR']=(z[active+'P']/z[active+'H'])/(z[control+'P']/z[control+'H'])
            z['aPOR']=z['POR']*z[control+'A']/z[active+'A']
            z['ROR']=(z[active+'R']/z[active+'H'])/(z[control+'R']/z[control+'H']) if z[control+'R'] else np.nan
            rate1=z[active+'E']/z[active+'PT'];rate0=z[control+'E']/z[control+'PT']
            z['IDRR']=rate1/rate0 if rate0 else np.nan
            rows.append(z)
    return pd.DataFrame(rows)

def compare(result,truth):
    key=['t','w'];x=result.merge(truth,on=key,suffixes=('_new','_verified'))
    if len(x)!=len(truth):raise ValueError('Missing scenarios')
    for k in truth:
        if k in key:continue
        if not np.allclose(x[k+'_new'],x[k+'_verified'],atol=1e-8,rtol=1e-11,equal_nan=True):
            raise ValueError('External aggregate verification mismatch: '+k)
    return 'All 15 independently computed records agree with independently verified public-source aggregates.'

def plots(df,dir):
    sample=df[df.w==90]
    fig,ax=plt.subplots(figsize=(7.6,4.0))
    ax.plot(sample.t,sample.IDRR,'o-',label='Observed recurrence events / at-risk time')
    ax.plot(sample.t,sample.ROR,'s--',label='Recent-onset OR, 90-day window')
    ax.plot(sample.t,sample.POR,'^:',label='Prevalence odds ratio')
    ax.axhline(1,c='0.55',ls=':',lw=.9)
    ax.set(xlabel='Days since randomization (single closed cohort)',ylabel='Ratio, Lev+5FU / observation',title='Public colon trial: descriptive nonstationary stress test')
    ax.grid(alpha=.14);ax.legend(fontsize=7,frameon=False);fig.tight_layout()
    fig.savefig(dir/'colon_public_stress_test.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--download',action='store_true');p.add_argument('--verified-aggregates',action='store_true');p.add_argument('--outdir',type=Path,default=BASE/'results/python');a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    if sum([bool(a.input),a.download,a.verified_aggregates])!=1:raise ValueError('Specify exactly one input mode')
    ref=BASE/'data/colon_verified_aggregates.csv'
    if a.verified_aggregates:
        df=pd.read_csv(ref);status='Plotting previously independently verified public aggregate results; raw data analysis requires networked --download.'
    else:
        raw=pd.read_csv(URL if a.download else a.input)
        df=compute(raw);status=compare(df,pd.read_csv(ref))
    df.to_csv(a.outdir/'colon_public_application.csv',index=False,float_format='%.15g')
    plots(df,a.outdir)
    print(status,flush=True)
    print(df[['t','w','POR','aPOR','ROR','IDRR']].to_string(index=False),flush=True)

if __name__=='__main__':main()
