#!/usr/bin/env python3
"""Paired patient-level bootstrap for the public survival::colon application.

The two longitudinal event records for each patient remain paired; patients
are resampled with replacement *within the originally randomized arms*.
Exactly the same resampled patients contribute to the reconstructed prevalence
and event/person-time benchmark at every (t,w). All ratios are descriptive.

Raw source: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/colon.csv
GitHub source blob SHA: aa9f8ebcc4997da14d08a60fed0b32ec2996a9d8

Usage:
 python code/python_colon_bootstrap.py --input data/colon.csv --reps 4000 --seed 20261112
 python code/python_colon_bootstrap.py --download --reps 4000 --seed 20261112

NOTE: for a local private verification only, --internal-two-arm takes two-arm,
minimal derived patient records; it is not used by the public reproducibility
workflow or distributed with its raw records.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PUBLIC_URL='https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/colon.csv'
EXPECTED_GITHUB_BLOB='aa9f8ebcc4997da14d08a60fed0b32ec2996a9d8'
DAYS=(365,730,1095,1460,1825)
WINDOWS=(90,180,365)
GROUPS=('Obs','Lev+5FU')
MEASURES=('POR','aPOR','ROR','IDRR')
FEATURES=('H','P','R','age_sum','E','PT','retained','admin_censored','death_observed')


def load_records(input_path=None,download=False,internal_two_arm=None):
    if internal_two_arm is not None:
        d=pd.read_csv(internal_two_arm)
        assert set(['rx','rec_time','rec_status','death_time','death_status']).issubset(d.columns)
        assert len(d)==619 and d.rx.value_counts().to_dict()=={'Obs':315,'Lev+5FU':304}
        return d.copy(),{'source':'Public GitHub colon.csv, 619 event-history records reconstructed for an internal raw-source audit; no patient records distributed','github_blob_sha1':EXPECTED_GITHUB_BLOB,'compact_two_arm_fnv1a32':'2585649813','full_raw_validated':False}
    if not download and input_path is None:
        raise ValueError('Use --input path/to/colon.csv or --download (public source)')
    if input_path is not None:
        rawbytes=Path(input_path).read_bytes();raw=pd.read_csv(input_path)
    else:
        import urllib.request,io
        rawbytes=urllib.request.urlopen(PUBLIC_URL,timeout=40).read()
        raw=pd.read_csv(io.BytesIO(rawbytes))
    must={'id','rx','time','status','etype'}
    assert must.issubset(raw.columns),f'Missing {must-set(raw)}'
    assert len(raw)==1858 and raw.id.nunique()==929
    assert (raw.groupby('id').size()==2).all()
    assert (raw.groupby('id').etype.nunique()==2).all()
    rec=raw[raw.etype==1].set_index('id')
    death=raw[raw.etype==2].set_index('id')
    d=rec[['rx','time','status']].rename(columns={'time':'rec_time','status':'rec_status'}).join(death[['time','status']].rename(columns={'time':'death_time','status':'death_status'})).reset_index()
    assert len(d)==929 and set(d.rx)=={'Obs','Lev','Lev+5FU'}
    sha1=hashlib.sha1(b'blob '+str(len(rawbytes)).encode()+b'\0'+rawbytes).hexdigest()
    assert sha1==EXPECTED_GITHUB_BLOB, f'Upstream GitHub source changed: {sha1} != {EXPECTED_GITHUB_BLOB}'
    return d,{'source':PUBLIC_URL,'github_blob_sha1':sha1,'full_raw_validated':True,'rows':len(raw),'patients':len(d)}


def contributions(d,t,w):
    """Per-person contribution vector, preserving right censoring and death."""
    rec=d.rec_time.to_numpy(dtype=float); dth=d.death_time.to_numpy(dtype=float)
    observed=dth>=t
    had=(d.rec_status.to_numpy(dtype=int)==1)
    rel=(had & (rec<=t))
    prev=observed & rel
    healthy=observed & (~prev) & (rec>=t)
    recent=prev & ((t-rec)<=w)
    # First-recurrence risk ends at recurrence, death or right censoring.
    pt=np.maximum(0,np.minimum(np.minimum(rec,dth),t)-max(0,t-w))
    event=had & (rec>t-w) & (rec<=t) & (rec<=dth)
    # Censoring+death that prevents being included in the survey at day t.
    admin_censored=(dth<t) & (d.death_status.to_numpy(dtype=int)==0)
    death_observed=(dth<t) & (d.death_status.to_numpy(dtype=int)==1)
    return np.column_stack([healthy.astype(float),prev.astype(float),recent.astype(float),
      prev*(t-rec),event.astype(float),pt,observed.astype(float),
      admin_censored.astype(float),death_observed.astype(float)])


def measures(x0,x1):
    # Arrays shape (..., n_scenarios, n_features)
    p0=x0[...,1];p1=x1[...,1];h0=x0[...,0];h1=x1[...,0]
    r0=x0[...,2];r1=x1[...,2];a0=x0[...,3];a1=x1[...,3]
    e0=x0[...,4];e1=x1[...,4];pt0=x0[...,5];pt1=x1[...,5]
    with np.errstate(divide='ignore',invalid='ignore'):
        por=(p1/h1)/(p0/h0)
        apor=por*(a0/p0)/(a1/p1)
        ror=(r1/h1)/(r0/h0)
        direct=(e1/pt1)/(e0/pt0)
        discrepancy=ror/direct
    return np.stack([por,apor,ror,direct,discrepancy],axis=-1)


def percentile_interval(x,alpha=.05,min_valid=.95):
    good=np.isfinite(x)&(x>0)
    fraction=float(good.mean());k=int(good.sum())
    if fraction<min_valid or k<100:return np.nan,np.nan,fraction
    lo,hi=np.quantile(x[good],[alpha/2,1-alpha/2])
    return float(lo),float(hi),fraction


def bootstrap(d,reps=4000,seed=20261112):
    assert reps>=200
    scenario=[(t,w) for t in DAYS for w in WINDOWS]
    arms={g:d[d.rx==g].reset_index(drop=True) for g in GROUPS}
    if any(len(z)==0 for z in arms.values()):raise ValueError('Missing expected comparison arm')
    pre={g:np.array([contributions(z,t,w) for t,w in scenario]).transpose(1,0,2) for g,z in arms.items()}
    # No person is sampled across treatment arms. Entire record is sampled.
    rng=np.random.default_rng(seed)
    obs,tx=GROUPS
    counts={g:rng.multinomial(len(arms[g]),[1/len(arms[g])]*len(arms[g]),size=reps) for g in GROUPS}
    agg={g:counts[g]@pre[g].reshape(len(arms[g]),-1) for g in GROUPS}
    sums={g:agg[g].reshape(reps,len(scenario),len(FEATURES)) for g in GROUPS}
    points={g:pre[g].sum(axis=0) for g in GROUPS}
    bs=measures(sums[obs],sums[tx]);pv=measures(points[obs],points[tx])
    out=[]
    for k,(t,w) in enumerate(scenario):
        row={'t':t,'w':w,'reps':reps,'seed':seed,'n_obs':len(arms[obs]),'n_tx':len(arms[tx])}
        for g,short in [(obs,'Obs'),(tx,'Lev+5FU')]:
            for i,label in enumerate(FEATURES):row[short+'_'+label]=float(points[g][k,i])
        for j,measure in enumerate(MEASURES+('ROR_over_IDRR',)):
            lo,hi,frac=percentile_interval(bs[:,k,j]);row[measure]=float(pv[k,j]);row[measure+'_ci_low']=lo;row[measure+'_ci_high']=hi
            row[measure+'_valid_fraction']=frac
        out.append(row)
    return pd.DataFrame(out),bs,scenario


def plot_summary(out,path):
    sub=out[out.w==90]
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
    for name,color,marker in [('ROR','#1b7c80','o'),('IDRR','#6a3892','s')]:
        x=sub.t.to_numpy(dtype=float); y=sub[name].to_numpy(dtype=float)
        lo=sub[name+'_ci_low'].to_numpy(dtype=float);hi=sub[name+'_ci_high'].to_numpy(dtype=float)
        valid=np.isfinite(y)&np.isfinite(lo)&np.isfinite(hi)&(y>0)
        axs[0].errorbar(x[valid],y[valid],yerr=[y[valid]-lo[valid],hi[valid]-y[valid]],fmt=marker+'-',ms=4,c=color,capsize=2,label=name if name=='ROR' else 'Direct IDRR')
    axs[0].axhline(1,c='gray',lw=.8,linestyle=':');axs[0].set(xlabel='Follow-up day',ylabel='Ratio (95% patient bootstrap CI)',title='Observed cases and person-time')
    axs[0].legend(fontsize=8,frameon=False)
    x=sub.t.to_numpy(dtype=float);y=sub.ROR_over_IDRR.to_numpy(dtype=float)
    lo=sub.ROR_over_IDRR_ci_low.to_numpy(dtype=float);hi=sub.ROR_over_IDRR_ci_high.to_numpy(dtype=float)
    valid=np.isfinite(lo)&np.isfinite(hi)&np.isfinite(y)&(y>0)
    axs[1].errorbar(x[valid],y[valid],yerr=[y[valid]-lo[valid],hi[valid]-y[valid]],fmt='o-',color='#c07c22',ms=4,capsize=2)
    axs[1].axhline(1,c='gray',lw=.8,ls=':')
    axs[1].set(xlabel='Follow-up day',ylabel='ROR / direct IDRR (95% CI)',title='Paired within-cohort comparison')
    fig.suptitle('Descriptive colon-cancer trial stress test; not stationary population',fontsize=10)
    fig.savefig(path,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input',type=Path)
    src.add_argument('--download',action='store_true')
    src.add_argument('--internal-two-arm',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--reps',type=int,default=4000)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--outdir',type=Path,default=ROOT/'results/python')
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    d,meta=load_records(a.input,a.download,a.internal_two_arm)
    out,_,_=bootstrap(d,reps=a.reps,seed=a.seed)
    expected=pd.read_csv(ROOT/'data/colon_verified_aggregates.csv')
    assert len(out)==len(expected)
    for field in ('t','w','POR','aPOR','ROR','IDRR','ObsH','ObsP','ObsR','ObsA','ObsPT','ObsE','Lev+5FUH','Lev+5FUP','Lev+5FUR','Lev+5FUA','Lev+5FUPT','Lev+5FUE'):
        if field in out:
            actual=out[field]
        elif field.endswith('A'):
            group=field[:-1]
            actual=out[group+'_age_sum']/out[group+'_P']
        else:
            actual=out[field.replace('Obs','Obs_').replace('Lev+5FU','Lev+5FU_')]
        reference=expected[field]
        assert np.allclose(actual.to_numpy(),reference.to_numpy(),rtol=1e-10,atol=1e-8,equal_nan=True),(field,float(np.max(np.abs(actual-reference))))
    out.to_csv(a.outdir/'colon_bootstrap_ci.csv',index=False,float_format='%.15g')
    plot_summary(out,a.outdir/'colon_bootstrap_comparison.pdf')
    import json
    (a.outdir/'colon_bootstrap_provenance.json').write_text(json.dumps({'reps':a.reps,'seed':a.seed,**meta,'method':'stratified within randomized arm patient-level nonparametric percentile bootstrap of paired outcomes; positive finite ratios only; intervals suppressed when <95% of replicates defined','independent_data_claim':False,'stationarity_claim':False},indent=2)+'\n')
    print('Verified all 15 point estimators against original public-source calculations; completed paired bootstrap')
    print(out[['t','w','ROR','ROR_ci_low','ROR_ci_high','IDRR','IDRR_ci_low','IDRR_ci_high','ROR_over_IDRR','ROR_over_IDRR_ci_low','ROR_over_IDRR_ci_high','ROR_over_IDRR_valid_fraction']].to_string(index=False))

if __name__=='__main__':main()
