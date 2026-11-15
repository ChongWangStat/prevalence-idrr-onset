#!/usr/bin/env python3
"""Exact stationary survey simulation, 56 scenarios x five sizes x eight estimators.

Provenance: method parameters and definitions from code/00_functions.R,
code/02_main_simulation.R. No R execution or historical random streams.

Statistical model: independent Poisson H and P counts. Conditional on P,
prevalent cases have independent backward ages with density S(a)/E(T).
Use the exact length-biased total duration and U(0,1) age construction.

Run: python code/python_main_reproduction.py --reps 10000 --out results/python
Run one scenario: ... --scenario 1 --reps 10000
"""
import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats, special
from numba import njit

D0=180.0; I0=0.001; GAM=0.0005
SHAPE={'constant':0,'uniform':1,'exponential':2,'lognormal':3}
ESTS=('POR','aPOR','ROR30','ROR60','ROR90','EOR90','EOR180','EORInf')
SIZES=(200,500,1000,2000,5000)
SEED=2026111200

@njit(cache=True)
def draw_age(shape,d,beta):
    # A (in a stationary prevalence sample) has density S(a)/mean_duration.
    while True:
        u=np.random.random()
        if shape==0:
            a=d*u
        elif shape==1:
            lo=d*.5;hi=d*1.5
            t=math.sqrt(lo*lo+np.random.random()*(hi*hi-lo*lo))
            a=u*t
        elif shape==2:
            a=-d*math.log(max(1e-300,np.random.random()))
        else:
            a=u*math.exp(math.log(d)+.5+np.random.normal())
        if beta<=0 or np.random.random()<math.exp(-beta*a):return a

@njit(cache=True)
def linear_logistic(a,y):
    # Return estimated intercept, model-based variance and fitted slope.
    n=len(a)
    if n<4:return np.nan,np.nan,np.nan
    n1=0
    for z in y:n1+=z
    if n1<2 or n-n1<2:return np.nan,np.nan,np.nan
    scale=np.mean(a)
    if scale<=0:return np.nan,np.nan,np.nan
    b0=math.log(n1/(n-n1));b1=0.;con=False
    for k in range(50):
        i00=0.;i01=0.;i11=0.;g0=0.;g1=0.
        for j in range(n):
            x=a[j]/scale;z=b0+b1*x
            p=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
            w=p*(1-p)
            i00+=w;i01+=w*x;i11+=w*x*x
            dd=y[j]-p;g0+=dd;g1+=dd*x
        det=i00*i11-i01*i01
        if not math.isfinite(det) or det<=1e-12*i00*i11:return np.nan,np.nan,np.nan
        step0=(i11*g0-i01*g1)/det;step1=(i00*g1-i01*g0)/det
        b0+=step0;b1+=step1
        if not math.isfinite(b0) or not math.isfinite(b1) or max(abs(b0),abs(b1))>100:
            return np.nan,np.nan,np.nan
        if max(abs(step0),abs(step1))<1e-9:
            con=True;break
    if not con:return np.nan,np.nan,np.nan
    i00=0.;i01=0.;i11=0.
    for j in range(n):
        x=a[j]/scale;z=b0+b1*x
        p=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        w=p*(1-p);i00+=w;i01+=w*x;i11+=w*x*x
    det=i00*i11-i01*i01
    if det<=0:return np.nan,np.nan,np.nan
    return b0,i11/det,b1/scale

@njit(cache=True)
def ks_stat(x,y,s1,s0):
    i=0;j=0;n=len(x);m=len(y);mx=0.
    while i<n or j<m:
        if j>=m or (i<n and x[i]*s1<=y[j]*s0):i+=1
        else:j+=1
        z=abs(i/n-j/m)
        if z>mx:mx=z
    return mx

@njit(cache=True)
def rank_z(x,y):
    n=len(x);m=len(y);j=0;score=0.
    for i in range(n):
        while j<m and y[j]<x[i]:j+=1
        score+=j
    return (score-.5*n*m)/math.sqrt(n*m*(n+m+1)/12)

@njit(cache=True)
def hl_chi(a,y,theta0,theta1):
    n=len(a)
    if n<50:return np.nan
    ix=np.argsort(a);o=np.zeros(10);e=np.zeros(10);groups=np.zeros(10)
    for j in range(n):
        k=min(9,int(10*j/n));z=ix[j]
        eta=theta0+theta1*a[z]
        p=1/(1+math.exp(-eta)) if eta>=0 else math.exp(eta)/(1+math.exp(eta))
        o[k]+=y[z];e[k]+=p;groups[k]+=1
    v=0.
    for j in range(10):
        if e[j]<=0 or e[j]>=groups[j]:return np.nan
        v+=(o[j]-e[j])**2*(1/e[j]+1/(groups[j]-e[j]))
    return v

@njit(cache=True)
def analyse_survey(h1,h0,a1,a0,row):
    """Estimates, standard errors and test statistics of one survey.
    h1, h0: numbers of healthy individuals; a1, a0: times since onset of the prevalent cases
    (exposed, unexposed). row: columns 0:8 estimates; 8:16 log SE; 16:19 diagnostic tests;
    19:22 recent-fraction z; 22 goodness-of-fit chi-square; 23:27 H1 H0 P1 P0."""
    p1=len(a1);p0=len(a0)
    row[23]=h1;row[24]=h0;row[25]=p1;row[26]=p0
    if h1<1 or h0<1 or p1<1 or p0<1:return
    por=(p1/h1)/(p0/h0);row[0]=por
    v=1/p1+1/h1+1/p0+1/h0;row[8]=math.sqrt(v)
    if p1>1 and p0>1:
        ma1=np.mean(a1);ma0=np.mean(a0)
        var1=np.var(a1)*p1/(p1-1);var0=np.var(a0)*p0/(p0-1)
        row[1]=por*ma0/ma1
        row[9]=math.sqrt(v+var1/(ma1*ma1*p1)+var0/(ma0*ma0*p0))
        x=np.sort(a1);y=np.sort(a0)
        row[16]=rank_z(x,y);row[17]=ks_stat(x,y,1.,1.)
        row[18]=ks_stat(x,y,1./ma1,1./ma0)
    for k in range(3):
        w=30.+30.*k;r1=0;r0=0
        for a in a1:
            if a<=w:r1+=1
        for a in a0:
            if a<=w:r0+=1
        if r1>0 and r0>0:
            row[2+k]=(r1/h1)/(r0/h0)
            row[10+k]=math.sqrt(1/r1+1/r0+1/h1+1/h0)
            if p1>r1 and p0>r0:
                zz=abs(math.log(r1/p1)-math.log(r0/p0))/math.sqrt(1/r1-1/p1+1/r0-1/p0)
                row[19+k]=zz
    for k in range(3):
        cut=90. if k==0 else 180. if k==1 else 1e100
        n1=0;n0=0
        for a in a1:
            if a<=cut:n1+=1
        for a in a0:
            if a<=cut:n0+=1
        if n1<2 or n0<2:continue
        ages=np.empty(n1+n0);labels=np.empty(n1+n0,dtype=np.int64);j=0
        for a in a1:
            if a<=cut:ages[j]=a;labels[j]=1;j+=1
        for a in a0:
            if a<=cut:ages[j]=a;labels[j]=0;j+=1
        b0,var0,b1=linear_logistic(ages,labels)
        if math.isfinite(var0):
            row[5+k]=math.exp(b0)*h0/h1
            row[13+k]=math.sqrt(var0+1/h1+1/h0)
            if k==2:row[22]=hl_chi(ages,labels,b0,b1)

@njit(cache=True)
def sim_batch(reps,n,code1,code0,rho,idrr,beta,seed):
    np.random.seed(seed)
    out=np.full((reps,27),np.nan)
    i1=I0*idrr;d1=D0 if beta>0 else D0*rho
    h1bar=n/(1+i1*D0*rho);h0bar=n/(1+I0*D0)
    p1bar=h1bar*i1*D0*rho;p0bar=h0bar*I0*D0
    for z in range(reps):
        h1=np.random.poisson(h1bar);h0=np.random.poisson(h0bar)
        p1=np.random.poisson(p1bar);p0=np.random.poisson(p0bar)
        a1=np.empty(p1);a0=np.empty(p0)
        for k in range(p1):a1[k]=draw_age(code1,d1,beta)
        for k in range(p0):a0[k]=draw_age(code0,D0,0.)
        analyse_survey(h1,h0,a1,a0,out[z])
    return out

def surv(x,shape,d,beta=0.):
    if shape=='constant':p=float(x<d)
    elif shape=='uniform':p=float(np.clip((1.5*d-x)/d,0,1))
    elif shape=='exponential':p=math.exp(-x/d)
    else:p=float(stats.lognorm.sf(x,s=1,scale=d*math.exp(-.5)))
    return p*math.exp(-beta*x)

def mean_dur(shape,d,beta):
    if beta==0:return d
    top=d if shape=='constant' else 1.5*d if shape=='uniform' else 60*d
    return integrate.quad(lambda x:surv(x,shape,d,beta),0,top,points=[d] if shape=='constant' else [d*.5] if shape=='uniform' else None,epsabs=1e-8)[0]

def get_scenarios():
    out=[]
    for idrr in (1,2):
        for rho in (1.,.8,.6,1.25):
            for sh in ('constant','uniform','exponential','lognormal'):
                out.append(dict(set='A',shape1=sh,shape0=sh,rho=rho,idrr=idrr,beta=0.))
    for idrr in (1,2):
        for rho in (1.,.6):
            for a,b in [('exponential','constant'),('constant','exponential'),('lognormal','uniform')]:
                out.append(dict(set='B',shape1=a,shape0=b,rho=rho,idrr=idrr,beta=0.))
    for idrr in (1,2):
        for rho in (.8,.6):
            for sh in ('constant','uniform','lognormal'):
                beta=optimize.brentq(lambda q:mean_dur(sh,D0,q)-rho*D0,1e-9,1.,xtol=1e-12)
                out.append(dict(set='C',shape1=sh,shape0=sh,rho=rho,idrr=idrr,beta=beta))
    d=pd.DataFrame(out);d.insert(0,'id',np.arange(1,len(d)+1))
    return d

def summarize(raw,s,n,reps):
    ident=dict(id=int(s.id),set=s.set,shape1=s.shape1,shape0=s.shape0,rho=float(s.rho),idrr=int(s.idrr),n=int(n),reps=int(reps))
    rows=[]
    for k,est in enumerate(ESTS):
        vals=raw[:,k];v=raw[:,8+k]
        good=np.isfinite(vals)&(vals>0)&np.isfinite(v)
        num=int(good.sum())
        r=dict(ident,estimator=est,undefined=1-num/reps)
        if num:
            le=np.log(vals[good]);se=v[good];err=le-math.log(s.idrr)
            co=(abs(err)<=stats.norm.ppf(.975)*se);rej=(abs(le)>stats.norm.ppf(.975)*se)
            r.update(mean_est=float(np.mean(vals[good])),median_est=float(np.median(vals[good])),gmean_est=float(np.exp(np.mean(le))),bias_log=float(np.mean(err)),mcse_bias=float(np.std(le,ddof=1)/math.sqrt(num)),emp_se=float(np.std(le,ddof=1)),model_se=float(np.mean(se)),rmse_log=float(np.sqrt(np.mean(err*err))),coverage=float(np.mean(co)),mcse_cov=float(np.sqrt(co.mean()*(1-co.mean())/num)),reject=float(np.mean(rej)),mcse_rej=float(np.sqrt(rej.mean()*(1-rej.mean())/num)))
        else:r.update({key:np.nan for key in ('mean_est','median_est','gmean_est','bias_log','mcse_bias','emp_se','model_se','rmse_log','coverage','mcse_cov','reject','mcse_rej')})
        rows.append(r)
    def rate(p):
        good=np.isfinite(p)
        return float(np.mean(p[good]<.05)) if good.any() else np.nan
    pWil=special.erfc(abs(raw[:,16])/math.sqrt(2))
    def pks(z):
        ne=np.maximum(1,raw[:,25]*raw[:,26]/np.maximum(raw[:,25]+raw[:,26],1))
        s=np.sqrt(ne);return special.kolmogorov(z*(s+.12+.11/s))
    pks0=pks(raw[:,17]);pks1=pks(raw[:,18])
    pp=special.erfc(abs(raw[:,19:22])/math.sqrt(2))
    pHL=stats.chi2.sf(raw[:,22],df=8)
    t=dict(ident,rej_wilcoxon=rate(pWil),rej_ks=rate(pks0),rej_shape=rate(pks1),rej_recent30=rate(pp[:,0]),rej_recent60=rate(pp[:,1]),rej_recent90=rate(pp[:,2]),undef_recent30=float((~np.isfinite(pp[:,0])).mean()),rej_linear=rate(pHL),undef_linear=float((~np.isfinite(pHL)).mean()),mean_P1=float(raw[:,25].mean()),mean_P0=float(raw[:,26].mean()),mean_H1=float(raw[:,23].mean()),mean_H0=float(raw[:,24].mean()),var_P0=float(raw[:,26].var(ddof=1)),var_H0=float(raw[:,24].var(ddof=1)),cor_H0P0=float(np.corrcoef(raw[:,24],raw[:,26])[0,1]))
    return rows,t

def main():
    a=argparse.ArgumentParser();a.add_argument('--reps',type=int,default=10000);a.add_argument('--scenario',nargs='*',type=int);a.add_argument('--out',type=Path,default=Path('results/python'));a.add_argument('--resume',action='store_true');args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sc=get_scenarios();sc.to_csv(args.out/'scenarios.csv',index=False)
    if args.resume and (args.out/'main_summary_python.csv').exists():
        summaries=pd.read_csv(args.out/'main_summary_python.csv').to_dict('records')
        tests=pd.read_csv(args.out/'main_tests_python.csv').to_dict('records')
        seen=set((int(t['id']),int(t['n'])) for t in tests if int(t['reps'])==args.reps)
    else:summaries=[];tests=[];seen=set()
    start=time.perf_counter()
    for s in sc.itertuples(index=False):
        if args.scenario and s.id not in args.scenario:continue
        for i,n in enumerate(SIZES):
            if (s.id,n) in seen:continue
            raw=sim_batch(args.reps,int(n),SHAPE[s.shape1],SHAPE[s.shape0],float(s.rho),float(s.idrr),float(s.beta),SEED+10000*s.id+i)
            rr,tt=summarize(raw,s,n,args.reps);summaries.extend(rr);tests.append(tt)
        pd.DataFrame(summaries).to_csv(args.out/'main_summary_python.csv',index=False,float_format='%.15g')
        pd.DataFrame(tests).to_csv(args.out/'main_tests_python.csv',index=False,float_format='%.15g')
        print('scenario',s.id,'/',len(sc),'elapsed',round(time.perf_counter()-start,1),'sec',flush=True)
    print('completed',len(summaries),'estimator rows; ',len(tests),'diagnostic rows',flush=True)
if __name__=='__main__':main()
