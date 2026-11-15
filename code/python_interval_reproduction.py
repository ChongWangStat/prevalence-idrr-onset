#!/usr/bin/env python3
"""Python-only onset-interval inference, exact Fisher conditional intervals.

Reproduces R code/01_interval_functions.R and 08_interval_onset.R with
independent stationary Poisson sampling, prespecified seeds, and a numba
noncentral-hypergeometric confidence-interval inversion. Contains all 13
scenarios × 3 sizes × 10,000 replicates and sharp sensitivity limits.
"""
import argparse,math,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy import integrate,optimize,special,stats
from numba import njit
from python_main_reproduction import (D0,I0,GAM,SHAPE,draw_age,linear_logistic,get_scenarios,surv)
SEED=2026111210
SCEN_IDS=(19,27,11,26,28,39,42,43,44,54,56,57,58)
WINS=(30.,60.,90.);DELTAS=(0.,.005,.01,.02)
ESTS=('ROR30','EOR90','EOR90i','EOR60i','TW60','TW90')

@njit(cache=True)
def fisher_tails(a,b,c,d,theta):
    r=a+b;r0=c+d;col=a+c
    lo=max(0,col-r0);hi=min(col,r);n=hi-lo+1
    logw=np.empty(n);logw[0]=0.
    for j in range(1,n):
        k=lo+j-1
        logw[j]=logw[j-1]+math.log(r-k)-math.log(k+1)+math.log(col-k)-math.log(r0-col+k+1)+theta
    mx=np.max(logw);tot=0.;low=0.;up=0.
    for j in range(n):
        wt=math.exp(logw[j]-mx);tot+=wt
        if j+lo<=a:low+=wt
        if j+lo>=a:up+=wt
    return low/tot,up/tot

@njit(cache=True)
def fisher_ci(a,b,c,d,alpha):
    if b==0 or d==0:return 0.,np.inf
    lower=0.;upper=np.inf
    if a>0:
        lo=-30.;hi=30.
        for j in range(31):
            mid=(lo+hi)/2
            _,tail=fisher_tails(a,b,c,d,mid)
            if tail<alpha/2:lo=mid
            else:hi=mid
        lower=math.exp((lo+hi)/2)
    if c>0:
        lo=-30.;hi=30.
        for j in range(31):
            mid=(lo+hi)/2
            tail,_=fisher_tails(a,b,c,d,mid)
            if tail<alpha/2:hi=mid
            else:lo=mid
        upper=math.exp((lo+hi)/2)
    return lower,upper

@njit(cache=True)
def grouped_logistic(mid,c1,c0):
    total1=np.sum(c1);total0=np.sum(c0);n=np.sum(c1+c0)
    if total1<2 or total0<2:return np.nan,np.nan
    nt=0
    for j in range(len(mid)):
        if c1[j]+c0[j]>0:nt+=1
    if nt<2:return np.nan,np.nan
    scale=np.sum((c1+c0)*mid)/n
    b0=math.log(total1/total0);b1=0.;conv=False
    for it in range(70):
        i00=0.;i01=0.;i11=0.;g0=0.;g1=0.
        for j in range(len(mid)):
            nn=c1[j]+c0[j]
            if nn<=0:continue
            x=mid[j]/scale;z=b0+b1*x
            p=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
            w=nn*p*(1-p);i00+=w;i01+=w*x;i11+=w*x*x
            dif=c1[j]-nn*p;g0+=dif;g1+=dif*x
        det=i00*i11-i01*i01
        if det<=1e-12*i00*i11:return np.nan,np.nan
        dt0=(i11*g0-i01*g1)/det;dt1=(i00*g1-i01*g0)/det
        b0+=dt0;b1+=dt1
        if not math.isfinite(b0) or not math.isfinite(b1) or max(abs(b0),abs(b1))>100:return np.nan,np.nan
        if max(abs(dt0),abs(dt1))<1e-10:conv=True;break
    if not conv:return np.nan,np.nan
    i00=0.;i01=0.;i11=0.
    for j in range(len(mid)):
        nn=c1[j]+c0[j]
        if nn<=0:continue
        x=mid[j]/scale;z=b0+b1*x
        p=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        w=nn*p*(1-p);i00+=w;i01+=w*x;i11+=w*x*x
    det=i00*i11-i01*i01
    if det<=0:return np.nan,np.nan
    return b0,i11/det

@njit(cache=True)
def simulate_batch(reps,n,shape1,shape0,rho,idrr,beta,seed):
    np.random.seed(seed)
    d1=D0 if beta>0 else D0*rho
    h1bar=n/(1+I0*idrr*D0*rho);h0bar=n/(1+I0*D0)
    p1bar=h1bar*I0*idrr*D0*rho;p0bar=h0bar*I0*D0
    # 0:6 estimates; 6:12 SE; 12:30 3 windows x 6 CI limits
    out=np.full((reps,30),np.nan)
    x90=np.array([15.,45.,75.]);x60=np.array([15.,45.])
    for j in range(reps):
        h1=np.random.poisson(h1bar);h0=np.random.poisson(h0bar)
        p1=np.random.poisson(p1bar);p0=np.random.poisson(p0bar)
        a1=np.empty(p1);a0=np.empty(p0)
        for k in range(p1):a1[k]=draw_age(shape1,d1,beta)
        for k in range(p0):a0[k]=draw_age(shape0,D0,0.)
        c1=np.zeros(3,np.int64);c0=np.zeros(3,np.int64)
        b1=np.zeros(3,np.float64);b0=np.zeros(3,np.float64)
        half1=np.zeros(2,np.int64);half0=np.zeros(2,np.int64)
        for v in a1:
            for k in range(3):
                if v<=WINS[k]:c1[k]+=1
            if v<=90:b1[min(2,int(v/30))]+=1
            if v<=30:half1[0]+=1
            if v<=45:half1[1]+=1
        for v in a0:
            for k in range(3):
                if v<=WINS[k]:c0[k]+=1
            if v<=90:b0[min(2,int(v/30))]+=1
            if v<=30:half0[0]+=1
            if v<=45:half0[1]+=1
        if h1<=0 or h0<=0:continue
        for k in range(3):
            z=12+6*k;r1=c1[k];r0=c0[k]
            if r1>0 and r0>0:
                por=(r1/h1)/(r0/h0);se=math.sqrt(1/r1+1/h1+1/r0+1/h0)
                out[j,z]=por*math.exp(-1.95996398454*se)
                out[j,z+1]=por*math.exp(1.95996398454*se)
                if k==0:out[j,0]=por;out[j,6]=se
            l,u=fisher_ci(r1,h1,r0,h0,.05)
            out[j,z+2]=l;out[j,z+3]=u
            l,u=fisher_ci(r1,h1,r0,h0,.05/3)
            out[j,z+4]=l;out[j,z+5]=u
        if c1[2]>=2 and c0[2]>=2:
            ages=np.empty(c1[2]+c0[2]);yy=np.empty(c1[2]+c0[2],np.int64);k=0
            for v in a1:
                if v<=90:ages[k]=v;yy[k]=1;k+=1
            for v in a0:
                if v<=90:ages[k]=v;yy[k]=0;k+=1
            intercept,variance,_=linear_logistic(ages,yy)
            if math.isfinite(variance):
                out[j,1]=math.exp(intercept)*h0/h1
                out[j,7]=math.sqrt(variance+1/h1+1/h0)
        for kk in range(2):
            if kk==0:intc,var=grouped_logistic(x90,b1,b0)
            else:intc,var=grouped_logistic(x60,b1[:2],b0[:2])
            if math.isfinite(var):
                out[j,2+kk]=math.exp(intc)*h0/h1
                out[j,8+kk]=math.sqrt(var+1/h1+1/h0)
        for kk in range(2):
            a1n=half1[kk];a0n=half0[kk];p1n=c1[1+kk];p0n=c0[1+kk]
            cc1=4*a1n-p1n;cc0=4*a0n-p0n
            if cc1>0 and cc0>0:
                out[j,4+kk]=(cc1/h1)/(cc0/h0)
                out[j,10+kk]=math.sqrt((8*a1n+p1n)/(cc1*cc1)+(8*a0n+p0n)/(cc0*cc0)+1/h1+1/h0)
    return out

def rmean(w,shape,D,beta):
    bounds=[x for x in (D*.5,D,D*1.5) if 0<x<w] if shape in ('constant','uniform') else None
    end=min(w,D if shape=='constant' else D*1.5 if shape=='uniform' else 100*D)
    bounds=[x for x in (bounds or []) if x<end]
    return integrate.quad(lambda a:surv(a,shape,D,beta),0,end,points=bounds or None,epsabs=1e-9)[0]

def params(row):return (D0,float(row.beta)) if row.set=='C' else (D0*row.rho,0.)

def limits_all(sc):
    rows=[]
    for row in sc.itertuples(index=False):
        d1,beta=params(row);d0=D0
        def rm(w,sh,D,b=0.):return rmean(w,sh,D,b)
        vals=dict(id=row.id,set=row.set,shape1=row.shape1,shape0=row.shape0,rho=row.rho,idrr=row.idrr)
        for w in (30,60,90):
            vals[f'ROR{w}']=rm(w,row.shape1,d1,beta)/rm(w,row.shape0,d0)
        for w in (60,90):
            vals[f'TW{w}']=(4*rm(w/2,row.shape1,d1,beta)-rm(w,row.shape1,d1,beta))/(4*rm(w/2,row.shape0,d0)-rm(w,row.shape0,d0))
        delta={}
        for w in (30,90):
            x=np.linspace(w/2000,w,2000)
            a=np.array([surv(v,row.shape1,d1,beta) for v in x]);b=np.array([surv(v,row.shape0,d0,0) for v in x])
            slope=np.log(a/b)/x
            vals[f'd1_{w}']=max(0.,float(-np.nanmin(slope)))
            vals[f'd0_{w}']=max(0.,float(np.nanmax(slope)))
        # Grouped logistic population intercept from interval-integrated intensities.
        def limit_eor_window(breaks):
            a=np.asarray(breaks)
            n1=np.array([rm(x2,row.shape1,d1,beta)-rm(x1,row.shape1,d1,beta) for x1,x2 in zip(a[:-1],a[1:])])
            n0=np.array([rm(x2,row.shape0,d0)-rm(x1,row.shape0,d0) for x1,x2 in zip(a[:-1],a[1:])])
            t0,tv=grouped_logistic((a[1:]+a[:-1])/2,n1*1e6,n0*1e6)
            return math.exp(t0)
        vals['EOR90i']=limit_eor_window([0,30,60,90]);vals['EOR60i']=limit_eor_window([0,30,60]);vals['EOR28w']=limit_eor_window(list(range(0,29,7)));vals['EOR91w']=limit_eor_window(list(range(0,92,7)))
        # Continuous regression population solution by quadrature grid, using independent exposure weights.
        x=np.linspace(0,90,1501)
        w1=np.array([surv(v,row.shape1,d1,beta) for v in x]);w0=np.array([surv(v,row.shape0,d0,0) for v in x]);
        t0,tv=grouped_logistic(x,w1*1e6,w0*1e6)
        vals['EOR90']=math.exp(t0)
        rows.append(vals)
    return pd.DataFrame(rows)

def bfun(z):return -math.expm1(-z)/z if z>1e-12 else 1-z/2

def results(raw,row,n,reps,lim):
    ident={k:getattr(row,k) for k in ('id','set','shape1','shape0','rho','idrr')};ident.update(n=n,reps=reps)
    est=[];bounds=[];truth=math.log(float(row.idrr))
    for k,e in enumerate(ESTS):
        v=raw[:,k];se=raw[:,6+k];ok=(v>0)&np.isfinite(v)&np.isfinite(se)
        if ok.sum()>1:
            lv=np.log(v[ok]);error=lv-truth
            est.append(dict(ident,estimator=e,undefined=float(1-ok.mean()),limit=float(lim[e]),gmr=float(np.exp(lv.mean())/row.idrr),emp_se=float(lv.std(ddof=1)),mean_se=float(se[ok].mean()),rmse=float(np.sqrt(np.mean(error**2))),coverage=float(np.mean(abs(error)<=stats.norm.ppf(.975)*se[ok]))))
        else:est.append(dict(ident,estimator=e,undefined=1.,limit=float(lim[e]),gmr=np.nan,emp_se=np.nan,mean_se=np.nan,rmse=np.nan,coverage=np.nan))
    d1,d0=float(lim.d1_90),float(lim.d0_90)
    # Exact60 must use its OWN 60-day hazard bound, not the more conservative
    # 90-day bound previously used only as a labeling shortcut.
    duration1=D0 if row['set']=='C' else D0*float(row.rho)
    duration0=D0
    mesh=np.linspace(60/2000,60,2000)
    s1=np.array([surv(a,row.shape1,duration1,float(row.beta)) for a in mesh])
    s0=np.array([surv(a,row.shape0,duration0,0.) for a in mesh])
    slope=np.log(s1/s0)/mesh
    d60=max(0.,-float(np.nanmin(slope)),float(np.nanmax(slope)))
    for delta in (0,.005,.01,.02):
        f=np.array([bfun(delta*w) for w in WINS]);ivs={}
        for kk,w in enumerate(WINS):
            j=12+6*kk
            if kk==0:ivs['Wald30']=(raw[:,j]*f[kk],raw[:,j+1]/f[kk])
            ivs[f'Exact{int(w)}']=(raw[:,j+2]*f[kk],raw[:,j+3]/f[kk])
        ivs['ExactAll']=(np.maximum.reduce([raw[:,12+6*k+4]*f[k] for k in range(3)]),np.minimum.reduce([raw[:,12+6*k+5]/f[k] for k in range(3)]))
        for method,(lo,hi) in ivs.items():
            ok=np.isfinite(lo)&~np.isnan(hi);l=lo[ok];u=hi[ok]
            wi=2 if method=='ExactAll' else 0 if method in ('Exact30','Wald30') else 1 if method=='Exact60' else 2
            dmin=max(float(lim.d1_30),float(lim.d0_30)) if wi==0 else d60 if wi==1 else max(d1,d0)
            bounds.append(dict(ident,method=method,delta=delta,delta_min=dmin,valid=bool(dmin<=delta+1e-12),undefined=1-ok.mean(),coverage=float(np.mean((l<=row.idrr)&(row.idrr<=u))),med_lower=float(np.median(l)/row.idrr),med_upper=float(np.median(u)/row.idrr),lower_gt1=float(np.mean((l>1)&(l<=u))),upper_lt1=float(np.mean((u<1)&(l<=u))),unbounded=float(np.mean((l==0)|(~np.isfinite(u)))),empty=float(np.mean(l>u))))
    return est,bounds

def main():
    a=argparse.ArgumentParser();a.add_argument('--reps',type=int,default=10000);a.add_argument('--scenario',nargs='*',type=int);a.add_argument('--out',type=Path,default=Path('results/python'));a.add_argument('--resume',action='store_true');args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sc=get_scenarios();D=pd.DataFrame([dict(id=57,set='D',shape1='exponential',shape0='constant',rho=50/180,idrr=2,beta=0.),dict(id=58,set='D',shape1='exponential',shape0='constant',rho=15/180,idrr=2,beta=0.)]);sc=pd.concat([sc,D],ignore_index=True)
    limits=limits_all(sc);limits.to_csv(args.out/'interval_limits_python.csv',index=False,float_format='%.15g')
    if args.resume and (args.out/'interval_estimators_python.csv').exists():
        E=pd.read_csv(args.out/'interval_estimators_python.csv').to_dict('records');B=pd.read_csv(args.out/'interval_bounds_python.csv').to_dict('records');done=set((int(x['id']),int(x['n'])) for x in E if int(x['reps'])==args.reps)
    else:E=[];B=[];done=set()
    for id in SCEN_IDS:
        if args.scenario and id not in args.scenario:continue
        row=sc[sc.id==id].iloc[0];lim=limits[limits.id==id].iloc[0]
        for j,n in enumerate((200,1000,5000)):
            if (id,n) in done:continue
            t=time.perf_counter()
            x=simulate_batch(args.reps,n,SHAPE[row.shape1],SHAPE[row.shape0],float(row.rho),float(row.idrr),float(row.beta),SEED+10000*id+j)
            e,b=results(x,row,n,args.reps,lim);E+=e;B+=b
            pd.DataFrame(E).to_csv(args.out/'interval_estimators_python.csv',index=False,float_format='%.15g')
            pd.DataFrame(B).to_csv(args.out/'interval_bounds_python.csv',index=False,float_format='%.15g')
            print(id,n,'elapsed',round(time.perf_counter()-t,1),flush=True)
    print('rows',len(E),len(B))
if __name__=='__main__':main()
