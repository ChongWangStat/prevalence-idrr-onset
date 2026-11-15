#!/usr/bin/env python3
"""Every LaTeX table body generated from Python-only simulation CSVs.

No historical R files are read. Theoretical tables are computed by independent
Python quadrature and a population-level logistic score fit, reproducing the
paper's asymptotic definitions. All functions have deterministic inputs.
"""
from pathlib import Path
from functools import lru_cache
import math,argparse
import numpy as np,pandas as pd
from scipy import integrate,stats,special,optimize
ROOT=Path(__file__).resolve().parent.parent
OUTPUTS=ROOT/'results/python';TEX=ROOT/'tex';D0=180.;I0=.001
SH=('constant','uniform','exponential','lognormal')
LABEL=dict(constant='Constant',uniform='Uniform',exponential='Exponential',lognormal='Log-normal')
EST=('POR','aPOR','ROR30','ROR60','ROR90','EOR90','EOR180','EORInf')
ESTTEX=dict(POR='POR',aPOR='aPOR',ROR30='ROR(30)',ROR60='ROR(60)',ROR90='ROR(90)',EOR90='EOR(90)',EOR180='EOR(180)',EORInf=r'EOR($\infty$)')
SIZES=(200,500,1000,2000,5000);ORD=(19,27,11,26,28,39,42,43,44,54,56,57,58)
f1=lambda x:f'{float(x):.1f}'
f2=lambda x:f'{float(x):.2f}'
f3=lambda x:f'{float(x):.3f}'
f4=lambda x:f'{float(x):.4f}'
pc=lambda x:f'{float(x)*100:.1f}'
pc0=lambda x:f'{float(x)*100:.0f}'
fmtid=lambda x: str(int(x)).replace('10000',r'10\,000')

def row(x):return ' & '.join(map(str,x))+r' \\'
def write(n,rows):
    TEX.joinpath(n).write_text('\n'.join(z if isinstance(z,str) else row(z) for z in rows)+'\n')
def lab1(ss,s):return LABEL[s]+(r'$^{+}$' if ss=='C' else '')
def sub(d,**filters):
    for name,v in filters.items():d=d[np.isclose(d[name],v) if name in ('rho','kappa','delta') else d[name]==v]
    return d
def get(d,col,**kw):
    x=sub(d,**kw)
    if len(x)!=1:raise ValueError(f'{kw} -> {len(x)} rows for {col}')
    return x.iloc[0][col]
def lookup_dict(d,**kw):
    x=sub(d,**kw)
    if len(x)!=1:raise ValueError((kw,len(x)))
    return x.iloc[0]
def sv(x,shape,D,beta=0):
    x=np.asarray(x,dtype=float)
    if shape=='constant':v=np.array(x<D,dtype=float)
    elif shape=='uniform':v=np.clip((1.5*D-x)/D,0,1)
    elif shape=='exponential':v=np.exp(-x/D)
    else:v=stats.lognorm.sf(x,1.,scale=D*math.exp(-.5))
    return v*np.exp(-beta*x)
def rmean(w,shape,D,beta=0):
    if w<=0:return 0.
    hi= min(w,D if shape=='constant' else 1.5*D if shape=='uniform' else 1e6)
    pts=[p for p in (D/2,D,1.5*D) if 0<p<hi] if shape in ('constant','uniform') else None
    return integrate.quad(lambda x:float(sv(x,shape,D,beta)),0,hi,points=pts,epsabs=1e-9)[0]
def intmeanage(shape,D,beta=0):
    hi=D if shape=='constant' else 1.5*D if shape=='uniform' else 1e6
    pts=[p for p in (D/2,D,1.5*D) if 0<p<hi] if shape in ('constant','uniform') else None
    return integrate.quad(lambda x:float(x*sv(x,shape,D,beta)),0,hi,points=pts,epsabs=1e-8)[0]

def logistic_lim(w,s1,D1,b1,s0,D0x,idrr):
    m1=rmean(1e6,s1,D1,b1);m0=rmean(1e6,s0,D0x)
    r=(idrr*I0/(1+idrr*I0*m1))/(I0/(1+I0*m0))
    if math.isinf(w):
        # upper 1-1e-9 quantile (same truncation as original R code)
        def up(shape,D):
            if shape=='constant':return D
            if shape=='uniform':return 1.5*D
            if shape=='exponential':return -D*math.log(1e-9)
            return float(stats.lognorm.ppf(1-1e-9,1.,scale=D*math.exp(-.5)))
        w=max(up(s1,D1),up(s0,D0x))
    else:w=float(w)
    if w<=2000:a=np.linspace(0,w,20001)
    else:a=np.r_[np.arange(0,2000,.1),np.arange(2000,w,5),w]
    if len(a)<2:return float('nan')
    h=np.diff(a);q=(np.r_[h,0]+np.r_[0,h])/2
    v1=q*r*sv(a,s1,D1,b1);v0=q*sv(a,s0,D0x)
    mean=(v1+v0);x=a/max(float(a.max()),1.)
    theta=np.array([math.log(v1.sum()/v0.sum()),0.])
    for i in range(120):
        pr=special.expit(theta[0]+theta[1]*x);wt=mean*pr*(1-pr)
        g=np.array([(v1-mean*pr).sum(),((v1-mean*pr)*x).sum()])
        H=np.array([[wt.sum(),(wt*x).sum()],[(wt*x).sum(),(wt*x*x).sum()]])
        d=np.linalg.solve(H,g)
        theta+=d
        if np.max(np.abs(d))<1e-10:break
    return math.exp(theta[0])/r

@lru_cache(maxsize=1200)
def limit(e,ss,s1,s0,rho,idrr=2):
    if ss=='C':
        from python_main_reproduction import mean_dur
        D1=D0;b1=optimize.brentq(lambda b:mean_dur(s1,D0,b)-rho*D0,1e-9,1.,xtol=1e-12)
    else:D1=D0*rho;b1=0.
    m1=rmean(1e6,s1,D1,b1);m0=rmean(1e6,s0,D0)
    if e=='POR':return m1/m0
    if e=='aPOR':return (m1/m0)*(intmeanage(s0,D0,0)/m0)/(intmeanage(s1,D1,b1)/m1)
    if e.startswith('ROR'):return rmean(int(e[3:]),s1,D1,b1)/rmean(int(e[3:]),s0,D0)
    return logistic_lim(math.inf if e=='EORInf' else int(e[3:]),s1,D1,b1,s0,D0,idrr)

def main():
    need=['main_summary','main_tests','original_summary','original_stationarity','validation','trajectories','immigration','growth','changing_incidence','interval_limits','interval_estimators','interval_bounds']
    missing=[v for v in need if not (OUTPUTS/(v+'_python.csv')).is_file()]
    if missing:raise FileNotFoundError('Missing independently Python-generated results: '+str(missing))
    d={n:pd.read_csv(OUTPUTS/(n+'_python.csv')) for n in need}
    S=d['main_summary'];T=d['main_tests'];OS=d['original_summary'];ST=d['original_stationarity'];V=d['validation'];M=d['immigration'];G=d['growth'];CI=d['changing_incidence'];IL=d['interval_limits'];IE=d['interval_estimators'];IB=d['interval_bounds']
    assert (len(S),len(T),len(OS),len(ST),len(IL),len(IE),len(IB))==(2240,280,24,123,58,234,780)
    print('All Python result tables loaded, doing quadrature + typesetting',flush=True)
    def g(ss,s1,s0,rho,idrr,n,e,col):return get(S,col,set=ss,shape1=s1,shape0=s0,rho=rho,idrr=idrr,n=n,estimator=e)
    def gt(ss,s1,s0,rho,idrr,n,col):return get(T,col,set=ss,shape1=s1,shape0=s0,rho=rho,idrr=idrr,n=n)
    pairs=[('exponential','constant'),('constant','exponential'),('lognormal','uniform')]
    sB=[('B',s1,s0,rho) for s1,s0 in pairs for rho in (1.,.6)]
    sC=[('C',sh,sh,rho) for sh in ('constant','uniform','lognormal') for rho in (.8,.6)]
    scenarios=[('A',sh,sh,rho) for sh in SH for rho in (1.,.8,.6,1.25)] +sB+sC
    # Supplementary theory table and main text table
    rows=[]
    for sh in SH:
        for rho in (.6,.8,1.25):
            rows.append([LABEL[sh] if rho==.6 else '', LABEL[sh] if rho==.6 else '', f2(rho)]+[f3(limit(e,'A',sh,sh,rho)) for e in ('POR','aPOR','ROR30','ROR90','EOR90','EORInf')])
    rows.append(r'\midrule')
    for ss,s1,s0,rho in sB+sC:
        if ss=='C' and (ss,s1,s0,rho)==sC[0]:rows.append(r'\midrule')
        rows.append([lab1(ss,s1) if (rho==1 or (ss=='C' and rho==.8)) else '',LABEL[s0] if (rho==1 or (ss=='C' and rho==.8)) else '',f2(rho)]+[f3(limit(e,ss,s1,s0,rho)) for e in ('POR','aPOR','ROR30','ROR90','EOR90','EORInf')])
    write('tab_limits_body.tex',rows)
    rows=[]
    for sh in SH:
        for n in SIZES:
            cells=[LABEL[sh] if n==200 else '',fmtid(n)]
            for rho in (1.,.6):cells += [f3(g('A',sh,sh,rho,2,n,'POR',col)) if col!='coverage' else pc(g('A',sh,sh,rho,2,n,'POR',col)) for col in ('mean_est','median_est','emp_se','rmse_log','coverage')]
            rows.append(cells)
    write('tab_por_body.tex',rows)
    rows=[]
    for sh in SH:
        for n in (200,1000,5000):
            cells=[LABEL[sh] if n==200 else '',fmtid(n)]
            for e in ('aPOR','ROR30','EOR90','EORInf'):
                cells +=[f3(g('A',sh,sh,.6,2,n,e,'gmean_est')/2),f3(g('A',sh,sh,.6,2,n,e,'emp_se')),pc(g('A',sh,sh,.6,2,n,e,'coverage'))]
            rows.append(cells)
    write('tab_adj_body.tex',rows)
    rows=[]
    for rho in (1.,.8,.6,1.25):
        for n in (200,1000,5000):
            cells=[f2(rho) if n==200 else '',fmtid(n)]
            for e in ('POR','aPOR','ROR30','EOR90','EORInf'):
                v=[g('A',sh,sh,rho,1,n,e,'reject') for sh in SH];lo,hi=pc(min(v)),pc(max(v));cells.append(lo if lo==hi else lo+'--'+hi)
            rows.append(cells)
    write('tab_null_body.tex',rows)
    rows=[];rowsT=[]
    for k,(ss,s1,s0,rho) in enumerate(sB+sC):
        if k==len(sB):rows.append(r'\midrule');rowsT.append(r'\midrule')
        first=k==0 or k==len(sB) or k>0 and (sB+sC)[k-1][1:3]!=(s1,s0)
        ident=[lab1(ss,s1) if first else '',LABEL[s0] if first else '',f2(rho)]
        cells=ident.copy()
        for e in ('POR','aPOR','ROR30','EOR90','EORInf'):cells+=[f3(g(ss,s1,s0,rho,2,2000,e,'gmean_est')/2),pc(g(ss,s1,s0,rho,2,2000,e,'coverage'))]
        rows.append(cells)
        cells=ident.copy()
        for col in ('rej_wilcoxon','rej_ks','rej_recent30','rej_shape','rej_linear'):cells.append(pc(gt(ss,s1,s0,rho,2,2000,col)))
        rowsT.append(cells)
    write('tab_setBC_body.tex',rows);write('tab_setBC_tests_body.tex',rowsT)
    # 2240 supplementary estimator rows; calculate and cache exact theoretical limits.
    for idrr in (1,2):
        for sets,label in [(('A',),'A'),(('B','C'),'BC')]:
            rows=[]
            groups=sub(S,idrr=idrr)
            groups=groups[groups.set.isin(sets)][['set','shape1','shape0','rho']].drop_duplicates()
            for z in groups.itertuples(index=False):
                ss,s1,s0,rho=z
                for n in SIZES:
                    for e in EST:
                        first=(n==200 and e=='POR');r=lookup_dict(S,set=ss,shape1=s1,shape0=s0,rho=rho,idrr=idrr,n=n,estimator=e)
                        cells=[lab1(ss,s1) if first else '',LABEL[s0] if first else '',f2(rho) if first else '',fmtid(n) if e=='POR' else '',ESTTEX[e],f3(limit(e,ss,s1,s0,rho,idrr)),f3(r.gmean_est/idrr),f3(r.emp_se),f3(r.model_se),f3(r.rmse_log),pc(r.coverage),pc(r.reject),pc(r.undefined)]
                        rows.append(cells)
            write(f'tab_supp_{label}_idrr{idrr}_body.tex',rows)
            print('supp estimates',label,idrr,len(rows),flush=True)
    # diagnostics & recent onset tests
    unique=T[T.idrr==2][['set','shape1','shape0','rho']].drop_duplicates()
    def tests(columns,ns):
        rr=[]
        for idrr in (2,1):
            for k,z in enumerate(unique.itertuples(index=False)):
                ss,s1,s0,rho=z;cells=[str(idrr) if k==0 else '',lab1(ss,s1),LABEL[s0],f2(rho)]
                for col in columns:
                    for n in ns:cells.append(pc(gt(ss,s1,s0,rho,idrr,n,col)))
                rr.append(cells)
        return rr
    write('tab_supp_tests_body.tex',tests(('rej_wilcoxon','rej_ks','rej_recent30','rej_shape'),(200,1000,5000)))
    write('tab_supp_linear_body.tex',tests(('rej_linear','undef_linear'),SIZES))
    rows=[]
    for z in unique.itertuples(index=False):
        ss,s1,s0,rho=z;cells=[lab1(ss,s1),LABEL[s0],f2(rho)]
        for col in ('rej_recent30','rej_recent60','rej_recent90'):
            for n in (200,1000,2000,5000):cells.append(pc(gt(ss,s1,s0,rho,2,n,col)))
        rows.append(cells)
    write('tab_supp_window_body.tex',rows)
    rows=[]
    for r in V.itertuples(index=False):rows.append([LABEL[r.shape],f2(r.EH_theory),f2(r.EH_sim),f2(r.varH_sim),f2(r.EP_theory),f2(r.EP_sim),f2(r.varP_sim),f3(r.cor_HP),f2(r.EA_theory),f2(r.EA_sim),f3(r.F30_theory),f3(r.F30_sim),f3(r.ks_stat)])
    write('tab_supp_validation_body.tex',rows)
    rows=[]
    for r in M.itertuples(index=False):rows.append([f2(r.rho),f2(r.mrel),f3(r.por_all_theory),f3(r.por_all),f3(r.por_local),f3(r.apor_local),f3(r.ror30_local)])
    write('tab_supp_immigration_body.tex',rows)
    rows=[]
    for sh,kap in G[['shape','kappa']].drop_duplicates().itertuples(index=False):
        cells=[LABEL[sh],f'{kap:.3f}']
        for e in ('POR','aPOR','ROR30','EOR90','EORInf'):
            z=lookup_dict(G,shape=sh,kappa=kap,estimator=e);cells+=[f3(z.limit),f3(z.gmr),pc(z.coverage)]
        rows.append(cells)
    write('tab_supp_growth_body.tex',rows)
    text={'rise':'Rise','fall':'Fall','both':'Rise in both groups','step':'Step','rise6':r'Rise, $\rho=0.6$'}
    rows=[]
    for scen in CI.scenario.unique():
        cells=[text[scen]]
        for e in ('POR','aPOR','ROR30','EOR90','EORInf'):
            z=lookup_dict(CI,scenario=scen,estimator=e);cells+=[f3(z.limit),f3(z.gmr),pc(z.coverage)]
        rows.append(cells)
    write('tab_supp_change_body.tex',rows)
    # Original Aims 1--3 and tabulated stationarity
    for comparison,file in [('1-2','tab_orig_same_body.tex'),('1-3','tab_orig_diff_body.tex')]:
        rows=[]
        for sh in ('constant','uniform','exponential'):
            rows+=[r'\addlinespace[0.3em]',r'\multicolumn{6}{l}{\textit{\textbf{'+sh+r'}}}\\']
            for n in (100,200,500,1000):
                v=lookup_dict(OS,comparison=comparison,shape=sh,N0=n)
                rows.append(r'\hspace{1em} & $N_0='+str(n)+'$ & '+' & '.join(f4(v[c]) for c in ('mean','se','bias','rmse'))+r' \\')
            if sh!='exponential':rows.append(r'\cmidrule{1-6}')
        write(file,rows)
    rows=[]
    for sh in ('constant','uniform','exponential'):
        rows+=[r'\addlinespace[0.3em]',r'\multicolumn{10}{l}{\textit{\textbf{'+sh+r'}}}\\']
        for n in (100,200,500,1000):
            v2=lookup_dict(OS,comparison='1-2',shape=sh,N0=n);v3=lookup_dict(OS,comparison='1-3',shape=sh,N0=n)
            rows.append(r'\hspace{1em} & $N_0='+str(n)+'$ & '+' & '.join(f3(v[c]) for v in (v2,v3) for c in ('mean','se','bias','rmse'))+r' \\')
    write('tab_orig_body.tex',rows)
    rows=[]
    for day in (0,50,100,200,500,1000,1500,2000):
        cells=[str(day)]
        for gi in (1,2,3):
            r=lookup_dict(ST,group=gi,day=day);cells+=[f1(r.mean_N),f1(r.mean_P)]
        rows.append(cells)
    rows.append(r'\midrule');cells=['Expected']
    for gi in (1,2,3):
        r=lookup_dict(ST,group=gi,day=0);cells+=[f1(r.expected_N),f1(r.expected_P)]
    rows.append(cells)
    write('tab_supp_origstat_body.tex',rows)
    # Interval limits and bound results; scenarios kept in original specified order.
    byID={int(v.id):v for v in IL.itertuples(index=False)}
    def ie(id,n,e):return lookup_dict(IE,id=id,n=n,estimator=e)
    def ib(id,n,m,delta):return lookup_dict(IB,id=id,n=n,method=m,delta=delta)
    def ident(x):return [lab1(x.set,x.shape1),LABEL[x.shape0],f2(x.rho),str(x.idrr)]
    def cov(id,n,m,delta):
        x=ib(id,n,m,delta);a=pc(x.coverage)
        return a if bool(x.valid) else r'\textit{'+a+'}'
    def bound(x):return f2(x) if np.isfinite(x) else r'$\infty$'
    rows=[]
    for j,id in enumerate(ORD):
        x=byID[id]
        if j and x.set!=byID[ORD[j-1]].set:rows.append(r'\midrule')
        a=ie(id,5000,'EOR90');b=ie(id,5000,'EOR90i')
        cells=ident(x)+[f3(a.gmr),pc(a.coverage),f3(b.gmr),pc(b.coverage),f1(1000*max(x.d1_30,x.d0_30))]
        for delta in (.005,.01):
            z=ib(id,5000,'Exact30',delta);cells+=[cov(id,5000,'Exact30',delta),f2(z.med_lower),f2(z.med_upper)]
        rows.append(cells)
    write('tab_interval_body.tex',rows)
    rows=[];xil=IL[IL.idrr==2]
    for j,x in enumerate(xil.itertuples(index=False)):
        if j and x.set!=xil.iloc[j-1]['set']:rows.append(r'\midrule')
        rows.append([lab1(x.set,x.shape1),LABEL[x.shape0],f2(x.rho)]+[f3(getattr(x,k)) for k in ('ROR30','EOR90','EOR90i','EOR60i','EOR91w','TW60','TW90')]+[f1(1000*getattr(x,k)) for k in ('d1_30','d0_30','d1_90','d0_90')])
    write('tab_supp_intlim_body.tex',rows)
    stlabel={'ROR30':'ROR(30)','EOR90':'EOR(90), exact','EOR90i':'EOR(90), intervals','EOR60i':'EOR(60), intervals','TW60':'TW(60)','TW90':'TW(90)'}
    rows=[]
    for id in ORD:
        x=byID[id]
        for n in (200,1000,5000):
            for e in stlabel:
                r=ie(id,n,e);first=n==200 and e=='ROR30'
                rows.append((ident(x) if first else ['']*4)+[fmtid(n) if e=='ROR30' else '',stlabel[e],f3(r.limit),f3(r.gmr),f3(r.emp_se),f3(r.mean_se),f3(r.rmse),pc(r.coverage),pc(r.undefined)])
    write('tab_supp_intest_body.tex',rows)
    short={'constant':'Const.','uniform':'Unif.','exponential':'Exp.','lognormal':'Log-n.'}
    mt={'Wald30':'30, Wald','Exact30':'30, exact','Exact90':'90, exact','ExactAll':'30--90, exact'}
    rows=[]
    for id in ORD:
        x=byID[id]
        for n in (200,1000,5000):
            for m in mt:
                first=n==200 and m=='Wald30'
                z=ib(id,n,m,.01)
                cells=([short[x.shape1]+(r'$^{+}$' if x.set=='C' else '')+' / '+short[x.shape0],f2(x.rho),str(x.idrr)] if first else ['']*3)+[fmtid(n) if m=='Wald30' else '',mt[m],f1(1000*z.delta_min)]
                for dd in (.005,.01,.02):
                    a=ib(id,n,m,dd);cells +=[cov(id,n,m,dd),bound(a.med_lower),bound(a.med_upper),pc(a.lower_gt1+a.upper_lt1)]
                rows.append(cells)
    write('tab_supp_intbounds_body.tex',rows)
    print('Python-only LaTeX table files generated:',len(list(TEX.glob('tab_*_body.tex'))),flush=True)
if __name__=='__main__':main()
