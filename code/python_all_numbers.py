#!/usr/bin/env python3
"""Regenerate every text-number macro from independently seeded Python results.

Strictly errors if any previously used LaTeX macro lacks a Python formula. No
fallback to R-generated numeric values. All numerical prose must match current
Python outputs; scientifically unsupported threshold statements are audited
separately in python_claim_audit.py.
"""
from pathlib import Path
import re,math
import numpy as np,pandas as pd
from python_all_tables import ROOT,OUTPUTS,TEX,limit,rmean,SH,f1,f2,f3,pc,pc0,sub,get,lookup_dict
need=['main_summary','main_tests','validation','trajectories','immigration','growth','changing_incidence','interval_limits','interval_estimators','interval_bounds','original_summary','original_stationarity']
d={k:pd.read_csv(OUTPUTS/(k+'_python.csv')) for k in need}
S=d['main_summary'];T=d['main_tests'];V=d['validation'];J=d['trajectories'];M=d['immigration'];G=d['growth'];CH=d['changing_incidence'];IL=d['interval_limits'];IE=d['interval_estimators'];IB=d['interval_bounds'];OS=d['original_summary'];ST=d['original_stationarity']
A=sub(S,set='A');B=sub(S,set='B');C=sub(S,set='C');TA=sub(T,set='A');TB=sub(T,set='B');TC=sub(T,set='C')
orig=set(re.findall(r'\\newcommand\{\\(\w+)\}',(TEX/'numbers.tex').read_text()))
mm={}
def D(name,value):
    if not name.startswith('n'):name='n'+name
    if name in mm:raise RuntimeError(('Duplicate macro',name))
    mm[name]=str(value)
def val(df,field,**kw):
    x=sub(df,**kw)
    if len(x)!=1:raise ValueError(('scalar',field,kw,len(x)))
    return float(x.iloc[0][field])
def v(df,field,**kw):return sub(df,**kw)[field].to_numpy(float)
def rng(x,digits=1):
    x=np.asarray(x,float);fmt=lambda z:f'{100*z:.{digits}f}';lo,hi=fmt(np.nanmin(x)),fmt(np.nanmax(x))
    return lo if lo==hi else lo+'--'+hi
def up(x,n=1):return f'{math.ceil(float(x)*10**n-1e-9)/10**n:.{n}f}'
def big(x):return str(int(x)).replace('10000',r'10\,000')
def between(x):
    lo,hi=f2(np.nanmin(x)),f2(np.nanmax(x))
    return lo if lo==hi else 'between '+lo+' and '+hi

def base():
    D('ValRelErr',up(100*max(np.max(abs(V.EH_sim/V.EH_theory-1)),np.max(abs(V.EP_sim/V.EP_theory-1))),2))
    vm=np.r_[V.varH_sim/V.EH_sim,V.varP_sim/V.EP_sim]
    D('ValVMmin',f2(vm.min()));D('ValVMmax',f2(vm.max()))
    D('ValCorMin',f3(V.cor_HP.min()));D('ValCorMax',f3(V.cor_HP.max()))
    D('ValEArel',up(100*max(abs(V.EA_sim/V.EA_theory-1)),2));D('ValKSmax',f3(V.ks_stat.max()))
    D('TrajMaxErr',up(100*max(abs(J.POR/J.POR_theory-1)),1))
    for key,sh,tt in [('TrajExpA','exponential',360),('TrajLnA','lognormal',360),('TrajExpB','exponential',900),('TrajLnB','lognormal',900)]:D(key,f2(val(J,'POR',shape=sh,t=tt)))
    for tag,sh in [('con','constant'),('uni','uniform'),('exp','exponential'),('ln','lognormal')]:D('TrajROR'+tag,f2(sub(J,shape=sh).query('t>=30').ROR.mean()))
    p1=sub(A,rho=1,idrr=2,estimator='POR')
    for key,ex in [('PORgmMin',(p1.gmean_est/2).min()),('PORgmMax',(p1.gmean_est/2).max())]:D(key,f3(ex))
    for key,col,n,fn,fmt in [('meanSmallMin','mean_est',200,min,f2),('meanSmallMax','mean_est',200,max,f2),('medSmallMin','median_est',200,min,f2),('medSmallMax','median_est',200,max,f2),('meanLargeMin','mean_est',5000,min,f3),('meanLargeMax','mean_est',5000,max,f3)]:D('POR'+key,fmt(fn(v(p1,col,n=n))))
    D('PORseSmall',f2(np.mean(v(p1,'emp_se',n=200))));D('PORseLarge',f3(np.mean(v(p1,'emp_se',n=5000))))
    D('PORcovMin',pc(p1.coverage.min()));D('PORcovMax',pc(p1.coverage.max()))
    D('PORseDiff',f1(100*max(np.max(z)/np.min(z)-1 for _,z in p1.groupby('n').emp_se)))
    p1n=sub(A,rho=1,idrr=1,estimator='POR');D('PORnullMin',pc(p1n.reject.min()));D('PORnullMax',pc(p1n.reject.max()))
    p6=sub(A,rho=.6,idrr=2,estimator='POR')
    D('PORsixGMmin',f3((p6.gmean_est/2).min()));D('PORsixGMmax',f3((p6.gmean_est/2).max()))
    D('PORsixCovSmall',rng(v(p6,'coverage',n=200)));D('PORsixCovK',rng(v(p6,'coverage',n=1000)))
    D('PORsixCovKMax',pc(v(p6,'coverage',n=1000).max()));D('PORsixRmseLarge',f2(v(p6,'rmse_log',n=5000).mean()))
    for rho,tag in [(.8,'Eight'),(1.25,'OneTwoFive')]:
        q=sub(A,rho=rho,idrr=2,estimator='POR',n=5000)
        D('POR'+tag+'GM',f2((q.gmean_est/2).mean()))
        D('POR'+tag+'CovMin',pc(q.coverage.min()));D('POR'+tag+'CovMax',pc(q.coverage.max()))
    for tag,rho,n in [('NullSixSmall',.6,200),('NullSixK',.6,1000),('NullEightK',.8,1000),('NullEightLarge',.8,5000)]:D(tag,rng(v(A,'reject',rho=rho,idrr=1,estimator='POR',n=n)))
    adj=sub(A,estimator='aPOR');a5=adj[adj.n>=500];a2=sub(adj,n=200)
    for key,z in [('gm',a5),('gmSmall',a2)]:
        D('APOR'+key+'Min',f3((z.gmean_est/z.idrr).min()));D('APOR'+key+'Max',f3((z.gmean_est/z.idrr).max()))
    for key,z in [('cov',a5),('covSmall',a2)]:D('APOR'+key+'Min',pc(z.coverage.min()));D('APOR'+key+'Max',pc(z.coverage.max()))
    ratios=adj[['id','n','shape1','emp_se']].merge(sub(A,estimator='POR')[['id','n','emp_se']],on=['id','n'],suffixes=('_a','_p'))
    for tag,sh in [('Con','constant'),('Uni','uniform'),('Exp','exponential'),('Ln','lognormal')]:
        q=ratios[(ratios.shape1==sh)&(ratios.n>=500)];D('APORseRatio'+tag,f2((q.emp_se_a/q.emp_se_p).mean()))
    nul=adj[(adj.idrr==1)&(adj.n>=500)]
    D('APORnullMin',pc(nul.reject.min()));D('APORnullMax',pc(nul.reject.max()))
    D('APORnullSmallMax',pc(v(adj,'reject',idrr=1,n=200).max()))
    for w,tag in [(30,'Thirty'),(60,'Sixty'),(90,'Ninety')]:
        r=sub(A,estimator=f'ROR{w}');r5=r[r.n>=500]
        D('ROR'+tag+'gmMin',f3((r5.gmean_est/r5.idrr).min()));D('ROR'+tag+'gmMax',f3((r5.gmean_est/r5.idrr).max()))
        D('ROR'+tag+'covMin',pc(r5.coverage.min()));D('ROR'+tag+'covMax',pc(r5.coverage.max()))
        x=r[['id','n','emp_se']].merge(sub(A,estimator='POR')[['id','n','emp_se']],on=['id','n'],suffixes=('_r','_p'))
        z=x[x.n>=500].emp_se_r/x[x.n>=500].emp_se_p
        D('ROR'+tag+'seRatioMin',f1(z.min()));D('ROR'+tag+'seRatioMax',f1(z.max()))
        rr=r[(r.idrr==1)&(r.n>=500)]
        D('ROR'+tag+'nullMin',pc(rr.reject.min()));D('ROR'+tag+'nullMax',pc(rr.reject.max()))
        D('ROR'+tag+'undefSmall',pc(v(r,'undefined',n=200).max()));D('ROR'+tag+'undefK',f2(100*r[r.n>=500].undefined.max()))
        D('ROR'+tag+'covSmallMin',pc(v(r,'coverage',n=200).min()));D('ROR'+tag+'covSmallMax',pc(v(r,'coverage',n=200).max()))
        D('ROR'+tag+'gmSmallMin',f3((sub(r,n=200).gmean_est/sub(r,n=200).idrr).min()));D('ROR'+tag+'gmSmallMax',f3((sub(r,n=200).gmean_est/sub(r,n=200).idrr).max()))
    for n,tag in [(200,'Small'),(1000,'K'),(5000,'Large')]:
        for e in ('POR','aPOR','ROR30','ROR90','EOR90','EORInf'):
            name=e.replace('30','Thirty').replace('90','Ninety')
            D('Rmse'+name+tag,f2(val(A,'rmse_log',rho=.6,idrr=2,shape1='exponential',n=n,estimator=e)))
    D('RORNinetyCovLargeMin',pc(v(A,'coverage',estimator='ROR90',n=5000,rho=.6,idrr=2).min()))

def tests():
    w1=sub(TA,rho=1)
    D('Wsize',rng(w1.rej_wilcoxon));D('KSsize',rng(w1.rej_ks))
    D('RFsize',rng(w1[w1.n>=500].rej_recent30));D('RFsizeSmall',rng(w1[w1.n==200].rej_recent30))
    for name,col,rho,n in [('WpowSixSmall','rej_wilcoxon',.6,200),('WpowSixK','rej_wilcoxon',.6,1000),('WpowEightK','rej_wilcoxon',.8,1000),('WpowEightLarge','rej_wilcoxon',.8,5000),('WpowOneTwoFiveK','rej_wilcoxon',1.25,1000),('WpowOneTwoFiveLarge','rej_wilcoxon',1.25,5000),('KSpowSixK','rej_ks',.6,1000),('KSpowEightLarge','rej_ks',.8,5000),('RFpowSixK','rej_recent30',.6,1000),('RFpowSixLarge','rej_recent30',.6,5000),('RFpowEightLarge','rej_recent30',.8,5000)]:D(name,rng(v(TA,col,rho=rho,n=n,idrr=2),0))
    D('ShapeAmax',pc(TA.rej_shape.max()));D('RFpowSixLargeMin',pc(v(TA,'rej_recent30',rho=.6,n=5000,idrr=2).min()))
    D('WconEightK',pc0(val(TA,'rej_wilcoxon',shape1='constant',rho=.8,n=1000,idrr=2)))
    D('KSconEightK',pc0(val(TA,'rej_ks',shape1='constant',rho=.8,n=1000,idrr=2)))
    def bc(s1,s0,rho,e,col,n=2000,idrr=2):return val(B,col,shape1=s1,shape0=s0,rho=rho,estimator=e,n=n,idrr=idrr)
    for name,s1,s0 in [('BaporEC','exponential','constant'),('BaporCE','constant','exponential'),('BaporLU','lognormal','uniform')]:D(name,f2(bc(s1,s0,1,'aPOR','gmean_est')/2))
    D('BaporECcov',pc(bc('exponential','constant',1,'aPOR','coverage')))
    r=sub(B,estimator='ROR30');D('BrorMin',f2((r.gmean_est/r.idrr).min()));D('BrorMax',f2((r.gmean_est/r.idrr).max()))
    p=sub(B,estimator='POR');p=p[(p.n>=500)&np.isclose(p.rho,1)]
    D('BporMin',f3((p.gmean_est/p.idrr).min()));D('BporMax',f3((p.gmean_est/p.idrr).max()))
    def bt(rho,col,s1=None,n=2000):
        z=sub(TB,rho=rho,n=n,idrr=2)
        if s1:z=sub(z,shape1=s1)
        return z[col].to_numpy(float)
    D('BrfOne',rng(bt(1,'rej_recent30'),1));D('BrfOneAllMax',pc0(sub(TB,rho=1).rej_recent30.max()))
    D('BrfOneLarge',rng(bt(1,'rej_recent30',n=5000),1));D('BrfNinetyOne',rng(bt(1,'rej_recent90'),0))
    prs=[('exponential','constant'),('constant','exponential'),('lognormal','uniform')]
    for w,tag in [(30,'Thirty'),(90,'Ninety')]:
        z=[rmean(w,s1,180)/rmean(w,s0,180) for s1,s0 in prs]
        D('Brr'+tag+'Min',f2(min(z)));D('Brr'+tag+'Max',f2(max(z)))
    D('BrfSix',rng(bt(.6,'rej_recent30'),0))
    D('BwilcSixEC',pc(bt(.6,'rej_wilcoxon','exponential')[0]));D('BwilcSixLU',pc(bt(.6,'rej_wilcoxon','lognormal')[0]))
    D('BshapeK',rng(sub(TB,n=1000).rej_shape,0));D('BshapeSmall',rng(sub(TB,n=200).rej_shape,0))
    rngg=np.random.default_rng(2026111201)
    a1=rngg.exponential(108,1_000_000);a0=rngg.uniform(0,180,1_000_000)
    q1=f2(np.mean(a1>a0))
    a1=rngg.random(1_000_000)*rngg.lognormal(math.log(108)+.5,1.,1_000_000)
    u=rngg.random(1_000_000)*np.sqrt(90**2+rngg.random(1_000_000)*(270**2-90**2))
    q2=f2(np.mean(a1>u))
    D('Bpgt',q1+' in both' if q1==q2 else q1+' and '+q2)

def extrapolation():
    porSE=sub(A,estimator='POR')[['id','n','emp_se']]
    ei=sub(A,estimator='EORInf');ex=ei[(ei.shape1=='exponential')&(ei.n>=500)]
    D('EORinfExpGMmin',f3((ex.gmean_est/ex.idrr).min()));D('EORinfExpGMmax',f3((ex.gmean_est/ex.idrr).max()))
    D('EORinfExpCovMin',pc(ex.coverage.min()));D('EORinfExpCovMax',pc(ex.coverage.max()))
    sri=ei[['id','n','shape1','emp_se']].merge(porSE,on=['id','n'],suffixes=('_e','_p'))
    D('EORinfExpSeRatio',f2((sri[(sri.shape1=='exponential')&(sri.n>=500)].emp_se_e/sri[(sri.shape1=='exponential')&(sri.n>=500)].emp_se_p).mean()))
    def g6(sh,e,c):return val(A,c,shape1=sh,rho=.6,idrr=2,n=5000,estimator=e)
    for sh,tag in [('constant','Con'),('uniform','Uni'),('lognormal','Ln')]:D('EORinf'+tag+'Six',f2(g6(sh,'EORInf','gmean_est')/2))
    D('EORinfMisCovMax',pc(max(g6(sh,'EORInf','coverage') for sh in ('constant','uniform','lognormal'))))
    e9=sub(A,estimator='EOR90');e9k=e9[e9.n>=1000]
    D('EORninetyGMmin',f3((e9k.gmean_est/e9k.idrr).min()));D('EORninetyGMmax',f3((e9k.gmean_est/e9k.idrr).max()))
    D('EORninetyCovMin',pc(e9k.coverage.min()));D('EORninetyCovMax',pc(e9k.coverage.max()))
    j=e9[['id','n','emp_se']].merge(porSE,on=['id','n'],suffixes=('_e','_p'))
    ratios=j[j.n>=500].emp_se_e/j[j.n>=500].emp_se_p
    D('EORninetySeRatioMin',f1(ratios.min()));D('EORninetySeRatioMax',f1(ratios.max()))
    z=e9[['id','n','emp_se']].merge(sub(A,estimator='ROR30')[['id','n','emp_se']],on=['id','n'],suffixes=('_e','_r'))
    ratios=z[z.n>=500].emp_se_e/z[z.n>=500].emp_se_r
    D('EORninetyVsRORmin',f2(ratios.min()));D('EORninetyVsRORmax',f2(ratios.max()))
    e2=e9[e9.n==200]
    D('EORninetyGMsmallMin',f2((e2.gmean_est/e2.idrr).min()));D('EORninetyGMsmallMax',f2((e2.gmean_est/e2.idrr).max()))
    D('EORninetySeSmallMin',f2(e2.emp_se.min()));D('EORninetySeSmallMax',f2(e2.emp_se.max()))
    D('CovRORexpLarge',pc(g6('exponential','ROR30','coverage')));D('CovEORexpLarge',pc(g6('exponential','EOR90','coverage')))
    en=e9[(e9.idrr==1)&(e9.n>=500)];D('EORninetyNullMin',pc(en.reject.min()));D('EORninetyNullMax',pc(en.reject.max()))
    D('AbsEORninetyErr',f1(100*max(abs(e9k.gmean_est/e9k.idrr-1))))
    lin_true=pd.concat([TA[(TA.shape1=='exponential')|np.isclose(TA.rho,1)],TC])
    D('LinSize',rng(lin_true[lin_true.n>=500].rej_linear))
    for s,tag,n in [('constant','LinPowConK',1000),('uniform','LinPowUniK',1000),('lognormal','LinPowLnK',1000),('lognormal','LinPowLnLarge',5000)]:
        D(tag,pc0(val(TA,'rej_linear',shape1=s,rho=.6,idrr=2,n=n)))
    D('CshapeK',rng(sub(TC,n=1000).rej_shape,0));D('CshapeLarge',rng(sub(TC,n=5000).rej_shape,0))
    bi=sub(B,estimator='EORInf',n=2000,idrr=2)
    D('BeorInfMin',f2((bi.gmean_est/2).min()));D('BeorInfMax',f2((bi.gmean_est/2).max()))
    b9=sub(B,estimator='EOR90');b9=b9[b9.n>=1000]
    D('BeorNinetyMin',f2((b9.gmean_est/b9.idrr).min()));D('BeorNinetyMax',f2((b9.gmean_est/b9.idrr).max()))
    D('AbsEORninetyErrB',f1(100*max(abs(b9.gmean_est/b9.idrr-1))))
    k9=sub(S,estimator='EOR90');k9=k9[k9.n>=1000]
    k3=sub(S,estimator='ROR30');k3=k3[k3.n>=1000]
    D('AbsEORninetyErrAll',f1(100*max(abs(k9.gmean_est/k9.idrr-1))))
    D('AbsRORdiffErrK',f1(100*max(abs(k3.gmean_est/k3.idrr-1))))
    ca=sub(C,estimator='aPOR',n=2000,idrr=2)
    D('CaporMin',f2((ca.gmean_est/2).min()));D('CaporMax',f2((ca.gmean_est/2).max()))
    ci=sub(C,estimator='EORInf');ci=ci[ci.n>=500]
    D('CeorInfMin',f3((ci.gmean_est/ci.idrr).min()));D('CeorInfMax',f3((ci.gmean_est/ci.idrr).max()))
    D('CeorInfCovMin',pc(ci.coverage.min()));D('CeorInfCovMax',pc(ci.coverage.max()))
    c9=sub(C,estimator='EOR90');c9=c9[c9.n>=500]
    D('CeorNinetyMin',f3((c9.gmean_est/c9.idrr).min()));D('CeorNinetyMax',f3((c9.gmean_est/c9.idrr).max()))
    sr=ci[['id','n','emp_se']].merge(sub(C,estimator='POR')[['id','n','emp_se']],on=['id','n'],suffixes=('_e','_p'))
    D('CeorInfSeRatioMin',f2((sr.emp_se_e/sr.emp_se_p).min()));D('CeorInfSeRatioMax',f2((sr.emp_se_e/sr.emp_se_p).max()))
    cr=sub(C,estimator='ROR30');cr=cr[cr.n>=1000]
    D('CrorMaxErr',f1(100*max(abs(cr.gmean_est/cr.idrr-1))))
    cL=C[C.n==5000];D('CrorCovLargeMin',pc(sub(cL,estimator='ROR30').coverage.min()))
    D('CeorNinetyCovLargeMin',pc(sub(cL,estimator='EOR90').coverage.min()));D('CeorNinetyCovLargeMax',pc(sub(cL,estimator='EOR90').coverage.max()))
    bK=B[(B.n==2000)&(B.idrr==2)]
    D('BrorCovMin',pc(sub(bK,estimator='ROR30').coverage.min()));D('BeorNinetyCovMin',pc(sub(bK,estimator='EOR90').coverage.min()))
    D('CshapeLargeMin',pc(sub(TC,n=5000).rej_shape.min()))
    D('CuniShape',pc(val(TC,'rej_shape',shape1='uniform',rho=.6,idrr=2,n=2000)))
    D('CuniAporCov',pc(val(C,'coverage',shape1='uniform',rho=.6,idrr=2,n=2000,estimator='aPOR')))
    D('CuniAporGM',f2(val(C,'gmean_est',shape1='uniform',rho=.6,idrr=2,n=2000,estimator='aPOR')/2))
    D('LinUndefSmall',pc0(sub(T,n=200).undef_linear.max()))

def theory_and_limits():
    prsB=[('exponential','constant'),('constant','exponential'),('lognormal','uniform')]
    configs=[('A',s,s,rho) for s in SH for rho in (1.,.8,.6,1.25)]+ [('C',s,s,rho) for s in ('constant','uniform','lognormal') for rho in (.8,.6)] + [('B',s1,s0,rho) for s1,s0 in prsB for rho in (1.,.6)]
    lr=np.array([limit('ROR30',ss,s1,s0,rho) for ss,s1,s0,rho in configs])
    l9=np.array([limit('EOR90',ss,s1,s0,rho) for ss,s1,s0,rho in configs]);D('LimCfg',len(configs))
    D('LimRORwithin',int(np.sum(abs(lr-1)<.02)));D('LimEORwithin',int(np.sum(abs(l9-1)<.02)))
    D('LimRORmaxErr',pc0(max(abs(lr-1))));D('LimEORmaxErr',pc0(max(abs(l9-1))))
    bigs=S[S.n==5000].copy()
    vals=[]
    for z in bigs.itertuples(index=False):
        est=limit(z.estimator,z.set,z.shape1,z.shape0,z.rho,z.idrr)
        vals.append(abs(z.gmean_est/z.idrr/est-1))
    D('LimitMaxErr',up(100*max(vals),1))

def departures():
    gg=lambda e,s=None: sub(G,estimator=e) if s is None else sub(G,estimator=e,shape=s)
    D('GrowPORmin',f2(gg('POR').gmr.min()));D('GrowPORmax',f2(gg('POR').gmr.max()))
    D('GrowPORmaxErr',pc0(1-gg('POR').gmr.min()))
    D('GrowAPORuniMin',f2(gg('aPOR','uniform').gmr.min()));D('GrowAPORuniMax',f2(gg('aPOR','uniform').gmr.max()))
    D('GrowRORmin',f2(gg('ROR30').gmr.min()));D('GrowRORmax',f2(gg('ROR30').gmr.max()))
    ge=G[G.estimator.isin(['EOR90','EORInf'])]
    D('GrowEORmin',f3(ge.gmr.min()));D('GrowEORmax',f3(ge.gmr.max()))
    D('GrowEORcovMin',pc(ge.coverage.min()));D('GrowEORcovMax',pc(ge.coverage.max()))
    ch=lambda cases,e,col='gmr':CH[CH.scenario.isin(cases) & CH.estimator.isin(e)][col].to_numpy(float)
    D('ChgReps',big(CH.reps.iloc[0]));D('ChgLimErr',up(100*max(abs(CH.gmr/CH.limit-1)),1))
    cnt=np.r_[CH.mean_H1/CH.EH1,CH.mean_H0/CH.EH0,CH.mean_P1/CH.EP1,CH.mean_P0/CH.EP0]
    D('ChgCountErr',up(100*max(abs(cnt-1)),1))
    for tag,scenario in [('rise','rise'),('fall','fall'),('step','step'),('both','both')]:D('ChgPOR'+tag,f2(ch([scenario],['POR'])[0]))
    D('ChgPORriseErr',pc0(1-ch(['rise'],['POR'])[0]));eqd=['rise','fall','both','step']
    D('ChgRORmin',f2(ch(eqd,['ROR30']).min()));D('ChgRORmax',f2(ch(eqd,['ROR30']).max()))
    D('ChgEORmin',f3(ch(CH.scenario,['EOR90']).min()));D('ChgEORmax',f3(ch(CH.scenario,['EOR90']).max()))
    D('ChgEORmaxErr',up(100*max(abs(ch(CH.scenario,['EOR90'])-1)),1))
    D('ChgEORcovMin',pc(ch(CH.scenario,['EOR90'],'coverage').min()));D('ChgEORcovMax',pc(ch(CH.scenario,['EOR90'],'coverage').max()))
    D('ChgAPORmin',f2(ch(CH.scenario,['aPOR']).min()));D('ChgAPORmax',f2(ch(CH.scenario,['aPOR']).max()))
    D('ChgEORInfMin',f2(ch(CH.scenario,['EORInf']).min()));D('ChgEORInfMax',f2(ch(CH.scenario,['EORInf']).max()))
    def im(rho,mrel,col):return val(M,col,rho=rho,mrel=mrel)
    D('ImmAllHalf',f2(im(1,.5,'por_all')));D('ImmAllOne',f2(im(1,1,'por_all')))
    D('ImmLocalRange',between(v(M,'por_local',rho=1)))
    D('ImmMaxErr',up(100*max(abs(M.por_all/M.por_all_theory-1)),1))

def intervals():
    def ie(id,n,e,c):return val(IE,c,id=id,n=n,estimator=e)
    def ib(id,n,m,d,c):return val(IB,c,id=id,n=n,method=m,delta=d)
    D('IntScen',IE.id.nunique())
    abc=sub(IL,idrr=2);abc=abc[abc.set!='D'];D('IntLimDiffMax',f1(100*max(abs(abc.EOR90i/abc.EOR90-1))))
    D('IntLimWeekMax',f1(100*max(abs(abc.EOR91w/abc.EOR90-1))))
    for id,tag in [(57,'Dfifty'),(58,'Dfifteen')]:
        D('IntLim'+tag,f3(val(IL,'EOR90i',id=id)))
    D('TWlimMin',f2(min(abc.TW60.min(),abc.TW90.min())));D('TWlimMax',f2(max(abc.TW60.max(),abc.TW90.max())))
    D('TWlimDfifty',f2(val(IL,'TW90',id=57)));D('TWlimDfifteen',f2(val(IL,'TW90',id=58)))
    within=lambda c:int(np.sum(abs(abc[c]-1)<=.02))
    for key,col in [('TWninetyWithin','TW90'),('TWsixtyWithin','TW60'),('IntEORsixtyWithin','EOR60i'),('IntEORninetyIWithin','EOR90i')]:D(key,within(col))
    x=sub(IE,estimator='EOR90');x=x[(x.set!='D')&(x.n>=1000)].merge(sub(IE,estimator='EOR90i'),on=['id','n'],suffixes=('.x','.i'))
    D('IntGMdiffMax',f3(max(abs(x['gmr.i']-x['gmr.x']))));D('IntCovDiffMax',f1(100*max(abs(x['coverage.i']-x['coverage.x']))))
    D('IntSeRatioMin',f2(min(x['emp_se.i']/x['emp_se.x'])));D('IntSeRatioMax',f2(max(x['emp_se.i']/x['emp_se.x'])))
    D('IntDfiftyGM',f3(ie(57,5000,'EOR90i','gmr')));D('IntDfifteenGM',f3(ie(58,5000,'EOR90i','gmr')))
    D('IntDfifteenCov',pc(ie(58,5000,'EOR90i','coverage')))
    D('IntDexactCovMin',pc(min(ie(57,5000,'EOR90','coverage'),ie(58,5000,'EOR90','coverage'))))
    tw=IE[IE.estimator.isin(['TW60','TW90'])];t9=sub(tw,estimator='TW90')
    D('TWninetyUndefSmallMin',f1(100*sub(t9,n=200).undefined.min()));D('TWninetyUndefSmallMax',f1(100*sub(t9,n=200).undefined.max()))
    D('TWundefKMax',f2(100*sub(tw,n=1000).undefined.max()))
    a=sub(t9,n=5000);a=a[a.set!='D'].merge(sub(IE,estimator='EOR90'),on=['id','n'],suffixes=('.x','.y'))
    D('TWninetySeRatioMin',f2((a['emp_se.x']/a['emp_se.y']).min()));D('TWninetySeRatioMax',f2((a['emp_se.x']/a['emp_se.y']).max()))
    D('IntEORundefSmallMax',f1(100*sub(IE,estimator='EOR90i',n=200).undefined.max()))
    b=sub(IE,estimator='TW60',n=5000);b=b[b.set!='D'].merge(sub(IE,estimator='EOR60i'),on=['id','n'],suffixes=('.x','.y'))
    D('TWseRatioMin',f2((b['emp_se.x']/b['emp_se.y']).min()));D('TWseRatioMax',f2((b['emp_se.x']/b['emp_se.y']).max()))
    bx=IB[(IB.method=='Exact30')&(IB.delta>0)];bw=IB[(IB.method=='Wald30')&(IB.delta>0)&(IB.n>=1000)];ba=IB[(IB.method=='ExactAll')&(IB.delta>0)]
    D('IntCovValidMin',pc(bx[bx.valid].coverage.min()));D('IntCovValidWaldMin',pc(bw[bw.valid].coverage.min()))
    D('IntCovValidAllMin',pc(ba[ba.valid].coverage.min()))
    D('IntUnbSmallMax',f1(100*sub(bx,n=200).unbounded.max()));D('IntUnbKMax',f1(100*sub(bx,n=1000).unbounded.max()))
    D('IntCovInvalidMin',pc(bx[(~bx.valid)&(bx.set!='D')].coverage.min()))
    D('IntCovDfiftyLarge',pc(ib(57,5000,'Exact30',.01,'coverage')))
    D('IntCovDfifteenLargeMax',pc(sub(bx,id=58,n=5000).coverage.max()))
    for delta,tag in [(0,'Zero'),(.005,'Five'),(.01,'One')]:
        D('IntExL'+tag,f2(2*ib(27,5000,'Exact30',delta,'med_lower')))
        D('IntExU'+tag,f2(2*ib(27,5000,'Exact30',delta,'med_upper')))
        if delta>0:
            for n,suf in [(1000,'K'),(5000,'Large')]:D('IntExExcl'+tag+suf,pc(ib(27,n,'Exact30',delta,'lower_gt1')))
    D('IntExCovZero',pc(ib(27,5000,'Exact30',0,'coverage')))
    nul=bx[(bx.id==11)&(bx.valid)]
    D('IntNullExclMax',pc(max(nul.lower_gt1+nul.upper_lt1)))

def original_and_final():
    D('OrigReps',big(OS.reps.iloc[0]))
    def comp(c,n,col):return v(OS,col,comparison=c,N0=n)
    for tag,col in [('Mean','mean'),('Rmse','rmse'),('Med','median')]:
        for n,suf in [(100,'Small'),(1000,'Large')]:
            zz=comp('1-2',n,col)
            D('OrigSame'+tag+suf+'Min',f2(zz.min()));D('OrigSame'+tag+suf+'Max',f2(zz.max()))
    for tag,col in [('Mean','mean'),('Rmse','rmse')]:
        zz=comp('1-3',1000,col)
        D('OrigDiff'+tag+'LargeMin',f2(zz.min()));D('OrigDiff'+tag+'LargeMax',f2(zz.max()))
    for tag,col,n in [('OrigSameMeanSmall','mean',100),('OrigSameRmseSmall','rmse',100),('OrigSameMedSmall','median',100),('OrigSameMeanLarge','mean',1000),('OrigSameRmseLarge','rmse',1000)]:D(tag,between(comp('1-2',n,col)))
    D('OrigDiffMeanLarge',between(comp('1-3',1000,'mean')));D('OrigDiffRmseLarge',between(comp('1-3',1000,'rmse')))
    st=ST[ST.day>=200]
    D('OrigStatNmin',f1(st.mean_N.min()));D('OrigStatNmax',f1(st.mean_N.max()))
    D('OrigStatPrel',up(100*max(abs(st.mean_P/st.expected_P-1)),1))
    D('OrigStatNfifty',f1(sub(ST,day=50).mean_N.max()))
    a=sub(A,estimator='ROR30');b=sub(B,estimator='ROR30')
    D('AbsRORsameErr',f1(100*max(abs(a.gmean_est/a.idrr-1))))
    D('AbsRORdiffErr',f1(100*max(abs(b.gmean_est/b.idrr-1))))
    D('APORcovAllMin',pc(sub(A,estimator='aPOR').coverage.min()));D('APORcovAllMax',pc(sub(A,estimator='aPOR').coverage.max()))
    D('Reps',big(S.reps.iloc[0]))

def main():
    for fn in [base,tests,extrapolation,theory_and_limits,departures,intervals,original_and_final]:
        fn();print('finished section',fn.__name__,len(mm),flush=True)
    # Reference/bookkeeping labels are not statistical results; retain source-generated
    # equation/table/figure numbers until the next pdflatex cross-reference pass.
    existing=(TEX/'numbers.tex').read_text()
    for name,value in re.findall(r'^\\newcommand\{\\(nRef\w+|nWordCount)\}\{(.*)\}$',existing,re.MULTILINE):
        if name not in mm:mm[name]=value
    missing=orig-set(mm);extra=set(mm)-orig
    if missing or extra:
        print('MISSING MACROS',sorted(missing),'EXTRA MACROS',sorted(extra))
        raise RuntimeError('Python macro definitions do not match original LaTeX schema')
    (TEX/'numbers.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in mm.items()))
    print('Regenerated',len(mm),'LaTeX numeric macros solely from Python outputs')
    # numbers of the two real-data illustrations (python_nordcan_application.py, python_colon_bounds.py)
    import json
    app={}
    for stem in ('nordcan_macros','colon_bounds_macros','main_individual_macros'):
        app.update(json.loads((OUTPUTS/(stem+'.json')).read_text()))
    assert all(re.fullmatch(r'n(App|Ind)[A-Za-z]+',k) for k in app)
    (TEX/'numbers_app.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in app.items()))
    print('Wrote',len(app),'macros of the real-data illustrations and of the individual-history comparison')
if __name__=='__main__':main()
