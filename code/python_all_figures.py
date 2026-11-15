#!/usr/bin/env python3
"""Generate main and supplemental figures from the authoritative Python CSVs.

Reproduces the scientific panels of the original R plotting script. All axes,
comparators, and analytic reference curves are retained; graphical styling is
updated for readability. The simulated vignette in Figure 2 uses a fresh fixed
Python RNG seed, not the historical R random stream.
"""
from pathlib import Path
import math
import numpy as np,pandas as pd
from scipy import special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from python_all_tables import ROOT,OUTPUTS,limit,rmean,sv,D0,SH,LABEL,sub,get
F=ROOT/'figures';F.mkdir(exist_ok=True)
S=pd.read_csv(OUTPUTS/'main_summary_python.csv');T=pd.read_csv(OUTPUTS/'main_tests_python.csv')
J=pd.read_csv(OUTPUTS/'trajectories_python.csv');O=pd.read_csv(OUTPUTS/'original_summary_python.csv')
C={'POR':'black','aPOR':'#0072B2','ROR30':'#D55E00','ROR60':'#009E73','ROR90':'#CC79A7','EOR90':'#009E73','EOR180':'#CC79A7','EORInf':'#7A5195'}
ST={'POR':'-','aPOR':'--','ROR30':':','ROR60':'-.','ROR90':(0,(5,1,1,1)),'EOR90':'-.','EOR180':(0,(2,2)),'EORInf':(0,(7,2))}
M={'POR':'o','aPOR':'^','ROR30':'s','ROR60':'D','ROR90':'v','EOR90':'d','EOR180':'P','EORInf':'X'}
LB={'POR':'POR','aPOR':'Duration-adjusted POR','ROR30':'ROR(30)','ROR60':'ROR(60)','ROR90':'ROR(90)','EOR90':'EOR(90)','EOR180':'EOR(180)','EORInf':'EOR(all cases)'}
MAIN=('POR','aPOR','ROR30','EOR90','EORInf')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})

def save(name,fig):
    fig.savefig(F/name,bbox_inches='tight',pad_inches=.11)
    plt.close(fig);print('figure',name,flush=True)
def fine_survey(shape,seed):
    rng=np.random.default_rng(seed);n=20000
    vals=[]
    for I,D in [(.002,108),(.001,180)]:
        hm=n/(1+I*D);p=round(rng.poisson(hm*I*D));h=int(rng.poisson(hm))
        u=rng.random(p)
        if shape=='exponential':ages=rng.exponential(D,p)
        else:
            dur=u*np.sqrt((.5*D)**2+rng.random(p)*((1.5*D)**2-(.5*D)**2))
            ages=dur
        vals.append((h,ages))
    return vals

def logistic_fit(a,e):
    sc=a.mean();x=a/sc;theta=np.array([np.log(e.sum()/(len(e)-e.sum())),0.])
    for it in range(100):
        pp=special.expit(theta[0]+theta[1]*x);wt=pp*(1-pp)
        g=np.array([(e-pp).sum(),((e-pp)*x).sum()]);H=np.array([[wt.sum(),(wt*x).sum()],[(wt*x).sum(),(wt*x*x).sum()]])
        try:d=np.linalg.solve(H,g)
        except np.linalg.LinAlgError:break
        theta+=d
        if np.max(abs(d))<1.e-10:break
    return theta[0],theta[1]/sc

def fig2():
    aa=np.arange(0,360.5,.5)
    fig,axes=plt.subplots(2,2,figsize=(7.9,6.1),layout='constrained')
    x=axes[0,0]
    for k,sh in enumerate(SH):
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=sv(aa,sh,108)/sv(aa,sh,180)
        x.plot(aa,np.clip(ratio,.1,30),label=LABEL[sh],ls=['-','--',':','-.'][k],lw=1.5)
    x.set(title='(A) Same duration family, ratio 0.6',xlabel='Time since onset (days)',ylabel='POR(a) / IDRR',yscale='log',ylim=(.1,30));x.legend(frameon=False,loc='upper left')
    x=axes[0,1]
    from python_main_reproduction import mean_dur
    from scipy import optimize
    b=optimize.brentq(lambda beta:mean_dur('uniform',180,beta)-.6*180,1e-9,1.)
    for k,(sh1,d1,sh0,be,tag) in enumerate([('exponential',180,'constant',0,'Exponential / constant'),('constant',180,'exponential',0,'Constant / exponential'),('lognormal',180,'uniform',0,'Log-normal / uniform'),('uniform',180,'uniform',b,'Uniform, excess hazard')]):
        with np.errstate(divide='ignore',invalid='ignore'):
            yy=sv(aa,sh1,d1,be)/sv(aa,sh0,180)
        x.plot(aa,np.clip(yy,.1,30),label=tag,ls=['-','--',':','-.'][k],lw=1.5)
    x.set(title='(B) Different families / excess exits',xlabel='Time since onset (days)',ylabel='POR(a) / IDRR',yscale='log',ylim=(.1,30));x.legend(frameon=False,loc='upper left')
    for j,sh in enumerate(('exponential','uniform')):
        x=axes[1,j];(h1,a1),(h0,a0)=fine_survey(sh,2026111200+10+j)
        bins=np.arange(0,361,30);c1,_=np.histogram(a1,bins);c0,_=np.histogram(a0,bins);mid=(bins[:-1]+bins[1:])/2
        k=(c1>0)&(c0>0);r=(c1/h1)/(np.maximum(c0,1)/h0)
        se=np.sqrt(1/np.maximum(c1,1)+1/np.maximum(c0,1)+1/h1+1/h0)
        x.errorbar(mid[k],r[k],yerr=[r[k]-r[k]*np.exp(-1.96*se[k]),r[k]*np.exp(1.96*se[k])-r[k]],fmt='o',color='black',ms=3,capsize=0,label='30-day bin OR (95% CI)')
        with np.errstate(divide='ignore',invalid='ignore'):
            tr=2*sv(aa,sh,108)/sv(aa,sh,180)
        x.plot(aa,np.clip(tr,.1,10),c='0.5',lw=2.3,label='True POR(a)')
        a=np.r_[a1,a0];e=np.r_[np.ones(len(a1)),np.zeros(len(a0))]
        th,sl=logistic_fit(a,e);ix=a<=90;tw,sw=logistic_fit(a[ix],e[ix]);ratio=h0/h1
        x.plot(aa,np.exp(np.clip(th+sl*aa,-10,10))*ratio,ls='--',c=C['EORInf'],label='Fit, all cases')
        x.plot(aa[aa<=90],np.exp(np.clip(tw+sw*aa[aa<=90],-10,10))*ratio,ls='-',c=C['EOR90'],label='Fit, 90 days')
        x.axhline((len(a1)/h1)/(len(a0)/h0),ls=':',color='black',label='POR')
        x.axhline(2,ls=(0,(2,3)),color='gray',label='IDRR')
        x.set(title=f'({chr(67+j)}) Simulated survey, {LABEL[sh].lower()}',xlabel='Time since onset (days)',ylabel='Prevalence odds ratio',yscale='log',ylim=(.1,10),xlim=(0,360))
        if j==0:x.legend(frameon=False,fontsize=6)
    save('fig2_por_by_onset.pdf',fig)

def fig3():
    fig,ax=plt.subplots(2,2,figsize=(7.7,5.7),layout='constrained')
    for j,(x,sh) in enumerate(zip(ax.flat,SH)):
        d=sub(J,shape=sh).sort_values('t');t=d.t
        x.plot(t,d.POR_theory,lw=1.5,color='black',label='POR theory')
        x.plot(t,d.ROR_theory,lw=1.5,color=C['ROR30'],ls='--',label='ROR(30) theory')
        x.scatter(t[::3],d.POR[::3],color='black',s=6,label='POR simulated')
        x.scatter(t[::3],d.ROR[::3],facecolors='none',edgecolors=C['ROR30'],s=10,label='ROR(30) simulated')
        for y in (1.2,2.):x.axhline(y,color='gray',ls=':',lw=.65)
        x.set(title=f'({chr(65+j)}) {LABEL[sh]} duration',xlabel='Days since disease process began',ylabel='Odds ratio',ylim=(1,2.25))
        if j==0:x.legend(fontsize=6,frameon=False)
    save('fig3_trajectories.pdf',fig)

def fig4():
    fig,ax=plt.subplots(2,2,figsize=(7.7,5.8),layout='constrained');rhos=(.6,.8,1.,1.25);rr=np.exp(np.linspace(np.log(.55),np.log(1.3),55))
    for j,(x,sh) in enumerate(zip(ax.flat,SH)):
        for est in MAIN:
            ys=[limit(est,'A',sh,sh,float(r)) for r in rr]
            obs=[get(S,'gmean_est',set='A',shape1=sh,rho=float(r),idrr=2,n=5000,estimator=est)/2 for r in rhos]
            x.plot(rr,ys,color=C[est],ls=ST[est],lw=1.35,label=LB[est]);x.scatter(rhos,obs,c=C[est],marker=M[est],s=15)
        x.axhline(1,color='gray',ls=':',lw=.65)
        x.set(xscale='log',yscale='log',ylim=(.5,2.6),xlim=(.55,1.3),xticks=list(rhos),title=f'({chr(65+j)}) {LABEL[sh]} duration',xlabel='Mean-duration ratio',ylabel='Geometric mean / IDRR')
        x.set_xticklabels([str(z) for z in rhos]);
        if j==2:x.legend(frameon=False,fontsize=6)
    save('fig4_bias.pdf',fig)

def fig5():
    fig,ax=plt.subplots(2,2,figsize=(7.7,5.7),layout='constrained');ns=[200,500,1000,2000,5000]
    for j,(measure,rho) in enumerate([('rmse_log',1),('rmse_log',.6),('coverage',1),('coverage',.6)]):
        x=ax.flat[j]
        for est in MAIN:
            y=[get(S,measure,set='A',shape1='exponential',shape0='exponential',rho=rho,idrr=2,n=n,estimator=est) for n in ns]
            x.plot(ns,y,ls=ST[est],color=C[est],marker=M[est],markersize=3,label=LB[est])
        x.set_xscale('log');x.set_xticks(ns,labels=[str(n) for n in ns]);x.set(title=f'({chr(65+j)}) Mean duration ratio = {rho:g}',xlabel='Expected population per group',ylabel='RMSE log estimate' if measure=='rmse_log' else '95% CI coverage')
        if measure=='rmse_log':x.set_yscale('log');x.set_ylim(.03,1.)
        else:x.set_ylim(0,1);x.axhline(.95,color='gray',ls=':',lw=.75)
        if j==0:x.legend(frameon=False,fontsize=6)
    save('fig5_precision.pdf',fig)

def fig6():
    fig,axes=plt.subplots(2,2,figsize=(7.7,5.9),layout='constrained');ns=[200,500,1000,2000,5000]
    for j,(x,col,title) in enumerate(zip(axes.flat,('rej_wilcoxon','rej_ks','rej_recent30','rej_recent90'),('Wilcoxon test','Kolmogorov-Smirnov test','Recent fraction, 30 days','Recent fraction, 90 days'))):
        for k,rho in enumerate((1.,.8,.6,1.25)):
            y=[get(T,col,set='A',shape1='exponential',shape0='exponential',rho=rho,idrr=2,n=n) for n in ns]
            x.plot(ns,y,ls=['-','--',':','-.'][k],marker=['o','^','s','D'][k],ms=3,linewidth=1.2,label=f'{rho:g}',color='black')
        x.axhline(.05,color='gray',lw=.7,ls=':');x.set_xscale('log');x.set_xticks(ns,labels=[str(n) for n in ns]);x.set(title=f'({chr(65+j)}) {title}',xlabel='Expected population per group',ylabel='Rejection rate',ylim=(0,1))
        if j==2:x.legend(frameon=False,title='Duration ratio')
    save('fig6_test.pdf',fig)

def orig_fig():
    for what in ('bias','rmse'):
        fig,axes=plt.subplots(1,2,figsize=(7.5,3.3),layout='constrained')
        for k,(x,cmp) in enumerate(zip(axes,('1-2','1-3'))):
            for j,sh in enumerate(('constant','uniform','exponential')):
                a=sub(O,comparison=cmp,shape=sh).sort_values('N0')
                x.plot(a.N0,a[what],marker=['o','^','s'][j],ls=['-','--',':'][j],label=LABEL[sh])
            x.axhline(0,color='gray',lw=.8,ls=':')
            x.set(title='Identical mean duration' if k==0 else 'Different mean duration',xlabel='Population size',ylabel='Bias of POR' if what=='bias' else 'RMSE of POR',xscale='log',xticks=[100,200,500,1000])
            x.set_xticklabels(['100','200','500','1000']);
            if k==0:x.legend(frameon=False)
        save('fig_orig_'+what+'.pdf',fig)

def figS1():
    from python_all_tables import logistic_lim
    fig,axes=plt.subplots(2,2,figsize=(7.7,6.1),layout='constrained');ww=np.exp(np.linspace(np.log(5),np.log(720),35))
    for j,(x,e,rho) in enumerate(zip(axes.flat,('ROR','ROR','EOR','EOR'),(.6,.8,.6,.8))):
        for k,sh in enumerate(SH):
            if e=='ROR':ys=[rmean(w,sh,rho*D0)/rmean(w,sh,D0) for w in ww]
            else:ys=[logistic_lim(w,sh,rho*D0,0,sh,D0,2) for w in ww]
            x.plot(ww,ys,ls=['-','--',':','-.'][k],color='black',label=LABEL[sh])
        x.axhline(1,color='gray',ls=':',lw=.7)
        if e=='ROR':x.axhline(rho,color='gray',ls=':',lw=.7)
        x.set(xscale='log',yscale='log',xlabel='Window, days',ylabel=f'Limit of {e}(w) / IDRR',title=f'({chr(65+j)}) {"Recent onset" if e=="ROR" else "Extrapolated"}, duration ratio {rho:g}')
        if k==0:x.legend(frameon=False)
    save('figS1_window.pdf',fig)

def main():
    orig_fig();fig2();fig3();fig4();fig5();fig6();figS1()
    print('Generated every simulation-driven main/supplement figure')
if __name__=='__main__':main()
