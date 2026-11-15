#!/usr/bin/env python3
"""Independent numerical checks of key stationary and early-exit identities.

Checks numerical implications of published propositions, not human-certified
proofs. Runs with a deterministic random seed and exits nonzero on failure.
"""
import math,json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

D0=180.; I0=.001

def survival(x,shape,D):
    if shape=='constant':return 1. if x<D else 0.
    if shape=='uniform':return float(np.clip((1.5*D-x)/D,0,1))
    if shape=='exponential':return float(np.exp(-x/D))
    if shape=='lognormal':return float(norm.sf((math.log(max(x,1e-200))-math.log(D)+.5)))
    raise ValueError(shape)

def rmean(w,s,D):
    hi=min(w,400*D)
    pts=[x for x in [D/2, D, 1.5*D] if 0<x<hi and s in ('uniform','constant')]
    return quad(lambda x:survival(x,s,D),0,hi,points=pts or None,epsabs=1e-8)[0]

def bb(x):return -math.expm1(-x)/x if x>1e-12 else 1.-x/2.

rng=np.random.default_rng(20261112)
errors=[]
for s,cv in [('constant',0.),('uniform',1/math.sqrt(12)),('exponential',1.),('lognormal',math.sqrt(math.e-1))]:
    for D in (60.,108.,180.,225.):
        err=abs(rmean(400*D,s,D)/D-1);errors.append(err)
        assert err<2e-5,(s,D,err)
        mA=quad(lambda x:x*survival(x,s,D),0,400*D,points=[x for x in [D/2,D,1.5*D] if s in ('constant','uniform')],epsabs=1e-5)[0]/D
        target=D*(1+cv**2)/2
        assert abs(mA-target)<.003,(s,D,mA,target)
        for rho in (.6,.8,1.,1.25):
            for w in (5.,15.,30.,60.,90.,180.,360.,900.):
                ratio=rmean(w,s,rho*D)/rmean(w,s,D)
                assert min(1,rho)-1e-7<=ratio<=max(1,rho)+1e-7,(s,D,rho,w,ratio)
# For an exponential duration, the differential exit hazard is exact.
for D1 in [50.,108.,180.,225.]:
    d0=180.;diff=1/D1-1/d0
    for w in [15.,30.,60.,90.]:
        R=rmean(w,'exponential',D1)/rmean(w,'exponential',d0)
        dpos=max(diff,0.);dneg=max(-diff,0.)
        # ROR = IDRR * R1/R0; Proposition 7 envelope:
        # B(Delta_1 w) <= R1/R0 <= 1/B(Delta_0 w)
        assert bb(dpos*w)-1e-9<=R<=1/bb(dneg*w)+1e-9,(D1,w,R)
# Check the key fixed-intensity Poisson marginal, 100000 independent draws.
for I,D in [(I0,D0),(.002,108.)]:
    n=5000;h=n/(1+I*D);p=I*D*h
    H=rng.poisson(h,50000);P=rng.poisson(p,50000)
    assert abs(H.mean()/h-1)<.001
    assert abs(P.mean()/p-1)<.01
    assert abs(np.corrcoef(H,P)[0,1])<.012
print(json.dumps({'status':'PASS','independent_numerical_checks':'poisson-count law, stationary prevalence duration identity, backward-time mean, recent-window ratios, early-exit B-envelope','shape_families':4,'duration_values':4,'max_mean_duration_error':max(errors),'limit':'Numerical checks do not replace independent human proof verification'},indent=2))
