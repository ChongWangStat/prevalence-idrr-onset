#!/usr/bin/env python3
"""Validate the Numba noncentral-hypergeometric CI against SciPy's exact inversion."""
import numpy as np
from scipy.stats.contingency import odds_ratio
from python_interval_reproduction import fisher_ci
cases=[(40,2000,25,1900),(5,200,8,250),(0,100,2,90),(20,5000,5,4500),(30,500,0,900),(100,20,40,30)]
for a,b,c,d in cases:
    for alpha in (.05,.05/3):
        lhs=fisher_ci(a,b,c,d,alpha)
        ref=odds_ratio([[a,b],[c,d]],kind='conditional').confidence_interval(1-alpha)
        actual=(ref.low,ref.high)
        if not np.allclose(lhs,actual,atol=1e-7,rtol=1e-7,equal_nan=True):
            raise AssertionError((a,b,c,d,alpha,lhs,actual))
print('PASS: 12 SciPy--Numba exact-interval comparisons, including zero counts')
