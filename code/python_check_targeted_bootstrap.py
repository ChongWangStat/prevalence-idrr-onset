#!/usr/bin/env python3
"""Offline integrity and consistency checks for the public trial bootstrap output."""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
p=R/'results/python/colon_bootstrap_ci.csv'
d=pd.read_csv(p)
original=pd.read_csv(R/'data/colon_verified_aggregates.csv')
assert len(d)==15 and len(original)==15
assert (d.seed==20261112).all() and (d.reps==4000).all()
assert set(d.t)=={365,730,1095,1460,1825} and set(d.w)=={90,180,365}
for k in ('t','w','POR','aPOR','ROR','IDRR'):
 assert np.allclose(d[k],original[k],rtol=1e-12,atol=1e-12,equal_nan=True),k
for arm in ('Obs','Lev+5FU'):
 for feature,k in [('H','H'),('P','P'),('R','R'),('E','E'),('PT','PT')]:
  assert np.allclose(d[arm+'_'+feature],original[arm+k],rtol=1e-12,atol=1e-8),arm+k
 assert (d[arm+'_R']<=d[arm+'_E']).all(),arm
 assert (d[arm+'_retained']==d[arm+'_H']+d[arm+'_P']).all(),arm
 for _,trows in d.groupby('t'):
  assert trows[arm+'_P'].nunique()==1 and trows[arm+'_H'].nunique()==1
  assert trows[arm+'_R'].is_monotonic_increasing
  assert trows[arm+'_E'].is_monotonic_increasing
for k in ('POR','aPOR','ROR','IDRR','ROR_over_IDRR'):
 valid=d[k+'_valid_fraction']
 assert valid.between(0,1).all(),k
 hi=d[k+'_ci_high'];lo=d[k+'_ci_low']
 # The reporting gate precludes misleading percentile intervals when
 # many bootstrap resamples yield non-positive or undefined ratios.
 assert ((valid>=.95)==(hi.notna()&lo.notna())).all(),k
 assert (hi.dropna()>=lo.dropna()).all(),k
z=d.loc[(d.t==365)&(d.w==90)].iloc[0]
assert abs(z.ROR-.62176294820717)<1e-12
assert abs(z.IDRR-.55553411657091)<1e-12
assert z.ROR_over_IDRR_ci_low<1<z.ROR_over_IDRR_ci_high
z2=d.loc[(d.t==365)&(d.w==365)].iloc[0]
assert z2.ROR_over_IDRR_ci_high<1
assert (R/'tex/tab_supp_colon_bootstrap_body.tex').exists()
assert (R/'tex/tab_supp_colon_full_ci_body.tex').exists()
print('PASS: 15 verified patient-bootstrap scenarios, uncertainty intervals, validity gates and nested cohort counts')
