#!/usr/bin/env python3
"""Merge independent partitions and compare the Python exact-law MC to archived R MC.

Checks the 56 x 5 x 8 design, all reported outputs, and Monte Carlo uncertainty.
This is a statistical reproduction, not a claim of matching R's PRNG stream.
"""
import json, math
from pathlib import Path
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent.parent
out=HERE/'results/python';legacy=HERE/'results'
S=[];T=[]
for part in ['main','chunkA','chunkB','chunkC','chunkD']:
 r=out if part=='main' else out/part
 for kind,collect in [('main_summary',S),('main_tests',T)]:
  df=pd.read_csv(r/f'{kind}.csv')
  collect.append(df)
s=pd.concat(S,ignore_index=True).sort_values(['id','n','estimator'])
t=pd.concat(T,ignore_index=True).sort_values(['id','n'])
assert len(s)==56*5*8,(len(s),len(s.drop_duplicates(['id','n','estimator'])))
assert len(t)==56*5
assert s[['id','n','estimator']].duplicated().sum()==0
assert t[['id','n']].duplicated().sum()==0
assert (s.reps==10000).all() and (t.reps==10000).all()
s.to_csv(out/'main_summary_python.csv',index=False,float_format='%.15g')
t.to_csv(out/'main_tests_python.csv',index=False,float_format='%.15g')
R=pd.read_csv(legacy/'main_summary.csv')
T=pd.read_csv(legacy/'main_tests.csv')
joined=s.merge(R,on=['id','n','estimator'],suffixes=('_py','_r'),validate='one_to_one')
assert len(joined)==len(s)
rows=[]
for metric,sd in [('gmean_est',None),('emp_se',None),('coverage','mcse_cov'),('reject','mcse_rej'),('undefined',None)]:
 x=joined[f'{metric}_py'];y=joined[f'{metric}_r'];d=(x-y).abs()
 o=dict(metric=metric,n=int(d.count()),median_abs_diff=float(d.median()),max_abs_diff=float(d.max()))
 if sd:
  z=d/np.sqrt(joined[f'{sd}_py']**2+joined[f'{sd}_r']**2)
  o['fraction_over_3_combined_MCSE']=float((z>3).mean());o['max_combined_MCSE']=float(z.max())
 rows.append(o)
# Independent cross-check of log scale identities
ok=s[['gmean_est','bias_log','rmse_log','emp_se','mcse_bias','undefined','reps','idrr']].dropna()
assert np.allclose(np.log(ok.gmean_est/ok.idrr),ok.bias_log,atol=1e-10,rtol=1e-10)
maxmcse=(ok.mcse_bias-ok.emp_se/np.sqrt(ok.reps*(1-ok.undefined))).abs().max()
assert maxmcse<1e-8,maxmcse
jt=t.merge(T,on=['id','n'],suffixes=('_py','_r'),validate='one_to_one')
tests=[]
for name in ('rej_wilcoxon','rej_ks','rej_shape','rej_recent30','rej_recent60','rej_recent90','rej_linear'):
 d=(jt[f'{name}_py']-jt[f'{name}_r']).abs()
 tests.append(dict(test=name,median_abs_diff=float(d.median()),max_abs_diff=float(d.max()),fraction_diff_gt_0_05=float((d>.05).mean())))
RPT=dict(status='PASS',simulation_scenarios=56,expected_group_sizes=5,replications_per_setting=10000,total_surveys=56*5*10000,estimators=8,estimator_summary_rows=len(s),test_summary_rows=len(t),numerical_errors=rows,diagnostic_discrepancies=tests,max_mcse_identity_error=float(maxmcse),note='R and Python use independent simulation algorithms and seeds. Model differences in approximated KS/Wilcoxon p-values are evaluated separately.')
(out/'main_python_audit.json').write_text(json.dumps(RPT,indent=2))
print(json.dumps(RPT,indent=2))
