#!/usr/bin/env python3
"""Fast integrity checks for every Python numerical input used by the article."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];D=R/'results/python'
expected={'main_summary_python':2240,'main_tests_python':280,'original_summary_python':24,'original_stationarity_python':123,'validation_python':4,'trajectories_python':360,'immigration_python':8,'growth_python':20,'changing_incidence_python':35,'interval_limits_python':58,'interval_estimators_python':234,'interval_bounds_python':780,'colon_public_application':15}
report={}
for stem,n in expected.items():
 f=D/(stem+'.csv');x=pd.read_csv(f);assert len(x)==n,(stem,len(x),n)
 report[stem]={'rows':len(x),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 if 'coverage' in x:
  good=x.coverage.dropna();assert ((good>=0)&(good<=1)).all(),stem
 if 'undefined' in x:
  good=x.undefined.dropna();assert ((good>=0)&(good<=1)).all(),stem
 if 'reps' in x and stem in ('main_summary_python','main_tests_python','interval_estimators_python','interval_bounds_python'):
  assert (x.reps==10000).all(),stem
s=pd.read_csv(D/'main_summary_python.csv')
assert len(s[['id','n','estimator']].drop_duplicates())==2240
assert len(s.estimator.unique())==8 and len(s.id.unique())==56
m=s.dropna(subset=['bias_log','gmean_est','emp_se','mcse_bias','rmse_log'])
assert np.max(np.abs(m.bias_log-np.log(m.gmean_est/m.idrr)))<1e-10
nn=m.reps*(1-m.undefined)
assert np.max(np.abs(m.mcse_bias-m.emp_se/np.sqrt(nn)))<1e-10
# Rerunning exact interval code uses each window's own hazard assumption,
# so a valid delta must not be smaller than that window's true bound.
b=pd.read_csv(D/'interval_bounds_python.csv')
assert len(b.id.unique())==13 and set(b.method)=={'Wald30','Exact30','Exact60','Exact90','ExactAll'}
assert (b.valid.astype(bool)==(b.delta+1e-12>=b.delta_min)).all()
trial=pd.read_csv(D/'colon_public_application.csv');verified=pd.read_csv(R/'data/colon_verified_aggregates.csv')
assert trial.equals(verified) or np.allclose(trial.select_dtypes('number'),verified.select_dtypes('number'),rtol=1e-10,atol=1e-8,equal_nan=True)
assert (R/'tex/tab_supp_colon_body.tex').exists()
boot=pd.read_csv(D/'colon_bootstrap_ci.csv')
assert len(boot)==15 and (boot.reps==4000).all() and (boot.seed==20261112).all()
for e in ('POR','aPOR','ROR','IDRR'):
 assert np.allclose(boot[e],trial[e],rtol=1e-12,atol=1e-12,equal_nan=True)
 for part in ('_ci_low','_ci_high','_valid_fraction'):
  assert e+part in boot.columns
  if part=='_valid_fraction':assert boot[e+part].between(0,1).all()
assert np.isfinite(boot.loc[(boot.t==365)&(boot.w==90),'ROR_over_IDRR_ci_low']).all()
assert (R/'tex/tab_supp_colon_bootstrap_body.tex').exists()
assert (R/'tex/tab_supp_colon_full_ci_body.tex').exists()
assert (R/'figures/figS3_colon_trial.pdf').exists()
prov=json.loads((D/'colon_bootstrap_provenance.json').read_text())
assert prov['full_raw_validated'] is True and prov['rows']==1858 and prov['patients']==929
# real-data illustrations: bounds in the trial snapshot and the registry example
cb=pd.read_csv(D/'colon_bounds.csv');assert len(cb)==3 and list(cb.w)==[90,180,365] and (cb.t==365).all()
assert ((cb.ROR_lo<=cb.ROR)&(cb.ROR<=cb.ROR_hi)).all()
w365=cb[cb.w==365].iloc[0];t365=trial[(trial.t==365)&(trial.w==365)].iloc[0]
assert abs(w365.ROR-t365.ROR)<1e-9 and abs(w365.direct-t365.IDRR)<1e-9          # same snapshot as the earlier table
assert len(pd.read_csv(D/'colon_postrecurrence_km.csv'))==74
ex=pd.read_csv(R/'data/nordcan_denmark_extract.csv');sv=pd.read_csv(R/'data/nordcan_denmark_survival.csv')
assert len(ex)==120 and len(sv)==8 and set(ex.cancer_code)=={160,130,520,290}
nc=pd.read_csv(D/'nordcan_application.csv');assert len(nc)==4
ag=pd.read_csv(R/'data/nordcan_denmark_age.csv',keep_default_na=False);assert len(ag)==864 and set(ag.measure)=={'incidence','prevalence'}
na=pd.read_csv(D/'nordcan_age_stratified.csv');assert len(na)==4
for k in ('IDRR_age','ROR1_age','POR_age'):assert ((na[k+'_lo']<na[k])&(na[k]<na[k+'_hi'])).all()
assert ((na.bound_age_lo<=na.ROR1_age_lo+1e-12)&(na.bound_age_hi>=na.ROR1_age_hi-1e-12)).all()
# the comparison of the estimators repeated with individual histories
ind=pd.read_csv(D/'main_individual_summary.csv');cmpx=pd.read_csv(D/'main_individual_comparison.csv')
assert (ind.reps==10000).all() and len(ind)%(56*8)==0 and len(cmpx)<=len(ind) and len(cmpx)%(56*8)==0 and len(cmpx)>0
assert np.nanmax(np.abs(cmpx.z_gm))<6 and np.nanmax(np.abs(cmpx.z_cov))<6          # no gross disagreement between the two simulations
assert ((nc.ROR1_lo<=nc.ROR1)&(nc.ROR1<=nc.ROR1_hi)&(nc.bound_lo<=nc.ROR1_lo+1e-12)&(nc.bound_hi>=nc.ROR1_hi-1e-12)).all()
for f in ('tex/tab_nordcan_body.tex','tex/tab_supp_nordcan_body.tex','tex/tab_supp_nordcan_age_body.tex','tex/tab_supp_individual_body.tex','tex/tab_supp_individual_tests_body.tex','results/python/main_individual_macros.json','tex/tab_supp_colon_bounds_body.tex','tex/tab_orig_body.tex','figures/fig_nordcan.pdf','figures/fig_colon_bounds.pdf','results/python/nordcan_macros.json','results/python/colon_bounds_macros.json'):
 assert (R/f).exists(),f
for stem in ('colon_bounds','colon_postrecurrence_km','nordcan_application','nordcan_age_stratified','main_individual_summary','main_individual_tests','main_individual_comparison'):
 f=D/(stem+'.csv');report[stem]={'rows':len(pd.read_csv(f)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(R/'PYTHON_VALIDATION_REPORT.json').write_text(json.dumps({'status':'PASS', 'outputs':report,'main_estimator_rows':2240,'interval_bound_rows':780,'external_data_note':'Trial aggregates and bootstrap computed from the public colon file (not redistributed); registry series in data/ retrieved from NORDCAN 9.6.1'},indent=2)+'\n')
print('PASS all 20 numeric datasets, 56 scenario groups, 13 interval scenarios, Monte Carlo identities, trial and registry checks')
