#!/usr/bin/env python3
"""Create two concise supplementary tables from seeded paired patient bootstrap."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'results/python/colon_bootstrap_ci.csv')
assert len(d)==15 and set(d.t)=={365,730,1095,1460,1825} and set(d.w)=={90,180,365}
assert (d.reps==4000).all() and (d.seed==20261112).all()
def fmt(v):
    return '--' if not np.isfinite(v) else f'{v:.2f}'
def est(r,n):
    v=fmt(r[n]);lo=r[n+'_ci_low'];hi=r[n+'_ci_high']
    if not np.isfinite(lo) or not np.isfinite(hi):return f'{v} [--]'
    return f'{v} [{lo:.2f}, {hi:.2f}]'

rows=[]
for _,r in d.sort_values(['t','w']).iterrows():
    cells=[str(int(r.t)),str(int(r.w)),f'{int(r["Lev+5FU_R"])}/{int(r.Obs_R)}',f'{int(r["Lev+5FU_E"])}/{int(r.Obs_E)}',est(r,'ROR'),est(r,'IDRR'),est(r,'ROR_over_IDRR')]
    rows.append(' & '.join(cells)+r' \\')
(ROOT/'tex/tab_supp_colon_bootstrap_body.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for _,r in d[d.w==90].sort_values('t').iterrows():
    cells=[str(int(r.t)),f'{int(r["Lev+5FU_retained"])}/{int(r.Obs_retained)}',est(r,'POR'),est(r,'aPOR'),est(r,'ROR'),est(r,'IDRR')]
    rows.append(' & '.join(cells)+r' \\')
(ROOT/'tex/tab_supp_colon_full_ci_body.tex').write_text('\n'.join(rows)+'\n')
print('Wrote 15-row paired-window and five-row all-estimator bootstrap tables')
