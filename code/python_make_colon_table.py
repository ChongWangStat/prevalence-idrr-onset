#!/usr/bin/env python3
"""Regenerate the supplementary colon trial table from validated aggregate CSV."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
a=pd.read_csv(R/'results/python/colon_public_application.csv')
assert len(a)==15 and set(a.t)=={365,730,1095,1460,1825} and set(a.w)=={90,180,365}
with (R/'tex/tab_supp_colon_body.tex').open('w') as out:
    for _,x in a.iterrows():
        v=[str(int(x.t)),str(int(x.w)),str(int(x['ObsH'])),str(int(x['ObsP'])),str(int(x['Lev+5FUH'])),str(int(x['Lev+5FUP'])),f'{int(x["Lev+5FUE"])}/{int(x["ObsE"])}',*[f'{x[key]:.2f}' for key in ('POR','aPOR','ROR','IDRR')]]
        out.write(' & '.join(v)+r' \\'+'\n')
print('Rebuilt full public trial supplemental table, 15 rows')
