"""Independent metric-library recomputation of the saved selection audit."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
d=pd.read_csv(REPO/'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv',float_precision='round_trip')
d=d[d.outcome.eq('group')]
rows=[]
for repeat,g in d.groupby('repeat'):
    row={'repeat':repeat}
    for name,col in [('baseline','p_m2'),('selector','p_adaptive_m3'),('E0','p_e0'),('E1','p_e1'),('E2','p_e2')]:
        row[name]=log_loss(g.y,g[col],labels=[0,1])
    row['independent']=.44*row['E0']+.56*row['E2']
    rows.append(row)
result=pd.DataFrame(rows).set_index('repeat')
original=pd.read_csv(HERE/'outputs/REPEAT_LOGLOSSES.csv',index_col='repeat')
assert np.max(np.abs(result.to_numpy()-original[result.columns].to_numpy()))<1e-12
result.to_csv(HERE/'outputs/INDEPENDENT_SKLEARN_CHECK.csv',float_format='%.17g')
print({key:int(result.selector.gt(result[key]).sum()) for key in ['E0','E1','E2','independent']})
