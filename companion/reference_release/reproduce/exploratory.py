"""Recompute E3/E4 family models with unchanged archived evaluation functions."""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_posthoc as core
ROOT=core.ROOT;OUT=Path(os.environ['EEG_RUN_ROOT'])/'exploratory'
def main():
    OUT.mkdir(exist_ok=True);data=core.load_data()
    battery=core.load_module(core.AUDIT/'exploratory_eeg_battery/exploratory_eeg_battery.py','battery')
    phase={'ids':data['ids']}
    for family in ('E3','E4'):
        phase[family.lower()+'_frame']=pd.read_csv(core.AUDIT/f'exploratory_eeg_battery/{family}_RAW_FEATURES.csv')
        errors=[]
        with threadpool_limits(1):pred,fold,selection=battery.run_family(family,phase,core.PRIMARY,data['old'],data['folds'],errors)
        p=pd.DataFrame(pred);p.to_csv(OUT/f'{family}_predictions.csv',index=False,float_format='%.17g')
        pd.DataFrame(selection).to_csv(OUT/f'{family}_selection.csv',index=False,float_format='%.17g')
        pd.DataFrame(fold).to_csv(OUT/f'{family}_fold_results.csv',index=False,float_format='%.17g')
        assert not errors
        old=pd.read_csv(core.AUDIT/'exploratory_eeg_battery/OUTER_PREDICTIONS.csv')
        old=old[old.family.eq(family)]
        m=p.merge(old,on=['family','repeat','outer_fold','participant_id','y'],suffixes=('_new','_old'),validate='one_to_one')
        error=float(np.max(np.abs(m.p_eeg_new-m.p_eeg_old)))
        assert len(m)==960 and error<1e-7 and (m.selected_C_new==m.selected_C_old).all()
        (OUT/f'{family}_checks.json').write_text(json.dumps({'predictions':len(m),'max_error':error,'choices_exact':True,'failures':errors},indent=2)+'\n')
        print(f'PASS: {family}, 50 outer fits, max prediction error {error}',flush=True)
if __name__=='__main__':main()
