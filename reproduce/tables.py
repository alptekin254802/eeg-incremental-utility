"""Generate numerical table inputs, for the reported primary and post-hoc comparisons."""
from pathlib import Path
import os,json
import pandas as pd
ROOT=Path(os.environ['EEG_RELEASE_ROOT']);RUN=Path(os.environ['EEG_RUN_ROOT']);OUT=RUN/'tables';OUT.mkdir(exist_ok=True)
A=ROOT/'audit';P=A/'preregistered_analysis';E=A/'exploratory_eeg_battery'
rr=pd.read_csv(E/'REPEAT_LEVEL_RESULTS.csv')
e5=pd.read_csv(ROOT/'results/e5/REPEAT_RESULTS.csv')
rr=pd.concat([rr[~rr.family.eq('E5')],e5],ignore_index=True)
rr.to_csv(OUT/'Table_S5.csv',index=False,float_format='%.17g')
master=pd.read_csv(E/'EXPLORATORY_EEG_BATTERY_MASTER_TABLE.csv')
e5_row=master.loc[master.family.eq('E4')].copy()
e5_row['family']='E5';e5_row['model_with_EEG']='M2+E5'
master=pd.concat([master,e5_row],ignore_index=True)
c=json.loads((ROOT/'results/e5/RESULTS.json').read_text(encoding='utf-8'))['metrics']
for column,key in {'logloss_with_EEG':'eeg_logloss','Delta_LL':'delta_ll','AUROC_with_EEG':'eeg_auroc','Delta_AUROC':'delta_auroc','Brier_with_EEG':'eeg_brier','Delta_Brier':'delta_brier'}.items():
    assert column in master;master.loc[master.family.eq('E5'),column]=c[key]
for column,value in {'Delta_LL_median':e5.delta_ll.median(),'Delta_LL_IQR_type7':e5.delta_ll.quantile(.75)-e5.delta_ll.quantile(.25),'Delta_LL_min':e5.delta_ll.min(),'Delta_LL_max':e5.delta_ll.max()}.items():
    master.loc[master.family.eq('E5'),column]=value
master.loc[master.family.eq('E5'),'status']='COMPLETED'
# Preserve original schemas so units and metric direction remain explicit.
master.to_csv(OUT/'Table_1_exploratory.csv',index=False,float_format='%.17g')
for source,target in [('PREREGISTERED_RESULTS_MASTER_TABLE.csv','Table_1_primary.csv'),('MODALITY_LADDER_CONSOLIDATION.csv','Table_S3_ladder.csv'),('DIAGNOSED_SENSITIVITY.csv','Table_S4.csv'),('CALIBRATION_RESULTS.csv','Table_S6.csv')]:
    p=P/source
    assert p.exists();pd.read_csv(p).to_csv(OUT/target,index=False,float_format='%.17g')
for folder,source,target in [('results','comparison_summary.csv','Table_S7_all_comparisons.csv'),('signal_controls','comparison_summary.csv','Table_S8_all_controls.csv')]:
    pd.read_csv(RUN/'posthoc'/folder/source).to_csv(OUT/target,index=False,float_format='%.17g')
(OUT/'README.md').write_text('Numerical table inputs. Table 1 combines primary and exploratory files; Table 2 is the selected subset of the complete Table S7 comparisons. Table S5 includes all three post-hoc EEG representations. Figure/table regeneration starts from the immutable released result tables; fit reproduction is a separate command. Units, metric direction, and formulas are specified in the manuscript.\n',encoding='utf-8')
print('Table inputs generated in '+str(OUT))
