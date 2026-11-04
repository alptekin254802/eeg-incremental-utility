from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def main():
    here=Path(__file__).resolve().parent;out=here/'outputs'
    base,linked,e4,nuisance=baseline_data()
    keep=linked.diagnosed_y.notna().to_numpy()
    ids=base.index[keep].tolist();y=linked.loc[ids,'diagnosed_y'].to_numpy(int)
    assert len(ids)==84 and set(y)=={0,1}
    manifest=read(REPO/'audit/preregistered_analysis/DIAGNOSED_FOLD_MANIFEST.csv')
    assert set(manifest.participant_id)==set(ids)
    b=base.loc[ids,B].to_numpy(float);e=e4.loc[ids].to_numpy(float);n=nuisance[keep]
    designs={}
    for suffix,width in [('',0),('_DQ',3)]:
        for model,feature in [('baseline',np.empty((len(ids),0))),('entropy',e)]:
            raw=np.column_stack([b,n[:,:width],feature])
            assert np.isfinite(raw).all()
            designs[model+suffix]=lambda tr,te,raw=raw:scale(raw[tr],raw[te])
    pred=nested(ids,y,manifest,designs,out)
    results=[summarize(pred,'p_baseline'+s,'p_entropy'+s,'diagnosis_entropy'+s,out)
             for s in ['', '_DQ']]
    recorded=read(REPO/'audit/preregistered_analysis/DIAGNOSED_SENSITIVITY.csv')
    replay=read(out/'diagnosis_entropy_REPEATS.csv').merge(recorded,on='repeat',validate='one_to_one')
    error=float(np.max(np.abs(replay.baseline_logloss-replay.m2_logloss)))
    assert error<1e-7,('Diagnosis baseline replay failed',error)
    pd.DataFrame(results).to_csv(out/'SUMMARY.csv',index=False,float_format='%.17g')
    provenance(out,[here/'PLAN_2026-11-05.md',Path(__file__),
        REPO/'audit/preregistered_analysis/DIAGNOSED_FOLD_MANIFEST.csv',
        REPO/'audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv',
        REPO/'audit/exploratory_eeg_battery/E4_RAW_FEATURES.csv',
        REPO/'audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv',
        REPO/'audit/stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv'],
        {'recorded_baseline_max_abs_error':error,'new_encoder_fits':0})
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':main()
