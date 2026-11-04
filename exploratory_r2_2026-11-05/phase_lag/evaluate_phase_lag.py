from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed

HERE=Path(__file__).resolve().parent

def evaluate(index):
    threadpool_limits(1)
    base,linked,e4,nuisance=baseline_data();ids=base.index.tolist()
    y=linked.group_y.to_numpy(int);b=base[B].to_numpy(float)
    manifest=read(REPO/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv')
    with np.load(HERE/'outputs/PHASE_LAG_FEATURES.npz',allow_pickle=False) as f:
        assert f['participant_id'].tolist()==ids
        raw=f['observed'] if index<0 else f['features'][index]
    bat=load(REPO/'audit/exploratory_eeg_battery/exploratory_eeg_battery.py','frozen_battery_eval')
    name='observed' if index<0 else f'phase_control_{index:02d}'
    out=HERE/'evaluation'/name
    summary=out/'SUMMARY.csv'
    if summary.exists():return read(summary).iloc[0].to_dict()
    pred=nested(ids,y,manifest,dict(baseline=lambda tr,te:scale(b[tr],b[te]),
        phase_lag=lambda tr,te:bat.model_features(b,raw,tr,te,'E3')),out)
    result=summarize(pred,'p_baseline','p_phase_lag',name,out)
    if index<0:
        original=read(REPO/'audit/exploratory_eeg_battery/OUTER_PREDICTIONS.csv')
        original=original[original.family.eq('E3')]
        check=pred.merge(original,on=['repeat','outer_fold','participant_id','y'],validate='one_to_one')
        assert len(check)==960
        errors={n:float(np.max(np.abs(check[n]-check[o])))
                for n,o in [('p_baseline','p_m2'),('p_phase_lag','p_eeg')]}
        assert max(errors.values())<1e-7,errors
        write(out/'OBSERVED_REPLAY_CHECK.json',errors)
    pd.DataFrame([result]).to_csv(summary,index=False,float_format='%.17g')
    return result

def main():
    assert (HERE/'outputs/PHASE_LAG_FEATURES.npz').exists()
    observed=evaluate(-1)
    results=[observed]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate,j) for j in range(20)]
        for count,done in enumerate(as_completed(futures),1):
            row=done.result();results.append(row)
            print(f'Completed phase-lag nested evaluation {count}/20: {row["comparison"]}',flush=True)
    result=pd.DataFrame(results).sort_values('comparison')
    result.to_csv(HERE/'outputs/SUMMARY.csv',index=False,float_format='%.17g')
    controls=result[result.comparison.ne('observed')]
    description=dict(observed_delta_ll=float(observed['delta_ll']),
        control_mean_delta_ll=float(controls.delta_ll.mean()),
        control_min_delta_ll=float(controls.delta_ll.min()),
        control_max_delta_ll=float(controls.delta_ll.max()),
        observed_minus_control_mean=float(observed['delta_ll']-controls.delta_ll.mean()),
        controls_above_observed=int(controls.delta_ll.gt(observed['delta_ll']).sum()),
        positive_control_means=int(controls.delta_ll.gt(0).sum()))
    write(HERE/'outputs/DESCRIPTIVE_COMPARISON.json',description)
    provenance(HERE/'outputs',[Path(__file__),HERE/'PLAN_2026-11-05.md',
        HERE/'outputs/PHASE_LAG_FEATURES.npz',REPO/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv',
        REPO/'audit/exploratory_eeg_battery/exploratory_eeg_battery.py'],description)
    print(result.to_string(index=False))

if __name__=='__main__':main()
