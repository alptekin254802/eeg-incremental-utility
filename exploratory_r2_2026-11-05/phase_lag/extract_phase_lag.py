"""Recreate the recorded 20 surrogates exactly, then extract frozen E3."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed

RAW=Path(r'F:\verisetleri\BALLADEER ADHD DATASET\dataset')
HERE=Path(__file__).resolve().parent
os.environ['EEG_RELEASE_ROOT']=str(REPO)
os.environ['EEG_RUN_ROOT']=str(HERE/'support')
os.environ['EEG_DATASET_ROOT']=str(RAW)
sys.path.insert(0,str(REPO/'reproduce'))
import run_signal_controls as control
BAT=control.BATTERY

def e3_fast(windows):
    data=np.asarray(windows,dtype=float)
    data=data-data.mean(axis=1,keepdims=True)
    spectra=np.fft.rfft(data[:,None,:,:]*BAT.E3_TAPERS[None,:,:,None],n=256,axis=2)
    frequencies=np.fft.rfftfreq(256,d=1/BAT.FS)
    result=[]
    for band,(low,high) in BAT.BANDS.items():
        bins=np.flatnonzero((frequencies>=low)&(frequencies<=high))
        selected=spectra[:,:,bins,:]
        values=[]
        for start in range(0,len(BAT.PAIR_LIST),13):
            pairs=BAT.PAIR_LIST[start:start+13]
            x=np.imag(selected[:,:,:,[i for i,j in pairs]]*
                      np.conj(selected[:,:,:,[j for i,j in pairs]]))
            x=x.reshape(-1,len(bins),len(pairs))
            sx=x.sum(axis=0);sq=np.square(x).sum(axis=0);sa=np.abs(x).sum(axis=0)
            den=sa**2-sq
            assert np.all(den>np.finfo(float).eps*np.maximum(1,sa**2))
            values.extend(np.median((sx**2-sq)/den,axis=0))
        result.extend(values)
    result=np.asarray(result)
    assert result.shape==(273,) and np.isfinite(result).all()
    return result

def one(pid):
    threadpool_limits(1)
    out=HERE/'participant_features';out.mkdir(exist_ok=True)
    target=out/(pid+'.npz')
    if target.exists():return pid,'checkpoint'
    s=REPO/'audit/stage1_blind_feature_extraction'
    base=read(s/'BLIND_PRIMARY_FEATURE_MATRIX.csv').set_index('participant_id')
    roster=read(s/'BLIND_COHORT_MANIFEST.csv').set_index('participant_id')
    qc=read(s/'EEG_PREPROCESSING_QC.csv').set_index('participant_id')
    records=json.loads((REPO/'expected/posthoc/signal_controls/reconstruction_and_seeds.json').read_text())
    recorded={r['participant_id']:r for r in records}[pid]
    path=RAW/recorded['raw_relative_path']
    assert path.resolve().is_relative_to(RAW.resolve())
    assert sha(path)==recorded['raw_sha256']
    got=BAT.stage1_capture(control.STAGE,dict(participant_id=pid,
        session_id=str(roster.loc[pid,'session_id']),eeg_path=path),{pid:{'age':float(base.loc[pid,'age'])}})
    assert got['qc']['preprocessing_status']=='PASS'
    for key in ['candidate_window_count','rejected_window_count','clean_window_fraction','clean_seconds','bad_channel_count']:
        assert abs(float(got['qc'][key])-float(qc.loc[pid,key]))<1e-10,(pid,key)
    windows=np.stack(got['accepted_windows'])
    assert len(windows)==recorded['accepted_windows']
    e0err=max(abs(got['e0'][k]-base.loc[pid,k]) for k in control.STAGE.EEG_FEATURES)
    assert e0err<1e-10
    original=read(REPO/'audit/exploratory_eeg_battery/E3_RAW_FEATURES.csv').set_index('participant_id')
    observed=e3_fast(windows)
    observed_error=float(np.max(np.abs(observed-original.loc[pid].to_numpy(float))))
    assert observed_error<1e-10,(pid,observed_error)
    with np.load(REPO/'expected/posthoc/signal_controls/surrogate_features.npz',allow_pickle=False) as f:
        expected=f['features'][:,f['participant_id'].tolist().index(pid),:]
    values=[];e4errors=[];spectrum_errors=[];frozen_error=0.
    for j,seed in enumerate(recorded['seeds']):
        assert seed==control.seed_for(pid,j)
        surrogate,spectrum_error=control.phase_surrogate(windows,seed)
        e4err=float(np.max(np.abs(control.e4_vectorized(surrogate)-expected[j])))
        assert e4err<1e-10,(pid,j,e4err)
        feature=e3_fast(surrogate)
        if pid=='UB0004' and j==0:
            direct=BAT.e3_extract(list(surrogate))
            frozen_error=float(np.max(np.abs(feature-np.array(list(direct.values())))))
            assert frozen_error<1e-10
        values.append(feature);e4errors.append(e4err);spectrum_errors.append(spectrum_error)
    assert sha(path)==recorded['raw_sha256']
    np.savez_compressed(target,features=np.asarray(values),observed=observed,
        observed_e3_error=observed_error,control_e4_error=max(e4errors),e0_error=e0err,
        spectrum_error=max(spectrum_errors),frozen_function_error=frozen_error)
    return pid,'complete'

def main():
    here=HERE;out=here/'outputs';out.mkdir(exist_ok=True)
    ids=read(REPO/'audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv').participant_id.tolist()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,pid) for pid in ids]
        for count,done in enumerate(as_completed(futures),1):
            pid,state=done.result();print(f'Exact phase reconstruction + phase-lag: {count}/96 {pid} {state}',flush=True)
    arrays=[];checks=[];observed=[]
    for pid in ids:
        path=here/'participant_features'/(pid+'.npz')
        with np.load(path,allow_pickle=False) as f:
            arrays.append(f['features']);observed.append(f['observed'])
            checks.append(dict(participant_id=pid,sha256=sha(path),
                **{k:float(f[k]) for k in ['observed_e3_error','control_e4_error','e0_error','spectrum_error','frozen_function_error']}))
    np.savez_compressed(out/'PHASE_LAG_FEATURES.npz',participant_id=np.array(ids),
                        features=np.stack(arrays,axis=1),observed=np.asarray(observed))
    pd.DataFrame(checks).to_csv(out/'RECONSTRUCTION_CHECKS.csv',index=False,float_format='%.17g')
    provenance(out,[Path(__file__),here/'PLAN_2026-11-05.md',
        REPO/'expected/posthoc/signal_controls/reconstruction_and_seeds.json',
        REPO/'expected/posthoc/signal_controls/surrogate_features.npz',
        REPO/'audit/exploratory_eeg_battery/E3_RAW_FEATURES.csv'],
        {'phase_datasets':20,'participants':96,'raw_root':str(RAW),'exact_saved_seed_reconstruction':True})
    print('All 20 phase datasets reproduced; phase-lag features complete.',flush=True)

if __name__=='__main__':main()
