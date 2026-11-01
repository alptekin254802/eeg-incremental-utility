"""Extract features from raw recordings and fit nested models.

Saved reference data are inaccessible to fitting processes. Checkpoints are
tied to this run, the source code, and raw-input identities."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, atexit
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_posthoc as core
import run_signal_controls as signal

ROOT = core.ROOT.resolve()
RAW = Path(os.environ['EEG_DATASET_ROOT']).resolve()
RUN = Path(os.environ['EEG_RUN_ROOT']).resolve()
FRESH = RUN / 'fresh_data'
CHECK = RUN / 'checkpoints'
ACCESS = RUN / 'access'
STAGE = signal.STAGE
BATTERY = signal.BATTERY
PRIMARY = core.PRIMARY
ENRICH = core.load_module(ROOT/'audit/stage1_5_enriched_eeg_feasibility/run_stage1_5_audit.py', 'fresh_enrich')
QUALITY = core.load_module(ROOT/'audit/cohort_screen/robots_quality_audit.py', 'fresh_quality')
QUALITY.DATASET = RAW
QUALITY.ROOT = RAW.parent
S1 = FRESH/'stage1_blind_feature_extraction'
S15 = FRESH/'stage1_5_enriched_eeg_feasibility'
EB = FRESH/'exploratory_eeg_battery'
PA = FRESH/'preregistered_analysis'
READS, DENIED = set(), set()
GUARDED = False
DATA = None

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, default=default, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')

def flush_access():
    if GUARDED:
        dump(ACCESS/f'{os.getpid()}.json', {'historical_data_reads_denied':sorted(DENIED), 'read_paths':sorted(READS)})

def install_guard():
    global GUARDED
    if GUARDED: return
    blocked = [os.path.normcase(str(ROOT/p)+os.sep) for p in ('audit','expected','results')]
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        name = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        mode, flags = args[1], args[2]
        reading = (isinstance(mode,str) and ('r' in mode or '+' in mode)) or (mode is None and not flags & os.O_WRONLY)
        if not reading: return
        if any(name.startswith(prefix) for prefix in blocked) and not name.endswith('.py'):
            DENIED.add(name)
            raise PermissionError('Fresh pipeline forbids historical numerical inputs: '+name)
        READS.add(name)
    sys.addaudithook(hook)
    GUARDED = True
    atexit.register(flush_access)

def bind():
    core.AUDIT = FRESH
    PRIMARY.METADATA_PATH = RAW/'users_demographics.json'
    PRIMARY.OUT = PA
    PRIMARY.PRIMARY_MATRIX_PATH = S1/'BLIND_PRIMARY_FEATURE_MATRIX.csv'
    BATTERY.STAGE1_MATRIX = PRIMARY.PRIMARY_MATRIX_PATH
    core.load_data = load_data
    for p in (FRESH,CHECK,ACCESS,S1,S15,EB,PA,core.RESULT,signal.OUT): p.mkdir(parents=True,exist_ok=True)
    threadpool_limits(1)
    install_guard()

def code_hashes():
    # Source only. No historical numeric input is opened by this process.
    paths = [Path(__file__), Path(core.__file__), Path(signal.__file__),
             Path(STAGE.__file__), Path(BATTERY.__file__), Path(PRIMARY.__file__),
             Path(ENRICH.__file__), Path(QUALITY.__file__)]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def run_contract():
    value = {'code':code_hashes(), 'metadata_sha256':digest(RAW/'users_demographics.json'),
             'dataset':str(RAW), 'python':sys.version, 'versions':{m.__name__:m.__version__ for m in (np,pd,core.scipy,core.sklearn)},
             'workers_maximum':2, 'blas_threads':1, 'reference_data_allowed':False}
    path = RUN/'FRESH_CONTRACT.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == value, 'Source/environment changed; use a NEW output directory'
    else: dump(path,value)
    return digest(path)

def checkpoint(kind, key, signature, compute):
    path = CHECK/kind/(str(key)+'.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['signature'] == signature, 'Stale checkpoint '+str(path)
        return saved['value']
    value = compute()
    dump(path, {'signature':signature, 'value':value})
    flush_access()
    # Return the same JSON types on a first run and a resumed run. In
    # particular, QC tuples must not become a different CSV representation.
    return json.loads(path.read_text(encoding='utf-8'))['value']

def raw_hashes(record):
    return {name:digest(record[name+'_path']) for name in ('eeg','eye','behavior')}

def screen_worker(payload):
    bind()
    record, signature = payload
    hashes = raw_hashes(record)
    def compute():
        rows=[]
        for name in ('eeg','eye','behavior'):
            path=record[name+'_path']
            row=getattr(QUALITY,'parse_'+name)(path,True)
            row.update(participant_id=record['participant_id'],session_id=record['session_id'],source_file=str(path.resolve()),raw_sha256=hashes[name])
            rows.append(row)
        assert raw_hashes(record)==hashes, 'Raw data changed during read'
        return rows
    return checkpoint('screen',str(record['participant_id'])+'_'+str(record['session_id']),[signature,hashes],compute)

def screen(jobs, signature):
    demographics=STAGE.load_allowed_demographics(RAW/'users_demographics.json')
    sessions=QUALITY.discover_sessions({pid:{} for pid in demographics})
    complete=[s for s in sessions if all(s[k+'_path'] is not None for k in ('eeg','eye','behavior'))]
    assert len(complete)==126
    rows=[[],[],[]]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i,result in enumerate(pool.map(screen_worker,[(s,signature) for s in complete]),1):
            for j,row in enumerate(result): rows[j].append(row)
            if i%20==0: print(f'Fresh measurement screen {i}/126',flush=True)
    dest=RUN/'screen/audit/cohort_screen'
    for name,records in zip(('eeg','eye','behavior'),rows): csv(pd.DataFrame(records),dest/f'robots_{name}_quality.csv')
    cohort, summary=STAGE.reconstruct_cohort(RUN/'screen')
    dump(RUN/'screen/SUMMARY.json',dict(raw_complete_sessions=len(complete),**summary,outcome_used_for_eligibility=False))
    return cohort,demographics

def feature_worker(payload):
    bind()
    record, demographic, signature=payload
    hashes=raw_hashes(record)
    pid=record['participant_id']
    def compute():
        captured=BATTERY.stage1_capture(STAGE,record,{pid:demographic})
        # stage1_capture omits this reporting field. Preserve the original
        # stage1 process_eeg rule: only up to MAX_BAD_CHANNELS are interpolated.
        bad_count=captured['qc']['bad_channel_count']
        captured['qc']['interpolation_count']=bad_count if bad_count<=STAGE.MAX_BAD_CHANNELS else 0
        result={'participant_id':pid,'raw_hashes':hashes,'qc':captured['qc']}
        if captured['qc']['preprocessing_status']=='PASS':
            windows,segments,indices=captured['accepted_windows'],captured['segments'],captured['accepted_ids']
            result['primary']={'participant_id':pid,**demographic,**STAGE.extract_behavior(record['behavior_path']),**STAGE.extract_gaze(record['eye_path']),**captured['e0']}
            topo,temp,entropy,tn,tm,en=ENRICH.e1_features(windows)
            for name,values,names in (('topo',topo,tn),('temporal',temp,tm),('entropy',entropy,en)):
                values=np.asarray(values,float).ravel()
                assert len(names)==len(values) and np.isfinite(values).all()
                result[name]={'participant_id':pid,**dict(zip(names,values))}
            covariance,_=ENRICH.e2_covariance(segments,indices,ENRICH.helmert_contrast_basis(14))
            result['covariances']=covariance
            result['E3']={'participant_id':pid,**BATTERY.e3_extract(windows)}
            result['E4']={'participant_id':pid,**BATTERY.e4_extract(windows)}
            result['E5']={'participant_id':pid,**BATTERY.e5_extract(segments,indices)}
        assert hashes==raw_hashes(record)
        return result
    return checkpoint('features',pid,[signature,hashes,demographic],compute)

def features(jobs,signature):
    cohort,demographics=screen(jobs,signature)
    results=[]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(feature_worker,(r,demographics[r['participant_id']],signature)) for r in cohort]
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%10==0 or i==98: print(f'Fresh feature extraction {i}/98',flush=True)
    results.sort(key=lambda r:r['participant_id'])
    accepted=[r for r in results if r['qc']['preprocessing_status']=='PASS']
    assert len(accepted)==96
    roster=[]
    for r in cohort:
        item={k:v for k,v in r.items() if not k.endswith('_path')}
        for name in ('eeg','eye','behavior'): item[name+'_source_file']='dataset/'+r[name+'_path'].relative_to(RAW).as_posix()
        roster.append(item)
    csv(pd.DataFrame(roster),S1/'BLIND_COHORT_MANIFEST.csv')
    csv(pd.DataFrame([r['qc'] for r in results]),S1/'EEG_PREPROCESSING_QC.csv')
    csv(pd.DataFrame([r['primary'] for r in accepted])[STAGE.FEATURE_COLUMNS],S1/'BLIND_PRIMARY_FEATURE_MATRIX.csv')
    for key,file in [('topo','E1_TOPOGRAPHIC_SPECTRAL_RAW.csv'),('temporal','E1_TEMPORAL_IQR_RAW.csv'),('entropy','E1_REGIONAL_ENTROPY.csv')]:
        csv(pd.DataFrame([r[key] for r in accepted]),S15/file)
    np.savez_compressed(S15/'E2_PARTICIPANT_COVARIANCES.npz',participant_id=np.array([r['participant_id'] for r in accepted]),bands=np.array(PRIMARY.BANDS),channel_order=np.array(BATTERY.CHANNELS),covariances=np.array([r['covariances'] for r in accepted]))
    for family in ('E3','E4','E5'): csv(pd.DataFrame([r[family] for r in accepted]),EB/f'{family}_RAW_FEATURES.csv')
    paths=sorted(p for folder in (S1,S15,EB) for p in folder.iterdir() if p.is_file())
    dump(RUN/'FEATURE_MANIFEST.json',{'contract_sha256':signature,'raw_sources':{r['participant_id']:r['raw_hashes'] for r in results},'sha256':{p.relative_to(FRESH).as_posix():digest(p) for p in paths},'generated_by':'fresh raw extraction; no saved feature inputs'})

def load_features():
    manifest=json.loads((RUN/'FEATURE_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['contract_sha256']==digest(RUN/'FRESH_CONTRACT.json')
    for name,sha in manifest['sha256'].items(): assert digest(FRESH/name)==sha, 'Fresh feature changed: '+name
    base=pd.read_csv(S1/'BLIND_PRIMARY_FEATURE_MATRIX.csv')
    ids=base.participant_id.tolist()
    assert len(ids)==len(set(ids))==96 and list(base)==STAGE.FEATURE_COLUMNS
    measurement={'ids':ids,'hashes':manifest['sha256']}
    for key,file in [('topo','E1_TOPOGRAPHIC_SPECTRAL_RAW.csv'),('temporal','E1_TEMPORAL_IQR_RAW.csv'),('entropy','E1_REGIONAL_ENTROPY.csv')]:
        block=pd.read_csv(S15/file)
        assert block.participant_id.tolist()==ids
        assert block.shape[1]-1=={'topo':70,'temporal':15,'entropy':3}[key]
        assert np.isfinite(block.drop(columns='participant_id').to_numpy(float)).all()
        measurement[key+'_columns']=[c for c in block if c!='participant_id']
        base=base.merge(block,on='participant_id',validate='one_to_one')
    with np.load(S15/'E2_PARTICIPANT_COVARIANCES.npz',allow_pickle=False) as archive:
        assert archive['participant_id'].tolist()==ids
        measurement['covariances']=archive['covariances']
    assert measurement['covariances'].shape==(96,3,13,13)
    linked,linkage=PRIMARY.link_outcomes(base[['participant_id']])
    y=linked.group_y.to_numpy(int)
    group=PRIMARY.build_folds(ids,y,'group')
    records={r['user']:r for r in json.loads((RAW/'users_demographics.json').read_text(encoding='utf-8'))}
    di=[i for i,pid in enumerate(ids) if str(records[pid].get('diagnosed','')).lower().strip() in ('yes','no')]
    dy=np.array([str(records[ids[i]]['diagnosed']).lower().strip()=='yes' for i in di],int)
    diagnosis=PRIMARY.build_folds([ids[i] for i in di],dy,'diagnosed')
    assert len(ids)==96 and y.sum()==41 and len(di)==84 and dy.sum()==54
    return base,measurement,group,diagnosis,di

def load_data():
    base,measurement,folds,_,_=load_features()
    ids=measurement['ids']
    e4=pd.read_csv(EB/'E4_RAW_FEATURES.csv').set_index('participant_id').loc[ids]
    qc=pd.read_csv(S1/'EEG_PREPROCESSING_QC.csv').set_index('participant_id').loc[ids]
    old=pd.read_csv(PA/'PRIMARY_OUTER_PREDICTIONS.csv') # Predictions fitted from the new raw-derived features.
    assert len(old)==960 and not old.duplicated(['repeat','participant_id']).any()
    return dict(ids=ids,y=folds['y'],folds=folds,raw=PRIMARY.prepare_raw(base,measurement),e4=e4.to_numpy(float),e4_names=e4.columns.tolist(),nuisance=np.column_stack([qc.device.eq('Epoc+').astype(float),qc.clean_window_fraction,qc.interpolation_count]),old=old,qc=qc,base=base)

def initialize_models():
    global DATA
    bind()
    base,measurement,group,diagnosis,di=load_features()
    DATA={'group':(PRIMARY.prepare_raw(base,measurement),group),'diagnosed':(PRIMARY.prepare_raw(base,measurement,di),diagnosis)}
    dump(ACCESS/f'learner_{os.getpid()}.json',{'feature_manifest_sha256':digest(RUN/'FEATURE_MANIFEST.json'),'source':'new raw-derived bundle opened and hash-checked inside worker','array_sha256':{outcome:{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in raw.items()} for outcome,(raw,_) in DATA.items()}})

def model_worker(job):
    outcome,mode,family,index=job
    raw,full=DATA[outcome]
    def compute():
        failures=[]
        result=PRIMARY.run_tuned_model(dict(full,outer=[full['outer'][index]]),raw,mode,family,failures)
        assert not failures, failures
        return result
    return checkpoint('models',f'{outcome}_{family}_{index}',digest(RUN/'FEATURE_MANIFEST.json'),compute)

def models(jobs):
    base,measurement,group,diagnosis,di=load_features()
    PRIMARY.write_folds(group,diagnosis,measurement['hashes'])
    modes=[('group',m,'adaptive_M3' if m=='adaptive' else m) for m in ('M0','M1','M2','adaptive','E0','E1','E2')]
    modes += [('diagnosed','M2','M2_diagnosed'),('diagnosed','adaptive','adaptive_M3_diagnosed')]
    work=[(*mode,i) for mode in modes for i in range(50)]
    preds,selections=[],[]
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize_models) as pool:
        for i,result in enumerate(pool.map(model_worker,work),1):
            preds.extend(result['predictions']);selections.extend(result['selection'])
            if i%25==0: print(f'Fresh nested primary/diagnosis fits {i}/450',flush=True)
    p=pd.DataFrame(preds).sort_values(['outcome','model_family','repeat','participant_index'])
    s=pd.DataFrame(selections).sort_values(['outcome','model_family','repeat','outer_fold','representation','C'])
    csv(p,PA/'ALL_MODEL_PREDICTIONS.csv');csv(s,PA/'ALL_INNER_SELECTION.csv')
    csv(s[s.outcome.eq('group')],PA/'PRIMARY_INNER_SELECTION.csv')
    wide=PRIMARY.merge_predictions({m:p[p.model_family.eq(m)] for _,_,m in modes if not m.endswith('_diagnosed')})
    csv(wide,PA/'PRIMARY_OUTER_PREDICTIONS.csv')

def exploratory():
    data=load_data()
    phase={'ids':data['ids']}
    predictions=[]
    for family in ('E3','E4','E5'):
        frame=pd.read_csv(EB/f'{family}_RAW_FEATURES.csv')
        if family=='E5':
            columns=[f'{region}_{band}_{q}' for band in BATTERY.BAND_ORDER for region in BATTERY.REGIONS for q in ('occupancy','median_high_state_duration_seconds','switching_rate_per_minute')]
            assert len(set(columns))==27 and set(columns)==set(frame.columns)-{'participant_id'}
            frame=frame[['participant_id']+columns]
        phase[family.lower()+'_frame']=frame
        def compute():
            failures=[]
            p,f,s=BATTERY.run_family(family,phase,PRIMARY,data['old'],data['folds'],failures)
            assert not failures,failures
            return [p,f,s]
        p,f,s=checkpoint('exploratory',family,[digest(RUN/'FEATURE_MANIFEST.json'),digest(PA/'PRIMARY_OUTER_PREDICTIONS.csv')],compute)
        predictions.extend(p)
        csv(pd.DataFrame(s),EB/f'{family}_INNER_SELECTION.csv')
        csv(pd.DataFrame(f),EB/f'{family}_FOLD_RESULTS.csv')
        print(f'Fresh {family}: 50 outer fits',flush=True)
    csv(pd.DataFrame(predictions),EB/'OUTER_PREDICTIONS.csv')

def initialize_posthoc():
    bind()
    core.initialize(load_data())

def posthoc_worker(index):
    return checkpoint('posthoc',index,[digest(RUN/'FEATURE_MANIFEST.json'),digest(PA/'PRIMARY_OUTER_PREDICTIONS.csv')],lambda:core.outer_job(core.DATA['folds']['outer'][index]))

def posthoc(jobs):
    data=load_data()
    rows=[[],[],[],[]];retries=0
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize_posthoc) as pool:
        for i,result in enumerate(pool.map(posthoc_worker,range(50)),1):
            for j in range(4): rows[j].extend(result[j])
            retries+=result[4]
            if i%5==0: print(f'Fresh post-hoc fits {i}/50 x 20 models',flush=True)
    core.postprocess(data,*[pd.DataFrame(r).sort_values(['model','repeat','outer_fold']) for r in rows],retries)
    dump(core.RESULT/'RUN_COMPLETE.json',{'fresh_feature_manifest':digest(RUN/'FEATURE_MANIFEST.json'),'completed_utc':datetime.now(timezone.utc).isoformat()})

def signal_worker(payload):
    bind()
    return checkpoint('signal_features',payload[0],digest(RUN/'FEATURE_MANIFEST.json'),lambda:signal.participant_job(payload))

def initialize_signal():
    bind()
    with np.load(signal.OUT/'surrogate_features.npz',allow_pickle=False) as archive:
        data=load_data()
        assert archive['participant_id'].tolist()==data['ids']
        signal.initialize_eval((data,archive['features']))

def signal_fit_worker(index):
    return checkpoint('signal_models',index,[digest(RUN/'FEATURE_MANIFEST.json'),digest(signal.OUT/'surrogate_features.npz')],lambda:signal.evaluate_seed(index))

def controls(jobs):
    signal.synthetic_check()
    data=load_data();ids=data['ids']
    roster=pd.read_csv(S1/'BLIND_COHORT_MANIFEST.csv').set_index('participant_id')
    matrix=data['base'].set_index('participant_id')
    payloads=[(pid,roster.loc[pid].to_dict(),float(matrix.loc[pid,'age']),matrix.loc[pid,STAGE.EEG_FEATURES].to_numpy(float),data['e4'][i],data['qc'].loc[pid].to_dict()) for i,pid in enumerate(ids)]
    array=np.full((20,96,9),np.nan);records=[]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i,result in enumerate(pool.map(signal_worker,payloads),1):
            pid,values,record=result
            array[:,ids.index(pid),:]=values;records.append(record)
            if i%10==0 or i==96: print(f'Fresh phase controls {i}/96',flush=True)
    assert np.isfinite(array).all()
    np.savez_compressed(signal.OUT/'surrogate_features.npz',participant_id=np.array(ids),feature_names=np.array(signal.NAMES),features=array)
    dump(signal.OUT/'reconstruction_and_seeds.json',records)
    csv(pd.DataFrame([{k:v for k,v in r.items() if k!='seeds'} for r in records]),signal.OUT/'reconstruction_checks.csv')
    rows=[];selection=[];retries=0
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize_signal) as pool:
        for i,(p,s,r) in enumerate(pool.map(signal_fit_worker,range(20)),1):
            rows.extend(p);selection.extend(s);retries+=r
            print(f'Fresh phase-control nested fits {i}/20',flush=True)
    pred=pd.DataFrame(rows).sort_values(['surrogate','repeat','outer_fold','participant_id'])
    original=pd.read_csv(core.RESULT/'outer_predictions.csv')
    combined=pd.concat([original[original.model.isin(['M2','E4'])],pred],ignore_index=True)
    comparisons=[]
    for i in range(20):
        name=f'surrogate_{i:02d}'
        comparisons.extend([(f'M2_vs_{name}','M2',name),(f'{name}_vs_observed_E4',name,'E4')])
    rr,ss=core.summarize_predictions(combined,comparisons)
    for frame,name in [(pred,'outer_predictions'),(pd.DataFrame(selection),'inner_selection'),(rr,'repeat_results'),(ss,'comparison_summary')]: csv(frame,signal.OUT/(name+'.csv'))
    dump(signal.OUT/'RUN_COMPLETE.json',{'fresh_feature_manifest':digest(RUN/'FEATURE_MANIFEST.json'),'convergence_retries':retries,'completed_utc':datetime.now(timezone.utc).isoformat()})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['features','models','exploratory','posthoc','controls','all'],default='all')
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.jobs<=2: parser.error('Fresh pipeline allows one or two CPU workers')
    assert not RUN.is_relative_to(ROOT) and not RUN.is_relative_to(RAW)
    bind();signature=run_contract()
    phases=['features','models','exploratory','posthoc','controls'] if args.phase=='all' else [args.phase]
    for phase in phases:
        print('FRESH PHASE: '+phase,flush=True)
        if phase=='features': features(args.jobs,signature)
        elif phase=='models': models(args.jobs)
        elif phase=='exploratory': exploratory()
        elif phase=='posthoc': posthoc(args.jobs)
        elif phase=='controls': controls(args.jobs)
        dump(RUN/('FRESH_COMPLETED_'+phase+'.json'),{'contract_sha256':signature,'completed_utc':datetime.now(timezone.utc).isoformat(),'reference_inputs_used':False})
    flush_access()

if __name__=='__main__': main()
