"""Reconstruct the 126-to-98 measurement screen directly from source files."""
import os, json, importlib.util, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(os.environ['EEG_RELEASE_ROOT']); RAW=Path(os.environ['EEG_DATASET_ROOT']); OUT=Path(os.environ['EEG_RUN_ROOT'])/'raw_screen'
def load(path,name):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
quality=load('audit/cohort_screen/robots_quality_audit.py','quality')
stage=load('audit/stage1_blind_feature_extraction/run_stage1_extraction.py','stage')
quality.DATASET=RAW;quality.ROOT=RAW.parent
def worker(session):
    rows=[]
    for name in ('eeg','eye','behavior'):
        p=session[name+'_path'];before=hashlib.sha256(p.read_bytes()).hexdigest()
        row=getattr(quality,'parse_'+name)(p,True)
        row.update(participant_id=session['participant_id'],session_id=session['session_id'],source_file=str(p.resolve()),raw_sha256=before)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==before
        rows.append(row)
    return rows
def main():
    OUT.mkdir(exist_ok=True)
    # Discovery needs only participant membership; outcomes are not used in eligibility.
    records=json.loads((RAW/'users_demographics.json').read_text(encoding='utf-8'))
    sessions=quality.discover_sessions({r['user']:{} for r in records})
    complete=[s for s in sessions if all(s[k+'_path'] is not None for k in ('eeg','eye','behavior'))]
    assert len(complete)==126
    rows=[[],[],[]]
    with ProcessPoolExecutor(max_workers=int(os.environ['EEG_JOBS'])) as pool:
        for i,result in enumerate(pool.map(worker,complete),1):
            for j,row in enumerate(result):rows[j].append(row)
            if i%20==0:print(f'Measurement screen: {i}/126',flush=True)
    dest=OUT/'audit/cohort_screen';dest.mkdir(parents=True,exist_ok=True)
    for name,data in zip(('eeg','eye','behavior'),rows):pd.DataFrame(data).to_csv(dest/f'robots_{name}_quality.csv',index=False)
    cohort,summary=stage.reconstruct_cohort(OUT)
    expected=pd.read_csv(ROOT/'audit/stage1_blind_feature_extraction/BLIND_COHORT_MANIFEST.csv',dtype={'participant_id':str,'session_id':str})
    assert {(r['participant_id'],r['session_id']) for r in cohort}==set(zip(expected.participant_id,expected.session_id))
    pd.DataFrame([{k:str(v) for k,v in r.items() if not k.endswith('_path')} for r in cohort]).to_csv(OUT/'MEASUREMENT_ROSTER.csv',index=False)
    (OUT/'CHECKS.json').write_text(json.dumps({'raw_complete_sessions':126,'measurement_eligible':len(cohort),'roster_exact':True,'outcomes_used_for_eligibility':False,'summary':summary},indent=2)+'\n',encoding='utf-8')
    print('PASS: 126 raw complete sessions -> exact saved 98-person measurement roster',flush=True)
if __name__=='__main__':main()
