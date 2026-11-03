"""Portable entry point with immutable dependency and intermediate-input checks."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def validate(release,transfer):
    for n,h in read(ROOT/'FILE_MANIFEST.json').items():
        if not (ROOT/n).is_file() or sha(ROOT/n)!=h:raise ValueError('Addendum file identity mismatch: '+n)
    for label,base in [('release',release),('transfer',transfer)]:
        for n,h in read(ROOT/'results/audit/INPUT_IDENTITIES.json')[label].items():
            if not (base/n).is_file() or sha(base/n)!=h:raise ValueError('Archived dependency identity mismatch: '+label+'/'+n)
    lock=read(ROOT/'results/audit/PLAN_LOCK.json')
    if sha(ROOT/'results/ANALYSIS_PLAN.txt')!=lock['plan_sha256']:raise ValueError('Plan identity mismatch')
    for part in ['age','bandwidth','extract','offset']:
        r=read(ROOT/'results'/part/'RUN.json')
        if r['code_sha256']!=sha(ROOT/'code/controls.py') or r['plan_sha256']!=lock['plan_sha256']:raise ValueError('Recorded computation provenance mismatch: '+part)
def initialize(out):
    out.mkdir(parents=True,exist_ok=False);(out/'audit').mkdir()
    for n in ['ANALYSIS_PLAN.txt','audit/PLAN_LOCK.json','audit/INPUT_IDENTITIES.json']:shutil.copy2(ROOT/'results'/n,out/n)
def call(args):subprocess.run([sys.executable,'-B',*map(str,args)],check=True)
def same_csv(a,b):
    x=pd.read_csv(a,float_precision='round_trip');y=pd.read_csv(b,float_precision='round_trip')
    if list(x.columns)!=list(y.columns) or x.shape!=y.shape:raise ValueError('CSV structure differs: '+str(a))
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):
            if not np.allclose(x[c],y[c],atol=1e-12,rtol=1e-12,equal_nan=True):raise ValueError('Numerical replay differs: '+c)
        elif not x[c].equals(y[c]):raise ValueError('Non-numerical replay differs: '+c)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['verify','init','age','extract','bandwidth','offset','summarize']);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--transfer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--raw',type=Path);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();validate(a.release,a.transfer)
    if a.operation=='init':initialize(a.output);print('New sensitivity workspace initialized; archived inputs verified.');return
    if a.operation=='verify':
        initialize(a.output)
        for part in ['age','bandwidth','extract']:shutil.copytree(ROOT/'results'/part,a.output/part)
        call([ROOT/'code/controls.py','offset','--release',a.release,'--transfer',a.transfer,'--output',a.output])
        call([ROOT/'code/summarize.py','--release',a.release,'--transfer',a.transfer,'--output',a.output])
        for folder in ['offset','summary']:
            for p in (ROOT/'results'/folder).glob('*.csv'):same_csv(p,a.output/folder/p.name)
        call([ROOT/'code/build_tables.py','--results',a.output,'--out',a.output/'tables'])
        for p in (ROOT/'tables').glob('*'):
            if sha(p)!=sha(a.output/'tables'/p.name):raise ValueError('Generated table differs: '+p.name)
        report=dict(status='PASS',scope='File identities; all saved-prediction scores and conditional draw replay; table regeneration. No predictive refitting and no raw-feature extraction.',tolerance=1e-12)
        (a.output/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    if read(a.output/'audit/PLAN_LOCK.json')!=read(ROOT/'results/audit/PLAN_LOCK.json') or sha(a.output/'ANALYSIS_PLAN.txt')!=sha(ROOT/'results/ANALYSIS_PLAN.txt'):raise ValueError('Workspace plan differs')
    if a.operation=='extract':
        if a.raw is None:raise ValueError('--raw must point to the permitted local BALLADEER dataset directory')
        for r in read(ROOT/'results/extract/RAW_RECONSTRUCTION.json'):
            p=(a.raw/r['raw_relative_path']).resolve()
            if not p.is_relative_to(a.raw.resolve()) or not p.is_file() or sha(p)!=r['raw_sha256']:raise ValueError('Raw input identity mismatch: '+r['raw_relative_path'])
    if a.operation=='bandwidth':
        for n in ['E4_LOWPASS30_FEATURES.csv','FEATURE_COMPARISON.csv']:same_csv(ROOT/'results/extract'/n,a.output/'extract'/n)
        if read(a.output/'extract/RUN.json')['code_sha256']!=sha(ROOT/'code/controls.py'):raise ValueError('Extracted features come from a different implementation')
    if a.operation=='summarize':call([ROOT/'code/summarize.py','--release',a.release,'--transfer',a.transfer,'--output',a.output]);call([ROOT/'code/build_tables.py','--results',a.output,'--out',a.output/'tables'])
    else:
        cmd=[ROOT/'code/controls.py',a.operation,'--release',a.release,'--transfer',a.transfer,'--output',a.output,'--workers',a.workers]
        if a.raw is not None:cmd+=['--raw',a.raw]
        call(cmd)
if __name__=='__main__':main()
