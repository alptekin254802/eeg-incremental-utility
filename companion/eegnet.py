"""Portable commands for the single exploratory EEGNet-like comparison."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
from tools.portable_pipeline import digest,read_json,verify_manifest,normalized_manifest,validate_output,input_checks,write_json,directory_inventory
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','train','verify','replay','synthetic-check'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',type=Path)
    ap.add_argument('--cache',type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--workers',type=int,choices=range(1,5),default=4)
    a=ap.parse_args();e=ROOT/'extensions/eegnet';r=ROOT/'reference_release'
    if a.command=='prepare' and a.dataset is None:ap.error('prepare requires --dataset')
    if a.command in ('train','replay') and a.cache is None:ap.error('training requires an explicitly reconstructed --cache')
    source=a.dataset if a.command=='prepare' else a.cache
    out=validate_output(ROOT,a.output,source)
    mapping=normalized_manifest(read_json(ROOT/'FILE_MANIFEST.json'));verify_manifest(ROOT,mapping,exact=True);input_checks(ROOT)
    directories=directory_inventory(ROOT);out.mkdir(parents=True,exist_ok=True)
    common=['--release',str(r)]
    if a.command=='prepare':
        command=[e/'code/prepare_windows.py',*common,'--raw',a.dataset,'--out',out/'windows','--workers',str(a.workers)]
    elif a.command in ('train','replay'):
        command=[e/'code/run_nested.py',*common,'--cache',a.cache,'--out',out/'results','--lock',e/'PLAN_LOCK.json','--device',a.device]
        if a.command=='replay':command+=['--fold','0']
    elif a.command=='verify':
        command=[e/'code/summarize.py',*common,'--results',e/'results','--out',out/'reports']
    else:
        command=[e/'code/test_isolation.py','--device',a.device,'--out',out/'SYNTHETIC_TESTS.json']
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1';env['MPLCONFIGDIR']=str(out/'mplconfig')
    with (out/'execution.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-B',*map(str,command)],stdout=log,stderr=subprocess.STDOUT,env=env)
    verify_manifest(ROOT,mapping,exact=True)
    if directory_inventory(ROOT)!=directories:raise RuntimeError('Source directory inventory changed')
    write_json(out/'SOURCE_UNCHANGED.json',dict(files_verified=len(mapping),manifest_sha256=digest(ROOT/'FILE_MANIFEST.json')))
    if result.returncode:raise RuntimeError('Command failed; retained execution.log records the failure')
    write_json(out/'COMPLETED.json',dict(command=a.command,device=a.device,scope={'verify':'Saved-result arithmetic and fixed-prediction resampling; no encoder fit','replay':'Six encoders for repeat 0 / outer fold 0; not all 300','train':'Complete planned 300-encoder comparison','prepare':'Frozen 96-participant raw-window reconstruction','synthetic-check':'Synthetic isolation and numerical execution checks'}[a.command]))
    print('Completed',a.command,'in',out)

if __name__=='__main__':main()
