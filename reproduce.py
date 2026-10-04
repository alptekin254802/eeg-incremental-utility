"""Portable, isolated reproduction commands for the manuscript companion."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent
SAVED_INPUT_STAGES=('raw-screen','raw-features','models','exploratory','e5','posthoc','signal-controls','summarize','figures','tables')
STAGES=('check','calibration',*SAVED_INPUT_STAGES,'raw-pipeline','all')
def verify():
    m=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,digest in m['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    return m
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--output',type=Path,required=True,help='Output directory outside the release and dataset')
    parser.add_argument('--dataset',type=Path,help='Directory containing users_demographics.json and UB participant folders')
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();assert args.jobs>0
    out=args.output.resolve();assert not out.is_relative_to(ROOT), 'Output must be outside the immutable release'
    data=args.dataset.resolve() if args.dataset else None
    if data: assert data.is_dir() and not out.is_relative_to(data) and (data/'users_demographics.json').is_file()
    before=verify();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(EEG_RELEASE_ROOT=str(ROOT),EEG_RUN_ROOT=str(out),EEG_JOBS=str(args.jobs),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    if data:env['EEG_DATASET_ROOT']=str(data)
    stages=list(SAVED_INPUT_STAGES) if args.stage=='all' else [args.stage]
    if any(s in stages for s in ('raw-screen','raw-features','models','signal-controls','raw-pipeline')):
        parser.error('--dataset is required for the requested stage') if data is None else None
    def run(script,*extra):
        subprocess.run([sys.executable,str(ROOT/'reproduce'/script),*extra],env=env,check=True)
    def expected_posthoc():
        if not (out/'posthoc/results/RUN_COMPLETE.json').exists():
            shutil.copytree(ROOT/'expected/posthoc',out/'posthoc',dirs_exist_ok=True)
            (out/'posthoc/STARTED_FROM_EXPECTED_RESULTS.json').write_text(json.dumps({'mode':'saved-result summary/figure regeneration; no model refit by this step'})+'\n')
    for stage in stages:
        print('STAGE: '+stage,flush=True)
        if stage=='check':
            subprocess.run([sys.executable,'-c',
                'import importlib.util,os; from pathlib import Path; p=Path(os.environ["EEG_RELEASE_ROOT"])/"audit/preregistered_analysis/run_preregistered_analysis.py"; s=importlib.util.spec_from_file_location("p",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); b,d=m.load_measurements(); assert len(b)==96; print("PASS: release manifest and 14 released measurement/estimator hashes; 96 participants")'],env=env,check=True)
        elif stage=='calibration':run('calibration.py')
        elif stage=='raw-screen':run('raw_screen.py')
        elif stage=='raw-features':run('verify_raw_features.py')
        elif stage=='raw-pipeline':run('raw_pipeline.py','--jobs',str(args.jobs))
        elif stage=='models':run('verify_models.py')
        elif stage=='exploratory':run('exploratory.py')
        elif stage=='e5':run('e5.py')
        elif stage=='posthoc':run('run_posthoc.py','--jobs',str(args.jobs))
        elif stage=='signal-controls':
            assert (out/'posthoc/results/RUN_COMPLETE.json').exists(),'Run posthoc first'
            run('run_signal_controls.py','--jobs',str(args.jobs))
        elif stage=='summarize':expected_posthoc();run('summarize_posthoc.py')
        elif stage=='figures':
            expected_posthoc();run('posthoc_figures.py')
            figures=out/'figures';figures.mkdir(exist_ok=True)
            for script in sorted((ROOT/'reproduce').glob('generate_figure*.py')):
                run(script.name,'--output',str(figures/script.stem.replace('generate_','')))
            mapping={'PH1_components_and_dimension':'Figure_S2_Components','PH2_device_quality_and_policy':'Figure_S3_Acquisition','PH3_participant_gains_losses':'Figure_S4_Participants','PH4_coefficient_stability':'Figure_S5_Coefficients','PH5_spectrum_preserving_controls':'Figure_6_Phase_Controls'}
            for p in (out/'posthoc/figures').glob('*'):
                shutil.copy2(p,figures/(mapping.get(p.stem,p.stem)+p.suffix))
        elif stage=='tables':expected_posthoc();run('tables.py');run('latex_tables.py')
    assert verify()==before
    (out/('COMPLETED_'+args.stage+'.json')).write_text(json.dumps({'stage':args.stage,'source_manifest_sha256':hashlib.sha256((ROOT/'RELEASE_MANIFEST.json').read_bytes()).hexdigest(),'source_manifest_unchanged':True,'dataset':str(data) if data else None,'python':sys.version},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
