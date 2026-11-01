"""Portable orchestration around preserved analysis packages.

The statistical implementations, plans, seeds and reference results are copied
unchanged into the output workspace. Only path fields in copied manifests are
normalized. Legacy import-time runtime writes and verifier/display writes are
therefore confined to that disposable workspace. No analysis source is patched.
Saved-result verification is distinct from predictive model refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ('check', 'verify-results', 'assets', 'replay-transfer',
            'replay-empirical', 'replay-simulation', 'replay-adult', 'raw-pipeline')
FIGURES = {
    'Figure 1': 'repeat_increments.pdf',
    'Figure 2': 'acquisition_phase_controls.pdf',
    'Figure 3': 'component_increments.pdf',
    'Figure 4': 'device_transfer.pdf',
    'Figure 5': 'independent_adult_increment.pdf',
    'Figure S1': 'calibration.pdf',
    'Figure S2': 'participant_changes.pdf',
    'Figure S3': 'coefficients.pdf',
    'Figure S4': 'model_family_sensitivity.pdf',
    'Figure S5': 'learning_curves.pdf',
    'Figure S6': 'controlled_simulations.pdf',
    'Figure S7': 'eegnet_comparison.pdf',
}


class IntegrityError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def relative_name(value: str) -> str:
    """Accept either stored separator; never accept drives or traversal."""
    if not isinstance(value, str) or not value or '\x00' in value:
        raise IntegrityError('Invalid manifest path')
    win = PureWindowsPath(value)
    name = value.replace('\\', '/')
    parts = name.split('/')
    if win.drive or name.startswith('/') or any(p in ('', '.', '..') for p in parts):
        raise IntegrityError('Manifest path must be a simple relative path: ' + value)
    return PurePosixPath(*parts).as_posix()


def normalized_manifest(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise IntegrityError('Expected a relative-path to SHA-256 mapping')
    result = {}
    for key, sha in value.items():
        name = relative_name(key)
        if name in result:
            raise IntegrityError('Duplicate normalized manifest path: ' + name)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise IntegrityError('Invalid SHA-256 for ' + name)
        result[name] = sha
    return result


def verify_manifest(root: Path, mapping: dict[str, str], exact: bool = False) -> int:
    root = root.resolve()
    mapping = normalized_manifest(mapping)
    for name, expected in mapping.items():
        p = root / name
        if not p.resolve().is_relative_to(root) or p.is_symlink():
            raise IntegrityError('Manifest path escapes package: ' + name)
        if not p.is_file() or digest(p) != expected:
            raise IntegrityError('Missing or changed input: ' + name)
    if exact:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        expected = set(mapping) | {'FILE_MANIFEST.json'}
        if actual != expected:
            raise IntegrityError('Unmanifested/missing package files: ' + repr(sorted(actual ^ expected)))
    return len(mapping)


def directory_inventory(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir())


def overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def validate_output(root: Path, output: Path, dataset: Path | None = None) -> Path:
    output = output.resolve()
    if overlap(output, root):
        raise IntegrityError('Output must not overlap the source package in either direction')
    if dataset is not None and overlap(output, dataset):
        raise IntegrityError('Output must not overlap the raw dataset in either direction')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntegrityError('Output must be new or empty; checkpoints/configuration are never reused')
    return output


def input_checks(root: Path) -> dict[str, Any]:
    r, u, d = root/'reference_release', root/'extensions/utility', root/'extensions/device_transfer'
    result = {'release_files': verify_manifest(r, read_json(r/'RELEASE_MANIFEST.json')['sha256'])}
    result['utility_files'] = verify_manifest(u, read_json(u/'PACKAGE_MANIFEST.json'))
    for name, package, manifest_path in [
        ('utility', u, u/'audit/INPUT_MANIFEST.json'),
        ('transfer', d, d/'analysis/results/INPUT_MANIFEST.json'),
    ]:
        result[name+'_release_dependencies'] = verify_manifest(r, read_json(manifest_path)['files'])
        lock = read_json(package/'audit/PLAN_LOCK.json')
        if digest(package/'ANALYSIS_PLAN.txt') != lock.get('sha256', lock.get('plan_sha256')):
            raise IntegrityError(name + ' plan lock mismatch')
    if digest(u/'EXTERNAL_ANALYSIS_PLAN.txt') != read_json(u/'audit/EXTERNAL_PLAN_LOCK.json')['sha256']:
        raise IntegrityError('Adult plan lock mismatch')
    source = read_json(u/'external/DOWNLOAD_MANIFEST.json')
    result['adult_source_files'] = verify_manifest(u/'external', {v['name']: v['sha256'] for v in source})
    e = root/'extensions/eegnet'
    if e.exists():
        result['eegnet_locked_sources'] = verify_manifest(e, read_json(e/'PLAN_LOCK.json')['scientific_files'])
        result['eegnet_reference_inputs'] = verify_manifest(r, read_json(e/'WINDOW_INPUT_MANIFEST.json')['reference_inputs'])
    return result


def computation_fingerprint(mapping: dict[str, str]) -> str:
    kept = {k: v for k, v in mapping.items() if k.startswith(('reference_release/', 'extensions/', 'tools/'))
            or k in ('reproduce.py', 'requirements.txt', '.python-version')}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stage_workspace(root: Path, output: Path) -> Path:
    workspace = output/'workspace'
    shutil.copytree(root, workspace)
    changes = []
    for rel in ['extensions/utility/audit/INPUT_MANIFEST.json',
                'extensions/device_transfer/analysis/results/INPUT_MANIFEST.json']:
        p = workspace/rel
        data = read_json(p)
        data['files'] = normalized_manifest(data['files'])
        for key in ('release_root', 'repository'):
            if key in data:
                data[key] = str(workspace/'reference_release')
        write_json(p, data)
        changes.append({'path': rel, 'changes': 'Normalize copied manifest file keys and local release path only'})
    write_json(output/'WORKSPACE_ROUTING.json', {
        'preserved_source_modified': False, 'analysis_code_modified': False,
        'plans_seeds_reference_results_modified': False, 'copied_metadata_adjustments': changes,
        'legacy_writes': 'All legacy runtime, report, verifier and plot writes occur below this output/workspace',
        'workspace_manifest_note': 'The copied root/utility manifests describe original bytes; normalized copied input metadata is deliberately different. Original manifests are verified before copying.'})
    return workspace


def child_env(output: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    for key in ('PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE'):
        env[key] = '1'
    env['MPLBACKEND'] = 'Agg'
    for key, sub in [('MPLCONFIGDIR', 'cache/matplotlib'), ('XDG_CACHE_HOME', 'cache'),
                     ('TMP', 'tmp'), ('TEMP', 'tmp'), ('TMPDIR', 'tmp')]:
        path = output/sub; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    return env


def execute(script: Path, arguments: list[str], output: Path, tag: str) -> None:
    log = output/'logs'/f'{tag}.txt'; log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run([sys.executable, '-B', str(script), *map(str, arguments)],
                                cwd=script.parent, env=child_env(output), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{tag} failed (exit {result.returncode}); see {log}')


def read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path, float_precision='round_trip')


def verify_saved_results(root: Path, output: Path) -> dict[str, Any]:
    """Re-score distributed probabilities and replay conditional resampling; no classifier fits."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    r, u, d = root/'reference_release', root/'extensions/utility', root/'extensions/device_transfer/analysis/results'
    maximum = {}; counts = {}
    def near(a, b, category):
        error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        maximum[category] = max(maximum.get(category, 0.), error)
        if error > 1e-10:
            raise IntegrityError(f'{category} arithmetic mismatch: {error}')
    def loss(y, p):
        p = np.clip(p, 1e-8, 1-1e-8); return -y*np.log(p)-(1-y)*np.log1p(-p)
    def metrics(y, p):
        return dict(logloss=loss(y,p).mean(), auroc=roc_auc_score(y,p), brier=np.mean((y-p)**2))
    primary = read_csv(r/'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv')
    labels = read_csv(r/'audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv').set_index('participant_id')
    if len(primary) != 960 or primary.duplicated(['repeat','participant_id']).any():
        raise IntegrityError('Primary prediction coverage mismatch')
    if not np.array_equal(primary.y.to_numpy(), labels.loc[primary.participant_id,'group_y'].to_numpy()):
        raise IntegrityError('Primary outcome linkage mismatch')
    fold = read_csv(r/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv')
    for (rep,k), f in fold.groupby(['repeat','outer_fold']):
        outer = f[f.level=='outer']; tr=set(outer[outer.role=='train'].participant_id); te=set(outer[outer.role=='test'].participant_id)
        if tr & te or tr|te != set(labels.index): raise IntegrityError('Outer split mismatch')
        if te != set(primary[(primary.repeat==rep)&(primary.outer_fold==k)].participant_id): raise IntegrityError('Test roster mismatch')
        for _, inn in f[f.level=='inner'].groupby('inner_fold'):
            a=set(inn[inn.role=='train'].participant_id); b=set(inn[inn.role.isin(['valid','validation'])].participant_id)
            if a&b or a|b != tr: raise IntegrityError('Inner split mismatch')
    reference=read_csv(r/'audit/preregistered_analysis/PRIMARY_REPEAT_SUMMARY.csv').set_index('repeat')
    for rep,f in primary.groupby('repeat'):
        for col, label in [('p_m2','m2'),('p_adaptive_m3','m3')]:
            for k,v in metrics(f.y.to_numpy(),f[col].to_numpy()).items(): near(v,reference.loc[rep,label+'_'+k],'primary')
    counts['primary_prediction_rows']=len(primary); counts['outer_partitions']=50; counts['inner_partitions']=250
    for family,name in [('e0','E0_CORE_SENSITIVITY.csv'),('e1','FIXED_E1_SECONDARY.csv'),('e2','FIXED_E2_SECONDARY.csv')]:
        saved=read_csv(r/'audit/preregistered_analysis'/name).set_index('repeat')
        for rep,f in primary.groupby('repeat'):
            for k,v in metrics(f.y.to_numpy(),f['p_'+family].to_numpy()).items():near(v,saved.loc[rep,'representation_'+k],'fixed_spectral')
    counts['fixed_spectral_repeat_rows']=30
    for predpath,resultpath in [
        ('audit/exploratory_eeg_battery/OUTER_PREDICTIONS.csv','audit/exploratory_eeg_battery/REPEAT_LEVEL_RESULTS.csv'),
        ('results/e5/OUTER_PREDICTIONS.csv','results/e5/REPEAT_RESULTS.csv'),
    ]:
        predictions=read_csv(r/predpath);saved=read_csv(r/resultpath).set_index(['family','repeat'])
        for (family,rep),f in predictions.groupby(['family','repeat']):
            if len(f)!=96 or f.participant_id.nunique()!=96:raise IntegrityError('Exploratory prediction coverage mismatch')
            base=primary[primary.repeat==rep].set_index('participant_id').loc[f.participant_id]
            near(f.p_m2.to_numpy(),base.p_m2.to_numpy(),'shared_baseline')
            for k,v in metrics(f.y.to_numpy(),f.p_eeg.to_numpy()).items():near(v,saved.loc[(family,rep),'eeg_'+k],'exploratory')
    counts['E3_E4_E5_repeat_rows']=30
    post=read_csv(r/'expected/posthoc/results/outer_predictions.csv')
    def contrasts(pred, table, category):
        for _,row in table.iterrows():
            a=pred[pred.model==row.baseline]; b=pred[pred.model==row.augmented]
            paired=a.merge(b,on=['repeat','outer_fold','participant_id','y'],suffixes=('_0','_1'),validate='one_to_one')
            if len(paired)!=960: raise IntegrityError('Contrast coverage mismatch')
            vals=[]
            for _,f in paired.groupby('repeat'):
                m0=metrics(f.y.to_numpy(),f.p_0.to_numpy());m1=metrics(f.y.to_numpy(),f.p_1.to_numpy())
                vals.append(dict(baseline_logloss=m0['logloss'],augmented_logloss=m1['logloss'],delta_ll=m0['logloss']-m1['logloss'],delta_brier=m0['brier']-m1['brier'],delta_auroc=m1['auroc']-m0['auroc']))
            vals=pd.DataFrame(vals)
            for k,v in vals.mean().items(): near(v,row[k],category)
            if sum(vals.delta_ll>0)!=row.positive_repeats: raise IntegrityError('Positive repeat count mismatch')
        counts[category+'_contrasts']=len(table)
    contrasts(post,read_csv(r/'expected/posthoc/results/comparison_summary.csv'),'posthoc')
    phase=read_csv(r/'expected/posthoc/signal_controls/outer_predictions.csv')
    contrasts(pd.concat([phase,post[post.model.isin(['M2','E4'])]]),read_csv(r/'expected/posthoc/signal_controls/comparison_summary.csv'),'phase')
    empirical=[];splits={}
    for path in sorted((u/'results/empirical_v2').iterdir()):
        if not path.is_dir(): continue
        s=read_json(path/'SPLIT.json');splits[(s['repeat'],s['outer_fold'],s['fraction'],s['adjustment'])]=s
        tr=set(s['train_ids']);te=set(s['test_ids'])
        if tr&te: raise IntegrityError('Extension train/test overlap')
        for inn in s['inner']:
            a,b=set(inn['train_ids']),set(inn['validation_ids'])
            if a&b or a|b!=tr: raise IntegrityError('Extension inner split mismatch')
        f=read_csv(path/'PREDICTIONS.csv');empirical.append(f)
        for _,g in f.groupby(['learner','model']):
            if set(g.participant_id)!=te or g.participant_id.duplicated().any(): raise IntegrityError('Extension prediction roster mismatch')
        tuning=read_csv(path/'TUNING.csv');choices=read_csv(path/'SELECTOR_CHOICES.csv')
        for _,row in choices.iterrows():
            order=['regional_spectral','expanded_spectral','covariance'];t=tuning[(tuning.learner==row.learner)&tuning.model.isin(order)]
            best=min(t.to_dict('records'),key=lambda v:(round(v['inner_logloss'],12),order.index(v['model']),v['candidate']))
            if (best['model'],best['candidate'])!=(row.selected_representation,row.candidate): raise IntegrityError('Selector choice mismatch')
            a=f[(f.learner==row.learner)&(f.model==row.selected_representation)].set_index('participant_id');b=f[(f.learner==row.learner)&(f.model=='spectral_covariance_selector')].set_index('participant_id')
            near(a.p,b.loc[a.index].p,'selector')
    if len(splits)!=200: raise IntegrityError('Expected 200 empirical tasks')
    for rep in range(10):
        for k in range(5):
            a,b,c=[splits[(rep,k,f,0)] for f in [.5,.75,1.]]
            if not set(a['train_ids'])<set(b['train_ids'])<set(c['train_ids']): raise IntegrityError('Learning subsets not nested')
            if not set(a['test_ids'])==set(b['test_ids'])==set(c['test_ids']): raise IntegrityError('Learning assessment changed')
    pred=pd.concat(empirical);saved=read_csv(u/'reports/EMPIRICAL_REPEAT_METRICS.csv').set_index(['fraction','adjustment','learner','model','repeat'])
    for key,f in pred.groupby(['fraction','adjustment','learner','model','repeat']):
        if len(f)!=96 or f.participant_id.nunique()!=96: raise IntegrityError('Repeat coverage mismatch')
        for k,v in metrics(f.y.to_numpy(),f.p.to_numpy()).items(): near(v,saved.loc[key,k],'empirical')
    counts.update(empirical_tasks=200,empirical_prediction_rows=len(pred),empirical_repeat_metric_rows=len(saved))
    increments=read_csv(u/'reports/EMPIRICAL_REPEAT_INCREMENTS.csv');summaries=read_csv(u/'reports/EMPIRICAL_SUMMARY.csv')
    for _,row in summaries.iterrows():
        f=increments
        for key in ['fraction','adjustment','learner','model']:f=f[f[key]==row[key]]
        if len(f)!=10:raise IntegrityError('Empirical summary coverage mismatch')
        for key in ['baseline_logloss','logloss','baseline_auroc','auroc','baseline_brier','brier','delta_logloss','delta_auroc','delta_brier']:near(f[key].mean(),row[key],'empirical_summary')
        if int(sum(f.delta_logloss>0))!=row.positive_repeats:raise IntegrityError('Empirical positive count mismatch')
    counts['empirical_summary_rows']=len(summaries)
    n=0;rows=0
    for path in sorted((u/'results/simulation').iterdir()):
        if not path.is_dir(): continue
        with np.load(path/'PREDICTIONS.npz',allow_pickle=False) as z:
            table=read_csv(path/'METRICS.csv')
            for _,row in table.iterrows():
                p=z[f'{row.assessment}__{row.learner}__{row.model}']
                for k,v in metrics(z['test_y'],p).items(): near(v,row[k],'simulation')
            rows+=len(table);n+=1
    if n!=240: raise IntegrityError('Expected 240 simulation tasks')
    counts.update(simulation_tasks=n,simulation_metric_rows=rows)
    increments=read_csv(u/'reports/SIMULATION_INCREMENTS.csv');summaries=read_csv(u/'reports/SIMULATION_SUMMARY.csv')
    for _,row in summaries.iterrows():
        f=increments
        for key in ['scenario','assessment','learner','train_n']:f=f[f[key]==row[key]]
        if len(f)!=20:raise IntegrityError('Simulation summary coverage mismatch')
        for key in ['baseline_logloss','eeg_logloss','delta_logloss','delta_auroc','delta_brier']:near(f[key].mean(),row[key],'simulation_summary')
        near(f.delta_logloss.std(ddof=1)/np.sqrt(20),row.mcse_delta_logloss,'simulation_mcse')
        if int(sum(f.delta_logloss>0))!=row.positive_replicates:raise IntegrityError('Simulation positive count mismatch')
    counts['simulation_summary_rows']=len(summaries)
    # Reconstruct the documented, shared class-stratified test resampling indices.
    for mode,base,seed,reps in [('device',d,20261103,5000),('adult',u/'results/external',20261137,10000)]:
        p=read_csv(base/'PREDICTIONS.csv');rng=np.random.default_rng(seed)
        if mode=='device':
            summaries=read_csv(base/'CONTRASTS_ALL.csv').set_index(['direction','model']);saved_draws=read_csv(base/'CONDITIONAL_DRAWS.csv');groupby=['direction'];rowkey='participant_id';nprincipal=0
        else:
            summaries=read_csv(base/'INCREMENTS.csv').set_index(['baseline','learner']);saved_draws=read_csv(base/'CONDITIONAL_RESAMPLES.csv');groupby=['baseline','learner'];rowkey='source_test_row';nprincipal=0
        shared_ix=None
        for key,f in p.groupby(groupby,sort=(mode=='adult')):
            if mode=='device': key=key[0] if isinstance(key,tuple) else key
            b=f[f.model=='baseline'].set_index(rowkey).sort_index();y=b.y.to_numpy(int);p0=b.p.to_numpy();n0=int(sum(y==0))
            if mode=='device' or shared_ix is None:
                ix=np.column_stack([rng.choice(np.flatnonzero(y==j),(reps,int(sum(y==j))),replace=True) for j in [0,1]])
                shared_ix=ix
            else: ix=shared_ix
            for model,g in f.groupby('model',sort=False):
                if model=='baseline': continue
                p1=g.set_index(rowkey).loc[b.index].p.to_numpy();dl=loss(y,p0)-loss(y,p1)
                sk=(key,model) if mode=='device' else key;row=summaries.loc[sk]
                ll='delta_ll' if mode=='device' else 'delta_logloss'
                near(dl.mean(),row[ll],mode+'_point')
                if mode=='device' and model not in ['spectral_covariance_selector','entropy_hjorth']:continue
                def auc(z):
                    z=z[ix];dif=z[:,n0:,None]-z[:,None,:n0];return ((dif>0)+.5*(dif==0)).mean((1,2))
                vals={ll:dl[ix].mean(1),'delta_brier':((y-p0)**2-(y-p1)**2)[ix].mean(1),'delta_auroc':auc(p1)-auc(p0)}
                if mode=='device':
                    vals['equal_class_delta_ll']=(dl[ix[:,:n0]].mean(1)+dl[ix[:,n0:]].mean(1))/2
                    dr=saved_draws[(saved_draws.direction==key)&(saved_draws.model==model)].sort_values('draw')
                else:dr=saved_draws[(saved_draws.baseline==key[0])&(saved_draws.learner==key[1])].sort_values('bootstrap')
                for k,v in vals.items():
                    near(v,dr[k].to_numpy(),mode+'_draws');near(np.quantile(v,[.025,.975]),[row[k+'_conditional_low'],row[k+'_conditional_high']],mode+'_quantiles')
                nprincipal+=1
        counts[mode+'_conditional_draw_rows']=nprincipal*reps
    report={'mode':'saved-result arithmetic and conditional-resampling replay','predictive_models_refitted':False,
            'raw_data_processed':False,'arithmetic_tolerance':1e-10,'counts':counts,'max_abs_errors':maximum,
            'limits':['Historical diagnosis-sensitivity individual predictions are absent.','Fixed fitted predictions do not estimate complete-training uncertainty.','File hashes do not establish independent chronological provenance.']}
    write_json(output/'RESULT_VERIFICATION.json',report)
    return report


def compare_csv(reference: Path, generated: Path, tolerance: float = 1e-7) -> float:
    import numpy as np
    import pandas as pd
    a,b=read_csv(reference),read_csv(generated)
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:
        raise IntegrityError('Replay table schema mismatch: '+reference.name)
    maximum=0.
    for k in a:
        if pd.api.types.is_numeric_dtype(a[k]) and pd.api.types.is_numeric_dtype(b[k]):
            if not np.allclose(a[k],b[k],rtol=0,atol=tolerance,equal_nan=True):raise IntegrityError('Replay numeric mismatch: '+reference.name+'/'+k)
            if len(a) and not a[k].isna().all():maximum=max(maximum,float(np.nanmax(abs(a[k].astype(float)-b[k].astype(float)))))
        elif not a[k].fillna('').equals(b[k].fillna('')):
            raise IntegrityError('Replay categorical/selection mismatch: '+reference.name+'/'+k)
    return maximum


def verify_replay(workspace: Path, run: Path, command: str, output: Path) -> None:
    import numpy as np
    refs=workspace/'extensions/utility/results';records=[]
    if command in ('replay-empirical','replay-simulation'):
        kind='empirical_v2' if command=='replay-empirical' else 'simulation'
        for done in sorted((run/'results'/kind).glob('*/DONE.json')):
            new=done.parent;old=refs/kind/new.name
            names=['TUNING.csv','INNER_FOLD_SCORES.csv']+(['PREDICTIONS.csv','SELECTOR_CHOICES.csv'] if kind=='empirical_v2' else ['METRICS.csv','ORACLE.csv'])
            errors={name:compare_csv(old/name,new/name) for name in names}
            if read_json(old/'SPLIT.json')!=read_json(new/'SPLIT.json'):raise IntegrityError('Replay split mismatch')
            if kind=='simulation':
                with np.load(old/'PREDICTIONS.npz',allow_pickle=False) as a,np.load(new/'PREDICTIONS.npz',allow_pickle=False) as b:
                    for key in a.files:
                        if key not in b.files or not np.allclose(a[key],b[key],rtol=0,atol=1e-7):raise IntegrityError('Simulation array mismatch: '+key)
                    errors['retained_probability_arrays']=max(float(np.max(abs(a[k]-b[k]))) for k in a.files)
            records.append({'task':new.name,'max_abs_errors':errors})
    else:
        old=refs/'external';new=run/'results/external'
        names=['PREDICTIONS.csv','TUNING.csv','INNER_FOLD_SCORES.csv','METRICS.csv','INCREMENTS.csv','CONDITIONAL_RESAMPLES.csv','PARTICIPANT_INCREMENTS.csv','MISSINGNESS.csv']
        records.append({'task':'adult','max_abs_errors':{name:compare_csv(old/name,new/name) for name in names}})
        for name in ['SPLIT.json','DROPPED_COLUMNS.json']:
            if read_json(old/name)!=read_json(new/name):raise IntegrityError('Adult replay configuration mismatch: '+name)
    if not records:raise IntegrityError('No completed replay tasks')
    write_json(output/'REPLAY_COMPARISON.json',{'mode':'new model fitting from frozen features','raw_data_reextracted':False,'absolute_tolerance':1e-7,'tasks':records})


def current_table_crossrefs(assets: Path) -> list[dict[str, str]]:
    """Update one historical section pointer in the output, never the generator."""
    path=assets/'tables/all_comparisons.tex'
    old='(Section~S9)';new=r'(Section~\ref{sec:s-data})'
    text=path.read_text(encoding='utf-8')
    if text.count(old)!=1:raise IntegrityError('Unexpected original comparison-table section pointer')
    path.write_text(text.replace(old,new),encoding='utf-8')
    return [{'file':'tables/all_comparisons.tex','original':old,'current':new,
             'scope':'Output-only section cross-reference; numerical cells and caption unchanged'}]


def build_assets(workspace: Path, output: Path) -> None:
    r,u,d=workspace/'reference_release',workspace/'extensions/utility',workspace/'extensions/device_transfer'
    execute(r/'reproduce.py',['figures','--output',output/'original_displays'],output,'original_displays')
    assets=output/'assets';assets.mkdir()
    for folder in ('figures','tables'):
        shutil.copytree(output/'original_displays/display_assets'/folder,assets/folder)
    crossrefs=current_table_crossrefs(assets)
    (assets/'audit').mkdir()
    execute(d/'analysis/code/build_transfer_assets.py',['--results',d/'analysis/results','--manuscript',assets],output,'transfer_displays')
    execute(u/'code/summarize_extension.py',['empirical','simulations','external'],output,'utility_summaries')
    execute(u/'code/build_extension_assets.py',['--manuscript',assets],output,'utility_displays')
    duration=workspace/'tools/clean_duration_summary.py'
    if duration.is_file():
        execute(duration,['--release-root',r,'--output',assets/'quality_summary'],output,'duration_summary')
    reporting=workspace/'tools/build_reporting_assets.py'
    if reporting.is_file():
        execute(reporting,['--repository',workspace,'--output',assets/'reporting'],output,'reporting_assets')
    e=workspace/'extensions/eegnet'
    if e.exists():
        execute(e/'code/summarize.py',['--results',e/'results','--release',r,'--out',assets/'eegnet_reports'],output,'eegnet_summaries')
        execute(e/'code/build_assets.py',['--reports',assets/'eegnet_reports','--assets',assets],output,'eegnet_displays')
    for label,name in FIGURES.items():
        if not (assets/'figures'/name).is_file():raise IntegrityError('Missing current article asset: '+label)
    write_json(assets/'ASSET_MAP.json',{'figures':FIGURES,'table_files':sorted(p.name for p in (assets/'tables').glob('*.tex')),
        'source':'Preserved generators and saved reference results; no predictive fits',
        'numbering':'Current article figure identities; legacy console messages are retained only in logs and do not determine numbering.',
        'current_section_crossrefs':crossrefs,
        'additional_reporting_files':sorted(p.relative_to(assets).as_posix() for p in (assets/'reporting').rglob('*') if p.is_file()) if (assets/'reporting').exists() else [],
        'original_display_numeric_checks':read_json(output/'original_displays/display_assets/DISPLAY_CHECKS.json')})


def run(args: argparse.Namespace, root: Path = ROOT) -> None:
    root=root.resolve();dataset=args.dataset.resolve() if args.dataset else None
    if args.command=='raw-pipeline' and dataset is None:raise IntegrityError('raw-pipeline requires --dataset explicitly')
    if dataset is not None and (not dataset.is_dir() or not (dataset/'users_demographics.json').is_file()):raise IntegrityError('Dataset must contain users_demographics.json')
    if dataset is not None and overlap(dataset,root):raise IntegrityError('Raw dataset must be separate from the source package')
    output=validate_output(root,args.output,dataset)
    manifest_path=root/'FILE_MANIFEST.json'
    if not manifest_path.is_file():raise IntegrityError('FILE_MANIFEST.json is required; use the complete versioned package')
    mapping=normalized_manifest(read_json(manifest_path));verify_manifest(root,mapping,exact=True)
    checks=input_checks(root);before_dirs=directory_inventory(root);manifest_sha=digest(manifest_path)
    output.mkdir(parents=True,exist_ok=True)
    config={'command':args.command,'first_only':args.first_only,'workers':args.workers,'dataset':str(dataset) if dataset else None,
        'package_manifest_sha256':manifest_sha,'computation_fingerprint':computation_fingerprint(mapping),
        'python':platform.python_version(),'dependencies':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','threadpoolctl']},
        'cache_policy':'No resume or result cache; a nonempty output is always rejected.','input_checks':checks}
    write_json(output/'CONFIG.json',config)
    succeeded=False
    try:
        if args.command=='check':
            execute(root/'reference_release/reproduce.py',['check','--output',output/'release_check'],output,'release_check')
        elif args.command=='verify-results':
            verify_saved_results(root,output)
            execute(root/'reference_release/reproduce.py',['descriptive-resampling','--output',output/'fixed_oof'],output,'fixed_oof')
            execute(root/'reference_release/reproduce.py',['calibration-numerical','--output',output/'calibration'],output,'calibration')
        else:
            workspace=stage_workspace(root,output);r=workspace/'reference_release';u=workspace/'extensions/utility';d=workspace/'extensions/device_transfer';runpath=output/'run'
            if args.command=='assets':build_assets(workspace,output)
            elif args.command=='replay-transfer':
                execute(d/'analysis/code/device_transfer_portable.py',['--release-root',r,'--output',runpath],output,'transfer_refit')
                execute(d/'analysis/code/verify_saved_results.py',['--release-root',r,'--replay',runpath],output,'transfer_verify')
                shutil.copy2(d/'audit/ANALYSIS_VERIFICATION.json',output/'REPLAY_COMPARISON.json')
            elif args.command in ('replay-empirical','replay-simulation','replay-adult'):
                part={'replay-empirical':'empirical','replay-simulation':'simulation','replay-adult':'external'}[args.command]
                options=[part,'--release-root',r,'--output-root',runpath,'--workers',str(args.workers)]
                if args.first_only:options.append('--first-only')
                execute(u/'code/portable_reproduce.py',options,output,'utility_refit')
                verify_replay(workspace,runpath,args.command,output)
            else:
                execute(r/'reproduce.py',['raw-pipeline','--dataset',dataset,'--output',runpath,'--jobs',str(min(args.workers,2))],output,'raw_pipeline')
        succeeded=True
    finally:
        try:
            verify_manifest(root,mapping,exact=True)
            if digest(manifest_path)!=manifest_sha or directory_inventory(root)!=before_dirs:raise IntegrityError('Source manifest/directory inventory changed during execution')
            write_json(output/'SOURCE_UNCHANGED.json',{'files_verified':len(mapping),'directories_unchanged':True,'manifest_sha256':manifest_sha})
        except Exception as exc:
            write_json(output/'SOURCE_INTEGRITY_FAILURE.json',{'error':str(exc)})
            raise
        if succeeded:write_json(output/'COMPLETED.json',{'command':args.command,'first_only':args.first_only,'source_unchanged':True,'computation_fingerprint':config['computation_fingerprint']})
    print(f'Completed {args.command}; outputs: {output}')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=COMMANDS)
    parser.add_argument('--output',type=Path,required=True,help='New or empty directory, disjoint from source and raw dataset')
    parser.add_argument('--dataset',type=Path,help='Explicit raw BALLADEER dataset root; used only by raw-pipeline')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=1)
    parser.add_argument('--first-only',action='store_true',help='One diagnostic task for empirical/simulation only')
    args=parser.parse_args()
    if args.first_only and args.command not in ('replay-empirical','replay-simulation'):parser.error('--first-only applies only to empirical/simulation replays')
    if args.dataset and args.command!='raw-pipeline':parser.error('--dataset applies only to raw-pipeline')
    try:run(args)
    except (IntegrityError,RuntimeError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()
