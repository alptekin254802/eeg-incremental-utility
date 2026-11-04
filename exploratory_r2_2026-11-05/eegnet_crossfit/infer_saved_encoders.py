"""CPU inference only; never trains an encoder or imports the head evaluator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
CODE=REPO/'companion/extensions/eegnet/code'
sys.path.insert(0,str(CODE))
from eegnet_model import Encoder,embeddings

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cache=Path(r'F:\proje\eeg_gain\private_work\eegnet_extension_20261101\window_cache')
    manifest=read(cache/'INPUT_MANIFEST.json')
    for name,digest in manifest['reference_inputs'].items():
        assert sha(REPO/name)==digest,name
    windows={}
    for item in manifest['participants']:
        path=cache/item['cache_file'];assert sha(path)==item['cache_sha256']
        windows[item['participant_id']]=np.load(path,allow_pickle=False,mmap_mode='r')
    out=HERE/'scores';out.mkdir(exist_ok=True)
    sources={};checks=[]
    for task in range(50):
        r,f=divmod(task,5);fold=REPO/f'companion/extensions/eegnet/results/r{r:02d}_f{f:02d}'
        outer=read(fold/'outer/fit.json');te=outer['assessment_ids'];tr=outer['train_ids']
        assert not set(te)&set(tr) and set(te)|set(tr)==set(windows)
        for j in range(5):
            source=fold/f'inner{j}';target=out/f'r{r:02d}_f{f:02d}_i{j}.npz'
            fit=read(source/'fit.json');iv=fit['assessment_ids'];it=fit['train_ids']
            assert not set(iv)&set(it) and set(iv)|set(it)==set(tr)
            assert not (set(iv)|set(it))&set(te)
            for name in ['fit.json','encoder.pt','embeddings.npz']:
                sources[str((source/name).relative_to(REPO))]=sha(source/name)
            if target.exists():
                continue
            with np.load(source/'embeddings.npz',allow_pickle=False) as z:
                ids=z['participant_id'].tolist();embed=z['embedding']
                mean=z['channel_mean'];scale=z['channel_scale']
            assert ids==it+iv
            weights=torch.load(source/'encoder.pt',map_location='cpu',weights_only=True)
            model=Encoder();model.load_state_dict(weights);model.eval()
            w=weights['head.weight'].numpy().astype(float).ravel()
            bias=float(weights['head.bias'].numpy().ravel()[0])
            zv=embed[[ids.index(pid) for pid in iv]]@w+bias
            zte=embeddings(model,windows,te,mean,scale,device='cpu')@w+bias
            assert np.isfinite(zv).all() and np.isfinite(zte).all()
            # Validate a complete held-out participant against its saved embedding.
            replay=embeddings(model,windows,iv[:1],mean,scale,device='cpu')
            replay_error=float(np.max(np.abs(replay-embed[[ids.index(iv[0])]])))
            assert replay_error<1e-4,(task,j,replay_error)
            np.savez_compressed(target,head_train_ids=np.array(iv),encoder_train_ids=np.array(it),
                test_ids=np.array(te),head_train_logits=zv,test_logits=zte,
                replay_max_embedding_error=np.array(replay_error))
            checks.append(dict(repeat=r,outer_fold=f,inner_fold=j,head_train_n=len(iv),
                               encoder_train_n=len(it),test_n=len(te),replay_max_error=replay_error))
        print(f'Frozen encoder inference {task+1}/50 outer folds',flush=True)
        write(HERE/'INFERENCE_PROGRESS.json',dict(completed_outer_folds=task+1))
    # Read all checkpoints so a resumed execution has a complete validation list.
    checks=[]
    for p in sorted(out.glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            assert not set(z['head_train_ids'])&set(z['encoder_train_ids'])
            assert not set(z['test_ids'])&(set(z['head_train_ids'])|set(z['encoder_train_ids']))
            checks.append(dict(file=p.name,sha256=sha(p),head_train_n=len(z['head_train_ids']),
                               replay_max_error=float(z['replay_max_embedding_error'])))
    assert len(checks)==250
    write(HERE/'INFERENCE_COMPLETE.json',dict(status='complete',new_encoder_fits=0,
        encoders_reused=250,torch=torch.__version__,numpy=np.__version__,device='cpu',
        code_sha256=sha(__file__),model_code_sha256=sha(CODE/'eegnet_model.py'),
        plan_sha256=sha(HERE/'PLAN_2026-11-05.md'),cache_manifest_sha256=sha(cache/'INPUT_MANIFEST.json'),
        input_sha256=sources,outputs=checks))

if __name__=='__main__':main()
