"""Synthetic-only checks; no clinical performance is used."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,copy,time
import numpy as np
import torch
from eegnet_model import Encoder, train_encoder, embeddings, training_moments
from prepare_windows import write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rng=np.random.default_rng(17)
    windows={str(i):rng.normal(size=(9,14,256)).astype(np.float32) for i in range(10)}
    tr=list(windows)[:8];ev=list(windows)[8:];labels=np.arange(8)%2
    start=time.perf_counter()
    m,mean,scale,curve=train_encoder(windows,tr,labels,17,device=a.device,epochs=2)
    original={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    z=embeddings(m,windows,ev,mean,scale,device=a.device)
    assert z.shape==(2,128) and np.isfinite(z).all()
    assert all(torch.equal(original[k],v.cpu()) for k,v in m.state_dict().items())
    changed=dict(windows);changed.update({i:windows[i]*1000+500 for i in ev})
    m2,mean2,scale2,curve2=train_encoder(changed,tr,labels,17,device=a.device,epochs=2)
    assert np.array_equal(mean,mean2) and np.array_equal(scale,scale2)
    assert curve==curve2 and all(torch.equal(original[k],v.cpu()) for k,v in m2.state_dict().items())
    assert np.allclose(mean.squeeze(),np.stack([windows[i].astype(float).mean((0,2)) for i in tr]).mean(0))
    # Assessment order must not alter participant embeddings or model state.
    zr=embeddings(m,windows,ev[::-1],mean,scale,device=a.device)
    assert np.array_equal(z,zr[::-1])
    # Finite gradients actually change at least one parameter from its seeded start.
    torch.manual_seed(17);initial=Encoder()
    assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in original.items() if v.dtype.is_floating_point)
    write(a.out,dict(status='PASS',device=a.device,torch=torch.__version__,parameters=sum(p.numel() for p in m.parameters()),seconds=time.perf_counter()-start,checks=['held-out signal perturbation leaves fitted states and training moments identical','assessment leaves model and batch-normalization state unchanged','participant assessment order invariance','participant-equal normalization','finite embeddings','optimization updates model'],scope='Synthetic data only; two-epoch execution check, not a study fit'))

if __name__=='__main__':main()
