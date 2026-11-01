"""Reproduce locked analyses into a separate output directory without altering supplied results."""
from pathlib import Path
import argparse,json,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_extension as core
SOURCE=core.ROOT
def worker_init(release,data,workspace):
 core.ROOT=Path(workspace);core.initialize(release,data)
def run(args):
 output=Path(args.output_root).resolve();output.mkdir(parents=True,exist_ok=True)
 if any(output.iterdir()):raise RuntimeError('Choose an empty output directory; existing results are retained')
 (output/'audit').mkdir()
 for name in ['ANALYSIS_PLAN.txt','EXTERNAL_ANALYSIS_PLAN.txt']:shutil.copy2(SOURCE/name,output/name)
 for name in ['PLAN_LOCK.json','EXTERNAL_PLAN_LOCK.json','INPUT_MANIFEST.json']:shutil.copy2(SOURCE/'audit'/name,output/'audit'/name)
 release=Path(args.release_root).resolve();manifest=json.loads((SOURCE/'audit/INPUT_MANIFEST.json').read_text());assert all(core.sha(release/k)==v for k,v in manifest['files'].items())
 assert core.sha(SOURCE/'ANALYSIS_PLAN.txt')==json.loads((SOURCE/'audit/PLAN_LOCK.json').read_text())['plan_sha256']
 if args.part=='external':
  import run_external as adult
  core.ROOT=output;adult.OUT=output/'results/external';adult.run(str(release));return
 core.load_frozen(release);data=core.MOD.load_data() if args.part=='empirical' else None
 if args.part=='empirical':
  specs=[(r,f,frac,width) for r in range(10) for f in range(5) for frac,width in [(1.,0),(1.,3),(.5,0),(.75,0)]];worker=core.empirical_job
 else:specs=[(s,r,n) for s in core.SCENARIOS for r in range(20) for n in [96,192,384]];worker=core.simulation_job
 if args.first_only:specs=specs[:1]
 start=time.monotonic()
 with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(release),data,str(output))) as pool:
  futures=[pool.submit(worker,s) for s in specs]
  for i,f in enumerate(as_completed(futures),1):
   f.result()
   if i==1 or i%10==0 or i==len(specs):print(args.part,i,'/',len(specs),flush=True)
 assert all(core.sha(release/k)==v for k,v in manifest['files'].items())
 core.dump(output/'audit/REPRODUCTION.json',{'part':args.part,'jobs':len(specs),'first_only':args.first_only,'seconds':time.monotonic()-start,'source_core_sha256':core.sha(SOURCE/'code/run_extension.py'),'wrapper_sha256':core.sha(__file__),'inputs_unchanged':True})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('part',choices=['empirical','simulation','external']);p.add_argument('--release-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--first-only',action='store_true');run(p.parse_args())
