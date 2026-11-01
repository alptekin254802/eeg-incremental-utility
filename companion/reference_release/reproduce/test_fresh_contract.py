"""Regression tests for the critical fresh-input boundary, without raw data."""
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class FreshBoundaryTests(unittest.TestCase):
    def child(self,body):
        with tempfile.TemporaryDirectory(prefix='eeg-fresh-test-') as directory:
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            env.update(EEG_RELEASE_ROOT=str(ROOT),EEG_RUN_ROOT=directory,EEG_DATASET_ROOT=directory,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            script="import sys;sys.path.insert(0,"+repr(str(ROOT/'reproduce'))+");import raw_pipeline as p\n"+body
            result=subprocess.run([sys.executable,'-B','-I','-c',script],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_saved_features_and_predictions_cannot_feed_learning(self):
        self.child("""
p.bind()
for relative in ['audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv','audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv','expected/posthoc/results/outer_predictions.csv','results/e5/OUTER_PREDICTIONS.csv']:
    try: (p.ROOT/relative).read_bytes()
    except PermissionError: pass
    else: raise AssertionError('Historical read allowed: '+relative)
fresh=p.FRESH/'new.csv';fresh.write_text('x\\n1\\n');assert fresh.read_text()=='x\\n1\\n'
assert len(p.DENIED)==4
""")

    def test_tampered_new_feature_is_rejected_before_fit(self):
        self.child("""
p.bind()
p.dump(p.RUN/'FRESH_CONTRACT.json',{'test_fixture':True})
fresh=p.FRESH/'new.csv';fresh.write_text('x\\n1\\n')
p.dump(p.RUN/'FEATURE_MANIFEST.json',{'contract_sha256':p.digest(p.RUN/'FRESH_CONTRACT.json'),'sha256':{'new.csv':p.digest(fresh)}})
fresh.write_text('x\\n2\\n')
try: p.load_features()
except AssertionError as e: assert 'Fresh feature changed' in str(e)
else: raise AssertionError('Tampered feature accepted')
""")

    def test_resume_rejects_changed_source_contract(self):
        self.child("""
p.bind()
saved=p.checkpoint('test','job',{'source':'A'},lambda: {'value':123})
assert saved=={'value':123}
try: p.checkpoint('test','job',{'source':'B'},lambda: {'value':999})
except AssertionError as e: assert 'Stale checkpoint' in str(e)
else: raise AssertionError('Stale checkpoint accepted')
""")

    def test_checkpoint_first_and_resumed_values_serialize_identically(self):
        self.child("""
p.bind()
first=p.checkpoint('test','tuple',{'source':'A'},lambda: {'bounds':[(0,256)]})
resumed=p.checkpoint('test','tuple',{'source':'A'},lambda: None)
assert first==resumed
assert str(first)==str(resumed)
""")

if __name__=='__main__':unittest.main()
