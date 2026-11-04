"""Rebuild Figure 1, 2a and S1 labels, proving identical plot records."""
from pathlib import Path
import sys,subprocess,json,shutil,hashlib

HERE=Path(__file__).resolve().parent;REPO=HERE.parent;WORK=REPO.parent
PAPER=Path(r'F:\proje\eeg_gain\makale\cmpb_r2_2026-11-05')
OUT=HERE/'figure_label_audit';OUT.mkdir(exist_ok=True)
SOURCE=REPO/'display_assets/figure_source/build_assets.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    original=SOURCE.read_text(encoding='utf-8')
    modified=original.replace('Primary EEG procedure','Primary procedure')
    assert original!=modified
    for label,code in [('before',original),('after',modified)]:
        script=OUT/(label+'_generator.py');script.write_text(code,encoding='utf-8')
        output=WORK/('figure_label_'+label)
        with (OUT/(label+'_RUN.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(script),'--release-root',str(REPO),
                            '--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    before=json.loads((WORK/'figure_label_before/figure_source/ASSET_VALUES.json').read_text(encoding='utf-8'))
    after=json.loads((WORK/'figure_label_after/figure_source/ASSET_VALUES.json').read_text(encoding='utf-8'))
    assert before['plots']==after['plots']
    assert before['table_cells']==after['table_cells']
    assert before['source_sha256']==after['source_sha256']
    archived=json.loads((REPO/'display_assets/figure_source/ASSET_VALUES.json').read_text(encoding='utf-8'))
    names={'repeat_increments':'Figure_1','acquisition_phase_controls':'Figure_2','participant_changes':'Figure_S1'}
    checks={}
    for name,upload in names.items():
        assert archived['plots'][name]==before['plots'][name],name
        src=WORK/'figure_label_after/figures'/(name+'.pdf')
        prior=PAPER/'figures'/(name+'.pdf')
        old_hash=sha(prior)
        shutil.copyfile(src,prior)
        shutil.copyfile(src,PAPER/'submission/artwork'/(upload+'.pdf'))
        checks[name]=dict(archived_plot_records_equal=True,before_after_plot_records_equal=True,
                         old_pdf_sha256=old_hash,new_pdf_sha256=sha(src),new_label='Primary procedure')
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='PASS',model_fits=0,
        source_generator_sha256=sha(SOURCE),input_hashes_unchanged=True,
        all_numeric_table_cells_unchanged=True,figures=checks),indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
