"""Place EEGNet result displays alongside the existing article assets."""
import argparse,json,shutil,re
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.reports/'SUMMARY.json').read_text())
    for name in ('eegnet_comparison.pdf','eegnet_comparison.svg'):
        shutil.copy2(a.reports/name,a.assets/'figures'/name)
    shutil.copy2(a.reports/'eegnet_summary.tex',a.assets/'tables/eegnet_summary.tex')
    target=a.assets/'tables/main_learner.tex';t=target.read_text()
    if 'EEGNet-like' in t:raise RuntimeError('Expected original six-row learner table, not an already extended table')
    row='Logistic regression & EEGNet-like & '+ ' & '.join('$'+format(v,'.4f' if i==0 else '+.4f')+'$' for i,v in enumerate([s['baseline_logloss'],s['delta_logloss'],s['delta_auroc'],s['delta_brier']]))+r'\\'+'\n'
    t=t.replace(r'\bottomrule',row+r'\bottomrule')
    caption=r'\caption{Paired EEG increments in the pediatric cohort, averaged over ten dependent repeats. Positive values favor EEG. Each procedure uses its learner\'s matched baseline. EEGNet-like denotes averaged learned embeddings with a logistic head; additional learner and representation comparisons are exploratory.}'
    caption=caption.replace("\\'", "'")
    t=re.sub(r'\\caption\{[^\n]*\}',lambda _:caption,t,count=1)
    target.write_text(t,encoding='utf-8')

if __name__=='__main__':main()
