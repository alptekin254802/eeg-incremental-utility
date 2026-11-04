"""Recompute requested manuscript numbers from recorded participant predictions."""
from common import *
from scipy.stats import spearmanr

def means(frame):
    frame=frame.copy();frame['loss']=loss(frame.y,frame.p)
    return frame.groupby('repeat').loss.mean().mean()

def main():
    out=HERE/'recorded_audit';out.mkdir(exist_ok=True)
    p=read(REPO/'expected/posthoc/results/outer_predictions.csv')
    results=[]
    def check(name,value,expected,places=4):
        results.append(dict(quantity=name,recomputed=float(value),requested=float(expected),decimals=places,
                            matches=round(float(value),places)==round(float(expected),places)))
    for suffix,expected in [('',.6244),('_D',.6094),('_DQ',.6180)]:
        check('Table4_baseline'+suffix,means(p[p.model.eq('M2'+suffix)]),expected)
    for suffix,expected in [('',.0267),('_D',.0204),('_DQ',.0071)]:
        check('entropy_delta'+suffix,means(p[p.model.eq('M2'+suffix)])-means(p[p.model.eq('E4'+suffix)]),expected)
    utility=read(REPO/'companion/extensions/utility/reports/EMPIRICAL_PREDICTIONS.csv')
    for adjustment,expected in [(0,.0139),(3,-.0068)]:
        d=utility[utility.fraction.eq(1)&utility.learner.eq('rbf_svm')&utility.adjustment.eq(adjustment)]
        check('SVM_entropy_delta_adjustment_'+str(adjustment),
            means(d[d.model.eq('baseline')])-means(d[d.model.eq('entropy_hjorth')]),expected)
    base,linked,e4,nuisance=baseline_data()
    check('control_diagnosed_yes',len(linked[linked.group_y.eq(0)&linked.diagnosed_raw.eq('yes')]),13,0)
    check('control_diagnosed_undetermined',len(linked[linked.group_y.eq(0)&linked.diagnosed_raw.eq('undetermined')]),12,0)
    screen=read(REPO/'audit/cohort_screen/robots_final_feasibility.csv')
    readable=int(screen[screen.scope.eq('primary_candidate')].iloc[0].denominator)
    check('excluded_records',readable-len(base),30,0)
    with np.load(REPO/'expected/posthoc/signal_controls/surrogate_features.npz',allow_pickle=False) as z:
        ids=z['participant_id'].tolist();names=z['feature_names'].tolist();control=z['features'].mean(axis=0)
    corr=[spearmanr(e4.loc[ids,name],control[:,j]).statistic for j,name in enumerate(names)]
    check('phase_feature_spearman_min',min(corr),.987,3)
    check('phase_feature_spearman_max',max(corr),.9997,4)
    result=pd.DataFrame(results)
    result.to_csv(out/'VERIFICATION.csv',index=False,float_format='%.17g')
    assert result.matches.all(),result[~result.matches].to_dict('records')
    pd.DataFrame([dict(adjustment=suffix or 'none',baseline_logloss=means(p[p.model.eq('M2'+suffix)]))
                  for suffix in ['', '_D','_DQ']]).to_csv(out/'TABLE4_BASELINES.csv',index=False,float_format='%.17g')
    provenance(out,[Path(__file__),REPO/'expected/posthoc/results/outer_predictions.csv',
        REPO/'companion/extensions/utility/reports/EMPIRICAL_PREDICTIONS.csv',
        REPO/'expected/posthoc/signal_controls/surrogate_features.npz',
        REPO/'audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv'])
    print(result.to_string(index=False))

if __name__=='__main__':main()
