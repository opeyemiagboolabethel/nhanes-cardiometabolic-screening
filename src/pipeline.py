"""Reproducible, survey-aware temporal validation of non-laboratory screening."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve
from sklearn.base import clone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['RIDAGEYR','RIAGENDR','BMXBMI','BMXWAIST']
LABELS={'RIDAGEYR':'Age (years)','RIAGENDR':'Recorded sex','BMXBMI':'BMI (kg/m²)','BMXWAIST':'Waist circumference (cm)'}
SEED=20261011

def cohort(prefix, weight):
    tables={k:pd.read_sas(ROOT/'data/raw'/f'{prefix(k)}.xpt',format='xport') for k in ['DEMO','BMX','GHB','DIQ']}
    d=tables['DEMO'].copy()
    for k in ['BMX','GHB','DIQ']:
        t=tables[k];assert t.SEQN.is_unique
        cols=['SEQN']+[c for c in t if c not in d]
        d=d.merge(t[cols],on='SEQN',how='left',validate='one_to_one')
    flow=[{'stage':'All respondents','n':len(d)}]
    filters=[('Age 20+',(d.RIDAGEYR>=20)),('Positive MEC examination weight',d[weight]>0),('No reported diabetes; valid questionnaire',d.DIQ010.isin([2,3])),('No reported insulin or diabetes tablets',(d.DIQ050!=1)&(d.DIQ070!=1)),('Not recorded pregnant',d.RIDEXPRG!=1),('HbA1c observed',d.LBXGH.notna())]
    keep=pd.Series(True,index=d.index)
    for label,f in filters:
        keep &= f;flow.append({'stage':label,'n':int(keep.sum())})
    d=d.loc[keep].copy();d['target']=(d.LBXGH>=5.7).astype(int)
    d['weight']=d[weight];d['group']=d.SDMVSTRA.astype(int).astype(str)+'_'+d.SDMVPSU.astype(int).astype(str)
    assert d.SEQN.is_unique and d.target.nunique()==2
    return d,flow

def weights(d):return d.weight.to_numpy()/d.weight.mean()
def metrics(y,p,w,t):
    p=np.clip(p,1e-6,1-1e-6);pred=p>=t
    return {'roc_auc':float(roc_auc_score(y,p,sample_weight=w)),'average_precision':float(average_precision_score(y,p,sample_weight=w)),'brier':float(brier_score_loss(y,p,sample_weight=w)),'log_loss':float(log_loss(y,p,sample_weight=w)),'prevalence':float(np.average(y,weights=w)),'sensitivity':float(np.average(pred[y==1],weights=w[y==1])),'specificity':float(np.average(~pred[y==0],weights=w[y==0])),'screen_positive_fraction':float(np.average(pred,weights=w)),'positive_predictive_value':float(np.average(y[pred],weights=w[pred])) if pred.any() else None}
def logit(p):return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
def predict(bundle,X):return bundle['calibrator'].predict_proba(logit(bundle['model'].predict_proba(X)[:,1]))[:,1]

def bootstrap_ci(d,p,t,B=400):
    """Stratified rescaled PSU bootstrap, frozen model; conditional test uncertainty."""
    rng=np.random.default_rng(SEED);base=d.weight.to_numpy();ys=d.target.to_numpy();out=[]
    pairs={s:d.loc[d.SDMVSTRA==s,'group'].unique() for s in d.SDMVSTRA.unique()}
    for _ in range(B):
        factors=dict.fromkeys(d.group.unique(),0)
        for groups in pairs.values():
            m=len(groups)
            if m<2:raise ValueError('Singleton PSU in test stratum')
            for g in rng.choice(groups,size=m-1,replace=True):factors[g]+=m/(m-1)
        w=base*d.group.map(factors).to_numpy()
        m=metrics(ys,p,w,t);out.append(m)
    a=pd.DataFrame(out)
    return {k:{'low':float(a[k].quantile(.025)),'high':float(a[k].quantile(.975))} for k in a}

def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    dev,df=cohort(lambda k:k+'_I','WTMEC2YR');test,tf=cohort(lambda k:'P_'+k,'WTMECPRP')
    assert set(dev.SEQN).isdisjoint(test.SEQN)
    fit_idx,cal_idx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED).split(dev,groups=dev.group))
    fit=dev.iloc[fit_idx];cal=dev.iloc[cal_idx]
    assert set(fit.group).isdisjoint(cal.group)
    models={
      'Logistic baseline':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=2000))]),
      'Spline logistic':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('spline',SplineTransformer(n_knots=4,degree=3)),('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=2000))]),
      'Random forest':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('model',RandomForestClassifier(n_estimators=250,min_samples_leaf=25,max_depth=8,max_features=1.0,n_jobs=2,random_state=SEED))]),
      'Gradient boosting':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('model',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,random_state=SEED))])}
    rows=[];cv=StratifiedGroupKFold(4,shuffle=True,random_state=SEED)
    for name,model in models.items():
        auc=[]
        for a,b in cv.split(fit[FEATURES],fit.target,fit.group):
            m=clone(model);m.fit(fit.iloc[a][FEATURES],fit.iloc[a].target,model__sample_weight=weights(fit.iloc[a]))
            auc.append(roc_auc_score(fit.iloc[b].target,m.predict_proba(fit.iloc[b][FEATURES])[:,1],sample_weight=fit.iloc[b].weight))
        rows.append({'model':name,'development_group_cv_auc':float(np.mean(auc)),'fold_aucs':auc})
        print(name,round(np.mean(auc),4),flush=True)
    selected=max(rows,key=lambda r:r['development_group_cv_auc'])['model'];model=models[selected]
    model.fit(fit[FEATURES],fit.target,model__sample_weight=weights(fit))
    cp=model.predict_proba(cal[FEATURES])[:,1];calibrator=LogisticRegression(C=1000,max_iter=1000)
    calibrator.fit(logit(cp),cal.target,sample_weight=weights(cal))
    bundle={'model':model,'calibrator':calibrator,'features':FEATURES,'selected':selected}
    cp=predict(bundle,cal[FEATURES]);fpr,tpr,thr=roc_curve(cal.target,cp,sample_weight=cal.weight)
    choices=np.where((tpr>=.8)&np.isfinite(thr))[0];idx=choices[np.argmin(fpr[choices])];threshold=float(thr[idx]);bundle['threshold']=threshold
    # The later release is opened for performance only after model, calibration and threshold are frozen.
    p=predict(bundle,test[FEATURES]);y=test.target.to_numpy();w=test.weight.to_numpy()
    result=metrics(y,p,w,threshold);ci=bootstrap_ci(test,p,threshold)
    baseline=np.full(len(test),np.average(fit.target,weights=fit.weight))
    raw_p=model.predict_proba(test[FEATURES])[:,1]
    missing=[]
    for release,d in [('development',dev),('temporal_test',test)]:
        for c in FEATURES:missing.append({'release':release,'feature':c,'missing_n':int(d[c].isna().sum()),'missing_pct':float(d[c].isna().mean()*100)})
    pd.DataFrame(missing).to_csv(out/'missingness.csv',index=False)
    sub=[]
    for axis,groups in [('sex',test.RIAGENDR.map({1:'Male',2:'Female'})),('age',pd.cut(test.RIDAGEYR,[19,39,59,80],labels=['20–39','40–59','60–80+'])),('race_ethnicity',test.RIDRETH3.map({1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other / multiracial'}))]:
        for label in groups.dropna().unique():
            ix=(groups==label).to_numpy();n=int(ix.sum());positives=int(y[ix].sum())
            if positives<30 or n-positives<30:sub.append({'axis':axis,'group':str(label),'n':n,'positives':positives,'status':'Suppressed: fewer than 30 in one outcome class'});continue
            sub.append({'axis':axis,'group':str(label),'n':n,'positives':positives,'status':'Descriptive audit; no subgroup CI',**metrics(y[ix],p[ix],w[ix],threshold)})
    pd.DataFrame(sub).to_csv(out/'subgroup_audit.csv',index=False)
    sensitivity=[]
    for name,ix in [('Complete predictors',test[FEATURES].notna().all(axis=1).to_numpy()),('HbA1c below 6.5%',(test.LBXGH<6.5).to_numpy())]:
        sensitivity.append({'cohort':name,'n':int(ix.sum()),**metrics(y[ix],p[ix],w[ix],threshold)})
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    # Test permutation importance is an explanatory audit, never a selection input.
    rng=np.random.default_rng(SEED);importance=[]
    for c in FEATURES:
        drops=[]
        for _ in range(10):
            X=test[FEATURES].copy();X[c]=rng.permutation(X[c].to_numpy());drops.append(result['roc_auc']-roc_auc_score(y,predict(bundle,X),sample_weight=w))
        importance.append({'feature':c,'label':LABELS[c],'auc_drop':float(np.mean(drops)),'sd':float(np.std(drops))})
    pd.DataFrame(importance).to_csv(out/'permutation_importance.csv',index=False)
    # Aggregate calibration bins, weighted rather than unweighted.
    bins=[]
    for lo in np.arange(0,1,.1):
        ix=(p>=lo)&(p<lo+.1)
        if ix.any():bins.append({'bin_low':float(lo),'n':int(ix.sum()),'predicted':float(np.average(p[ix],weights=w[ix])),'observed':float(np.average(y[ix],weights=w[ix]))})
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    plt.rcParams.update({'figure.facecolor':'#faf8f2','axes.facecolor':'#faf8f2','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,5));f,t,_=roc_curve(y,p,sample_weight=w);ax.plot(f,t,color='#147d87',lw=3,label=f"Temporal test AUC {result['roc_auc']:.3f}");ax.plot([0,1],[0,1],'--',color='#b6b6b6');ax.set(xlabel='False-positive rate',ylabel='Sensitivity',title='Discrimination on the untouched later NHANES release');ax.legend();fig.tight_layout();fig.savefig(out/'temporal_roc.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot([0,1],[0,1],'--',color='#b6b6b6');ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',color='#147d87');ax.set(xlabel='Mean predicted probability',ylabel='Observed elevated HbA1c proportion',title='Survey-weighted temporal calibration',xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(out/'temporal_calibration.png',dpi=180);plt.close(fig)
    imp=sorted(importance,key=lambda a:a['auc_drop']);fig,ax=plt.subplots(figsize=(8,5));ax.barh([a['label'] for a in imp],[a['auc_drop'] for a in imp],color='#147d87');ax.set(xlabel='Decrease in weighted AUC after shuffling',title='Model reliance on non-laboratory inputs');fig.tight_layout();fig.savefig(out/'feature_reliance.png',dpi=180);plt.close(fig)
    decision=[]
    for pt in np.arange(.05,.61,.025):
        ix=p>=pt;tp=np.sum(w*ix*y)/w.sum();fp=np.sum(w*ix*(1-y))/w.sum();prev=result['prevalence'];decision.append({'threshold':float(pt),'model_net_benefit':float(tp-fp*pt/(1-pt)),'screen_all_net_benefit':float(prev-(1-prev)*pt/(1-pt))})
    pd.DataFrame(decision).to_csv(out/'decision_curve.csv',index=False)
    summary={'title':'Cardiometabolic Screening Intelligence','task':'Elevated HbA1c (≥5.7%) at the survey examination; not incident disease prediction','development_n':len(dev),'fit_n':len(fit),'calibration_n':len(cal),'temporal_test_n':len(test),'selected_model':selected,'cv_results':rows,'threshold':threshold,'calibration_metrics':metrics(cal.target.to_numpy(),cp,cal.weight.to_numpy(),threshold),'temporal_metrics':result,'conditional_test_ci':ci,'constant_baseline':metrics(y,baseline,w,threshold),'uncalibrated_test':metrics(y,raw_p,w,threshold),'flow':{'development':df,'temporal_test':tf},'features':FEATURES,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__},'limitations':['Cross-sectional screening, not future disease prediction or diagnosis','United States adults; no Nigerian validation','Age top-coded at 80; exclude recorded pregnancy but unknown pregnancy status remains','MEC weights used without lab item-nonresponse reweighting; estimates apply to observed analytic cohort','Model conditional bootstrap excludes training uncertainty; subgroup estimates lack design-based confidence intervals','Development calibration set also chooses threshold; its performance is not unbiased validation','Decision curves are exploratory; no tested clinical utility','BMI and waist missingness imputed within each training fold; sex and age not imputed at application input']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));joblib.dump(bundle,out/'screening_model.joblib')
    pd.DataFrame(rows).drop(columns='fold_aucs').to_csv(out/'model_comparison.csv',index=False)
    splits=dev[['SEQN','group']].copy();splits['partition']='fit';splits.loc[cal.index,'partition']='calibration';splits.to_csv(out/'development_partitions.csv',index=False)
    # Public-use records with deterministic probabilities enable independent metric reproduction.
    preds=test[['SEQN','SDMVSTRA','SDMVPSU','weight','target']].copy();preds['probability']=p;preds.to_csv(out/'temporal_predictions.csv',index=False)
    print(json.dumps({'model':selected,'development_n':len(dev),'test_n':len(test),'metrics':result}),flush=True)

if __name__=='__main__':main()
