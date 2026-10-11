from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.pipeline import predict, FEATURES

ROOT=Path(__file__).parent
st.set_page_config(page_title='Cardiometabolic Screening Intelligence',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp{background:#faf8f2}h1,h2,h3{color:#0a2537}div[data-testid="stMetric"]{background:white;border:1px solid #d8e1df;border-radius:12px;padding:20px}.research-hero{background:linear-gradient(120deg,#082334,#176a70);padding:36px;border-radius:16px;color:white}.research-hero h1{color:white}.research-hero p{color:#d7e7e4}</style>''',unsafe_allow_html=True)
@st.cache_resource
def artifacts():
    return json.loads((ROOT/'outputs/summary.json').read_text()),joblib.load(ROOT/'outputs/screening_model.joblib')
s,bundle=artifacts()
st.markdown('<div class="research-hero"><p>OPEYEMI AGBOOLA · HEALTH DATA SCIENCE</p><h1>Cardiometabolic Screening Intelligence</h1><p>Non-laboratory screening for elevated HbA1c, tested on a separate NHANES release.</p></div>',unsafe_allow_html=True)
st.caption('Research demonstration · U.S. adults aged 20+ without reported diabetes · HbA1c ≥5.7% at examination · not future disease prediction')
tabs=st.tabs(['Evidence & performance','Screening scenario','Model transparency','Methods & limitations'])
with tabs[0]:
    m=s['temporal_metrics'];ci=s['conditional_test_ci']['roc_auc']
    a,c,d,e=st.columns(4);a.metric('Temporal test cohort',f"{s['temporal_test_n']:,}");c.metric('Weighted ROC AUC',f"{m['roc_auc']:.3f}");d.metric('Weighted Brier score',f"{m['brier']:.3f}");e.metric('Weighted sensitivity',f"{m['sensitivity']:.1%}")
    st.caption(f"AUC conditional 95% bootstrap interval {ci['low']:.3f}–{ci['high']:.3f}. Threshold fixed on development calibration data, then evaluated on later data.")
    a,c=st.columns(2);a.image(str(ROOT/'outputs/temporal_roc.png'));c.image(str(ROOT/'outputs/temporal_calibration.png'))
    st.caption('High estimated probabilities overpredicted observed proportions in the later release. Sigmoid calibration did not improve temporal Brier score over the uncalibrated model.')
    st.subheader('Screening trade-offs on the temporal test cohort')
    st.dataframe(pd.DataFrame([{'Sensitivity':m['sensitivity'],'Specificity':m['specificity'],'Positive predictive value':m['positive_predictive_value'],'Screen-positive fraction':m['screen_positive_fraction'],'Average precision':m['average_precision'],'Outcome prevalence':m['prevalence']}]).T.rename(columns={0:'Weighted estimate'}),width='stretch')
    st.caption('Accuracy alone is not the objective. Detection gains come with more screen-positive results; a laboratory test is still required.')
with tabs[1]:
    st.info('Explore how the fitted research model behaves. It does not diagnose diabetes, prescribe care or establish eligibility for screening. No inputs are stored.')
    with st.form('scenario'):
        a,c=st.columns(2);age=a.number_input('Age (years; 80 includes 80+ in NHANES)',20,80,45);sex=c.selectbox('Recorded sex category in NHANES',['Female','Male']);bmi=a.number_input('BMI (kg/m²)',12.0,70.0,27.0,.5);waist=c.number_input('Waist circumference (cm)',50.0,180.0,95.0,1.0)
        submit=st.form_submit_button('Explore screening estimate')
    if submit:
        X=pd.DataFrame([[age,1 if sex=='Male' else 2,bmi,waist]],columns=FEATURES);p=float(predict(bundle,X)[0]);st.metric('Estimated elevated-HbA1c probability',f'{p:.1%}')
        st.write(f"Development-selected research threshold: {bundle['threshold']:.1%}. This scenario falls {'above' if p>=bundle['threshold'] else 'below'} that threshold.")
        st.warning('This is a population research estimate for a U.S. survey setting. Individual reliability, local calibration and clinical usefulness have not been established, including in Nigeria. The model cannot replace HbA1c testing.')
with tabs[2]:
    st.subheader('Selection used development data only')
    st.dataframe(pd.read_csv(ROOT/'outputs/model_comparison.csv'),width='stretch',hide_index=True);st.write('Selected model:',s['selected_model'])
    st.image(str(ROOT/'outputs/feature_reliance.png'))
    st.caption('Permutation importance describes predictive reliance, not causal effects; BMI and waist are correlated. No HbA1c, glucose or diabetes-treatment information enters the predictor matrix.')
    st.subheader('Subgroup audit')
    st.dataframe(pd.read_csv(ROOT/'outputs/subgroup_audit.csv'),width='stretch',hide_index=True)
    st.caption('Race/ethnicity is used for descriptive auditing only, not model inputs or biological explanation. Small outcome cells are suppressed; observed differences do not establish fairness.')
    st.subheader('Sensitivity analyses');st.dataframe(pd.read_csv(ROOT/'outputs/sensitivity.csv'),width='stretch',hide_index=True)
with tabs[3]:
    st.write('Development: NHANES 2015–2016. Temporal test: 2017–March 2020 pre-pandemic. Join: SEQN, validated one-to-one.')
    st.write('Four predictors: age, recorded sex, BMI and waist circumference. Eligibility excludes reported diabetes, diabetes medication use and recorded pregnancy. Adults with borderline reported diabetes remain eligible.')
    st.write('Four-fold group cross-validation keeps survey PSUs together. A separate development PSU partition fits a sigmoid calibrator and fixes the threshold targeting at least 80% sensitivity. The temporal test does not train or select models.')
    st.write('Training weights normalised within partitions. Later-release MEC weights used for evaluation. Stratified rescaled PSU bootstrap estimates conditional test uncertainty. This is not full model-development uncertainty.')
    for limitation in s['limitations']:st.write('• '+limitation)
    for role,flow in s['flow'].items():
        st.subheader(role.replace('_',' ').title()+' cohort flow');st.dataframe(pd.DataFrame(flow),hide_index=True,width='stretch')
    st.link_button('CDC: NHANES official data','https://wwwn.cdc.gov/nchs/nhanes/Default.aspx')
    st.download_button('Download aggregate results',json.dumps(s,indent=2),'screening_results.json','application/json')
