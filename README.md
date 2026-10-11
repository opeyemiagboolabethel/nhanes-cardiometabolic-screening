# Cardiometabolic Screening Intelligence

An end-to-end health-data science project by Opeyemi O. Agboola. The question is specific: can age, recorded sex, BMI and waist circumference help screen for elevated HbA1c among U.S. adults without reported diabetes?

This is cross-sectional screening for HbA1c ≥5.7% at examination. It is not a prediction of future diabetes, a diagnosis, or a clinically validated decision tool. The broader portfolio title is Cardiometabolic Risk Prediction with NHANES; this implementation makes its actual outcome explicit.

## Data and reproducibility

Official CDC/NCHS public-use NHANES files only: DEMO, BMX, GHB and DIQ. Development uses 2015–2016; the untouched temporal test uses 2017–March 2020 pre-pandemic. No Kaggle or synthetic data. `data/source_manifest.json` records direct URLs, file sizes and SHA-256 checksums. The raw files are included in the source handoff archive; GitHub ignores them because `download_data.py` reproduces the source retrieval.

```bash
python -m pip install -r requirements.txt
python download_data.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/pipeline.py
python -m unittest test_project.py
streamlit run streamlit_app.py
```

Run from this repository root. Downloading refreshes the source manifest; compare it with the committed version to detect upstream changes. The model artifact must be loaded only from this trusted repository and with its pinned scikit-learn version.

## Study design

Adults aged 20+, positive MEC weights, valid no/borderline diabetes response, no reported insulin or diabetes-tablet use, no recorded pregnancy and observed HbA1c. Pregnancy-unknown respondents remain eligible. Borderline diabetes respondents remain eligible. No missing outcomes are imputed. Age is top-coded at 80 by NHANES.

Joins are one-to-one on SEQN. The outcome, glycemic biomarkers, diagnosis and treatment variables are excluded from the predictor matrix. Sex is the binary recorded NHANES variable; it does not capture gender identity. Race/ethnicity is used only for a descriptive subgroup audit.

Development survey PSUs are split into fitting and calibration partitions. Four-fold stratified group cross-validation compares logistic regression, spline logistic regression, random forest and histogram gradient boosting. Median imputation, missing indicators and scaling are learned within training folds. Training uses normalised MEC weights; model selection uses weighted AUC. Candidate settings are prespecified in source rather than tuned against the test release.

The selected model is fitted only on the fitting partition. A sigmoid calibrator uses the calibration partition; that same partition fixes a research threshold targeting at least 80% weighted sensitivity. Calibration-partition performance is therefore not unbiased validation. The temporal release trains neither model nor calibrator and selects no threshold.

Weighted temporal AUC, average precision, Brier score, log loss, sensitivity, specificity, predictive value and screening fraction are reported. Stratified rescaled PSU bootstrap intervals are conditional on the frozen model and exclude model-development uncertainty. There is no clinical utility claim. Decision-curve outputs are exploratory.

## Deliverables

- `src/pipeline.py`: full cohort, training, evaluation and plotting workflow.
- `streamlit_app.py`: evidence dashboard and bounded research scenario explorer; no credentials needed.
- `outputs/summary.json`: results, cohort flows, model selection, intervals and limitations.
- `outputs/temporal_predictions.csv`: public-use outcome, probability, weights and design fields for independent evaluation.
- Model comparisons, missingness, subgroup audit, sensitivity analyses, calibration bins, permutation importance and decision curves.
- Three publication-style figures and a saved model artifact.
- `test_project.py`: source-integrity, leakage, metric-reproduction and application checks.

## Interpretation limits

NHANES measures HbA1c at a single examination. A single marker is not a confirmed diagnosis and does not measure all cardiometabolic outcomes. HbA1c can vary with red-cell and haemoglobin-related factors; group audit differences require careful interpretation. U.S. validation does not establish transportability to Nigeria.

MEC weights are used without additional laboratory item-nonresponse reweighting. Weighted estimates describe the observed analytic domain under that assumption. Imputed predictors preserve more records but do not resolve every missing-data mechanism. Complete-predictor and HbA1c-below-6.5% sensitivity analyses are provided. Subgroup estimates are descriptive without survey-based intervals, and fewer than 30 participants in either outcome class triggers suppression.

Permutation importance measures predictive reliance, not causality. Correlated BMI and waist measures can share importance. Higher-performing complexity was not assumed: the baseline is retained when development evidence supports it.

## Sources

- CDC NHANES: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- 2015–2016 HbA1c: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.htm
- Pre-pandemic HbA1c: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GHB.htm
- Pre-pandemic demographics and weights: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm
- CDC diabetes testing: https://www.cdc.gov/diabetes/diabetes-testing/
- NIDDK A1c interpretation: https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test/

No data or outcomes from this research demonstration should be presented as clinical deployment results.
