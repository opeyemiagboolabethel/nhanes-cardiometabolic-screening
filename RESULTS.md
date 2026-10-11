# Validation results

Development: 4,294 participants; fitting 3,097; calibration 1,197. Temporal test: 6,644.

Selected model: Logistic baseline. Weighted temporal ROC AUC: 0.744 (conditional 95% PSU bootstrap interval 0.721–0.763). Average precision: 0.475; outcome prevalence: 27.1%.

At the development-selected threshold of 22.3%: sensitivity 80.9%, specificity 55.3%, positive predictive value 40.2%, and screen-positive fraction 54.5%. Most screen-positive cases therefore do not have elevated HbA1c; laboratory confirmation remains necessary.

Brier score: 0.1711 versus 0.1975 for the development-prevalence constant baseline. Sigmoid calibration did not improve temporal Brier score compared with the uncalibrated selected model (0.1711); the procedure is reported as specified rather than retrospectively changed using test results.

This is screening for current elevated HbA1c, not incident diabetes prediction. No clinical benefit or Nigerian generalisability is established. Full design and limitations are in README.md and outputs/summary.json.
