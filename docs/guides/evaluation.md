# Evaluation protocol

Write the generalization question before choosing a split. Same-run trial generalization, new-run transfer, new-session transfer and new-subject transfer are different claims.

## Before fitting

Record dataset version, original event mapping, subject/session/run/trial IDs, channel selection, voltage units, reference, passband, filter phase, epoch interval and exclusion rules. Fix descriptive plotting choices or restrict exploration to training data. Keep all windows and repetitions of one trial or character in a single group. Preserve continuous-run boundaries during filtering.

## Inside training only

Fit scaling, imputation, ICA/regression artifact models used for predictive evaluation, CSP, feature selection and hyperparameter search. A fixed deterministic bandpass is not a label-trained transform, but filtering across a temporal split can still share samples. Use separate runs/sessions or adequate guard regions.

Nested cross-validation uses an outer split for evaluation and inner splits for selection. A held-out session is evaluated once after model choices are fixed. Never choose a random seed because its test score looks best.

## Metrics and uncertainty

P300: ROC AUC plus precision/recall or average precision and confusion matrix; target rarity makes accuracy deceptive. Motor imagery: balanced accuracy, classwise recall and confusion matrix. SSVEP: selection accuracy versus decision-window length. A full interface additionally needs rejection rate, idle-state false commands per minute, correction burden, latency and user workload.

Do not treat folds as independent participants. Aggregate per-subject results before population inference; report individual variation. Permute labels only within a defensible exchangeability scheme. A cluster-based EEG test addresses a different question from predictive accuracy and requires its own assumptions.

## Required result record

Dataset and code versions; fixed seed; training and test IDs; retained/rejected counts by condition; primary metric; complete confusion matrix; selected hyperparameters; computation time; deviations from the protocol; limitations. Negative results are legitimate results.
