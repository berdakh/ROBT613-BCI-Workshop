# Learning and generalization

A classifier learns a mapping from a measured representation to a task label. Its success depends as much on the labels and split protocol as on the algorithm.

## Linear baselines first

LDA compares class means under a shared covariance model. Shrinkage improves stability with limited or correlated EEG features. Logistic regression directly models class probability through a linear score and sigmoid. Both are cheap, inspectable baselines. More complex models should be compared using the same preprocessing, split and metric.

## Spatial learning

CSP uses class covariance differences to find spatial filters. Because labels enter the fit, it must be trained inside each fold. The number of components is a hyperparameter. A multiclass implementation extends beyond the binary generalized eigenproblem; do not describe a four-class fit as four independent binary filters unless that is what the code actually does.

## Neural networks

Convolutions learn local temporal patterns with shared weights. Nonlinear layers and pooling compose representations. With few trials, parameter count and validation choices matter more than architectural novelty. A held-out validation set supports early stopping; a separate test set supports the final claim. An autoencoder minimizes reconstruction error, which does not guarantee class separation or physiological denoising.

## What does a score mean?

A within-person new-run score describes that setting. It cannot establish performance for a new person, new cap placement, new device or clinical population. Report a confusion matrix and classwise metrics. AUC assesses ranking, not a deployed threshold. Calibration assesses whether predicted probabilities match frequencies; it requires separate data if tuned.

A reproducibility record includes failed models and protocol deviations. If results are weak, inspect units, events, quality and splits before trying larger models. Chance-level performance may reflect insufficient signal, insufficient data or a difficult transfer setting; it is not proof that a participant cannot use a BCI.

Read alongside notebooks 02, 08–09 and 13. [Scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
