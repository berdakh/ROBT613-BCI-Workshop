# %% [markdown]
# # 15 · Capstone: a reproducible BCI experiment
#
# **ROBT613 · Brain–Computer Interfaces** | 6–12 hours
#
# ## Goal
# Complete a baseline experiment and produce an auditable result table, then extend one paradigm with a predeclared hypothesis.
#
# **Data:** Competition IV 2a baseline; student-selected extension
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# Your team must deliver a result another group can rerun and critique. The final task is not to maximize a leaderboard number: it is to connect a well-defined BCI question to a complete, auditable experiment.
#
# ### By the end you should be able to
#
# - Specify a reproducible experiment before inspecting final test results.
# - Produce predictions, metadata and a report that support the same claim.
# - Compare methods under a common partition and compute budget.
# - Explain limitations and propose the next independent evaluation.
#
# **How to work:** predict each result before running it, execute one cell at a time, and write a short interpretation. The worked examples use small controlled arrays; the later walkthrough uses the dataset stated above. End-of-lesson exercises contain editable workspaces. A pending exercise message is expected until you complete its function.

# %% [markdown]
# ## Setup
# The first cell installs the tested core versions in Colab. Downloads are cached in `mne_data/`; a new Colab runtime loses that cache. A failed download is an error, never silently replaced by synthetic data.

# %%
# Colab: install before importing numerical libraries. Restart if pip requests it.
import sys, subprocess, os
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'mne==1.13.2', 'moabb==1.7.2'])
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import (GroupKFold, cross_validate, GridSearchCV)
SEED = 613
rng = np.random.default_rng(SEED)
DATA_ROOT = Path(os.environ.get('BCI_DATA', './mne_data')).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MNE_DATA'] = str(DATA_ROOT)
mne.set_log_level('WARNING')
plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
import importlib.metadata as metadata
print({p: metadata.version(p) for p in ['mne','moabb','numpy','scipy','scikit-learn']})
print('Dataset cache:', DATA_ROOT)

# %% [markdown]
# ### Begin with a claim that can be tested
#
# A useful project question specifies the participant population, task, signal, prediction target and generalization unit. “Classify EEG with AI” leaves all of these open. “Compare fixed band-power and CSP–LDA features for four-class motor imagery across two sessions of one participant” is narrow enough to evaluate and reproduce. Its conclusion must remain equally narrow.
#
# Write the primary metric, split and candidate models before fitting. Separate exploratory plots from confirmatory evaluation. Exploration is valuable; the problem arises when choices made after seeing test results are presented as if they were specified in advance.

# %% [markdown]
# ### Keep the experiment reconstructable
#
# A saved score is insufficient. Preserve the dataset identifier and version, participant and session identifiers, event mapping, channels, units, preprocessing parameters, model configuration, random seed, software versions and trial-level predictions. The prediction table should contain true label, predicted label and a stable trial identifier within a documented ordering. Keep raw participant recordings out of the repository; download them from their licensed source.
#
# A manifest is a machine-readable record of these decisions. A report explains why they were made. They complement one another: a JSON field can record `tmin=0.5`, while prose explains why the cue period was excluded. Both should agree with the executed code.

# %% [markdown]
# ### A fair comparison changes one major factor
#
# When comparing methods, hold the data, split, scoring and available calibration information fixed. If a neural model receives additional tuning while a linear baseline receives none, the result compares both model class and development effort. Report that difference. A useful ablation removes or changes one component while preserving the rest of the pipeline.
#
# For paired evaluation units $g$, report $d_g=s_{A,g}-s_{B,g}$ as well as the mean difference. Consistent improvement across independent participants is stronger evidence than one favorable session. A single-participant demonstration can establish runnable methodology but cannot support a general population performance claim.

# %% [markdown]
# ### Error analysis is part of the result
#
# Inspect the confusion matrix and class recalls. Examine whether errors cluster in a run, late in a session or among low-quality trials. Do not use those observations to repeatedly modify the final model and keep calling the same session “unseen.” If the errors motivate a new hypothesis, mark it as exploratory and seek new evaluation data.
#
# Describe uncertainty in terms of the available independent units. A large number of correlated windows does not create a large number of participants. Report failed downloads, exclusions and discarded trials so another analyst can reconstruct the effective sample.

# %% [markdown]
# ### What this capstone delivers
#
# The runnable example provides a fixed CSP–LDA baseline on a documented competition dataset with one calibration session and one held-out session. Students extend it with one justified comparison and a short model card. The deliverable includes notebook outputs, the experiment record, predictions, an error analysis and a limitations paragraph. No source imaging is required or included; all features and interpretations remain at the sensor level.

# %% [markdown]
# ### Worked example · a minimal experiment record
#
# Build a small explicit record. Separate the development partition from the final evaluation partition.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_record={'question':'Four-class imagery across sessions','dataset':'BNCI2014_001','subject':1,'train_session':'calibration','test_session':'held out','primary_metric':'balanced_accuracy','seed':613}
print(pd.Series(demo_record))

# %% [markdown]
# **Read the result.** A record should use the actual session identifiers in the final experiment. Descriptive placeholders here illustrate the fields, not the dataset’s exact naming.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · paired method comparison
#
# Compute per-group differences before averaging. Inspect whether improvement is consistent.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_A=np.array([.65,.7,.55,.8])
demo_B=np.array([.6,.72,.5,.78])
demo_difference=demo_A-demo_B
print('Paired differences:',demo_difference,'Mean:',demo_difference.mean())

# %% [markdown]
# **Read the result.** One method is not better in every group. A mean alone hides this heterogeneity.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · confusion to class recalls
#
# Derive balanced accuracy from an illustrative multiclass confusion matrix with true classes in rows.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_cm=np.array([[8,1,1],[2,5,3],[0,2,8]])
demo_recall=np.diag(demo_cm)/demo_cm.sum(axis=1)
print('Class recalls:',demo_recall,'Balanced accuracy:',demo_recall.mean())

# %% [markdown]
# **Read the result.** The middle class is the weak point. A targeted scientific question is more useful than trying arbitrary models until the average rises.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Checkpoint · prediction integrity
#
# Check lengths, label range and a stable trial index before exporting a prediction table.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_predictions=pd.DataFrame({'trial_id':np.arange(6),'true':[0,1,2,0,1,2],'predicted':[0,1,1,0,2,2]})
assert demo_predictions.trial_id.is_unique
assert set(demo_predictions.predicted)<=set(demo_predictions.true)
assert not demo_predictions.isna().any().any()
print(demo_predictions)

# %% [markdown]
# **Read the result.** These checks catch bookkeeping errors. They do not establish that the scientific split is valid; that must be checked separately.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# A useful scientific question specifies the population, task, signal, comparison and generalization target. “Does CSP outperform bandpower on a new session of the same participant?” is testable; “build the best BCI” is not a protocol.
#
# Choose one primary metric before testing. For classes with unequal counts, macro recall (balanced accuracy) weights each class equally. For P300, ranking quality and precision–recall can be more informative than raw accuracy. For a complete interface, report selection accuracy and time separately from epoch classification.
#
# Every learned stage—including artifact models, normalization, CSP, feature selection and neural networks—must be fitted only on allowed calibration data. Dataset version and subject/run selection are part of the experiment. A result without its data provenance and split definition cannot be reproduced.

# %% [markdown]
# ## Run a complete baseline
# Use this as a starting experiment, then choose one extension. Keep the held-out session locked while developing it.
#
# The fixed baseline provides a reproducible starting point with no test-driven component search. Calibration fits CSP and LDA; the held-out session supplies predictions for the final metric and error table.

# %%
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.5, tmax=3.5)
epochs, y, meta = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)
X = epochs.get_data(copy=True)
y = np.asarray(y)
sessions = meta.session.astype(str).to_numpy()
groups = (meta.session.astype(str) + '/' + meta.run.astype(str)).to_numpy()
print(pd.crosstab(sessions, y))
assert len(np.unique(sessions)) == 2
train = sessions == sorted(np.unique(sessions))[0]
test = ~train

from mne.decoding import CSP
model=make_pipeline(CSP(n_components=4,reg='ledoit_wolf',log=True),
                    LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
model.fit(X[train],y[train])
pred=model.predict(X[test])
score=balanced_accuracy_score(y[test],pred)
print('Held-out session balanced accuracy:',score)
ConfusionMatrixDisplay.from_predictions(y[test],pred,xticks_rotation=45)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Check the partition and label counts before interpreting the score. Treat this result as one participant/session-transfer example, not a population estimate.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Export an experiment record
# The JSON and table are small derived results; raw datasets remain outside the repository.
#
# Export the configuration and trial predictions as small derived files. The notebook preserves data-loading instructions while avoiding redistribution of raw recordings. The record should be sufficient to reconstruct the reported result.

# %%
import json
record={'dataset':'BNCI2014-001 / BCI Competition IV 2a','subjects':[1],
        'train_sessions':sorted(set(sessions[train])), 'test_sessions':sorted(set(sessions[test])),
        'band_hz':[8,30],'epoch_seconds':[.5,3.5],'seed':SEED,
        'model':'CSP(4, Ledoit-Wolf) + shrinkage LDA',
        'n_train':int(train.sum()),'n_test':int(test.sum()),
        'balanced_accuracy':float(score),
        'versions':{p:metadata.version(p) for p in ['mne','moabb','scikit-learn']}}
out=Path('capstone_results'); out.mkdir(exist_ok=True)
(out/'experiment.json').write_text(json.dumps(record,indent=2))
pd.DataFrame({'truth':y[test],'prediction':pred,'session':sessions[test]}).to_csv(out/'predictions.csv',index=False)
print(json.dumps(record,indent=2))

# %% [markdown]
# ### Inspect and interpret
#
# Open the exported JSON and table, compare them with the notebook configuration and recompute the metric from the saved predictions. A mismatch is a reproducibility bug to fix before submission.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Balanced accuracy from a matrix
#
# Implement macro_recall for a confusion matrix whose true classes are rows, assuming each class has at least one observation.

# %%
def macro_recall(confusion):
    # TODO: average per-row recall.
    return None

# %%
answer=macro_recall(np.array([[8,2],[4,6]]))
if answer is not None:
    assert np.isclose(answer,.7); print('Metric check passed.')
else: print('Exercise pending: implement macro_recall.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Pre-analysis plan
#
# Write a one-page plan with question, data, split, primary metric, baseline, one comparison and a stopping rule for model development.

# %%
# Your investigation: add code here.
# Keep the original data and final test partition intact.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 3 · Run the comparison
#
# Implement the planned alternative using the same partitions. Save configuration and trial predictions; state whether the result supports the hypothesis.

# %%
# Your investigation: add code here.
# Keep the original data and final test partition intact.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 4 · Reproduction exchange
#
# Give the notebook and manifest to another student. Record what they needed to change and whether their outputs agree within expected numerical tolerance.

# %%
# Your investigation: add code here.
# Keep the original data and final test partition intact.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 5 · Model card
#
# Write intended use, data scope, performance, failure cases, computational needs and limitations. Include why the result is not yet a deployed BCI.

# %%
# Your investigation: add code here.
# Keep the original data and final test partition intact.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 6 · Exit presentation
#
# Prepare a five-minute explanation centered on one figure, one comparison and one limitation. Answer which independent dataset or participant group should be tested next.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [Course evaluation guide](../docs/guides/evaluation.md) · [Competition dataset](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
