# %% [markdown]
# # 08 · BCI Competition IV 2a: CSP and LDA
# 
# **ROBT613 · Brain–Computer Interfaces** | 120–150 min
# 
# ## Goal
# Use actual competition motor-imagery data, learn spatial filters inside training folds and evaluate transfer across sessions.
# 
# **Data:** BCI Competition IV 2a / BNCI2014-001, subject 1
# 
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

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
# ## Concepts and mathematics
# 
# Common spatial patterns finds directions with different class variances. For two class covariances $C_1,C_2$, solve $C_1w=\lambda(C_1+C_2)w$. Eigenvectors at opposite ends emphasize either class. A typical feature is $z_j=\log\mathrm{Var}(w_j^TX)$; optional variance normalization changes the definition. MNE supports multiclass CSP with approximate joint diagonalization; the binary eigenproblem explains the intuition but is not the complete four-class algorithm.
# 
# CSP is supervised: fitting it on all epochs before cross-validation leaks test labels. Keep CSP and LDA in one scikit-learn pipeline. Spatial **filters** transform sensors into components; spatial **patterns** describe how components project to sensors and are usually more suitable for physiological interpretation.
# 
# BNCI2014-001 corresponds to BCI Competition IV dataset 2a: left hand, right hand, feet and tongue imagery. It is fetched by MOABB and represented as MNE epochs. The original competition split and a modern cross-session reanalysis are not identical evaluation claims. Here session 1 is calibration and session 2 is held out; we report a course reanalysis, not an official competition submission.

# %% [markdown]
# ## Load competition data
# Use one participant for a bounded CPU lesson. Expand to all nine participants only after validating the protocol.

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

# %% [markdown]
# ## Tune on training runs, test once
# Shrinkage stabilizes covariance estimates. Component count is selected only inside the calibration session.

# %%
from mne.decoding import CSP
pipeline=make_pipeline(CSP(n_components=4,reg='ledoit_wolf',log=True,norm_trace=False),
                        LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
inner=GroupKFold(n_splits=3)
search=GridSearchCV(pipeline,{'csp__n_components':[2,4,6]},cv=inner,
                    scoring='balanced_accuracy',n_jobs=1)
search.fit(X[train],y[train],groups=groups[train])
pred=search.predict(X[test])
print('Training-only choice:',search.best_params_)
print('Held-out session balanced accuracy:',balanced_accuracy_score(y[test],pred))
ConfusionMatrixDisplay.from_predictions(y[test],pred,xticks_rotation=45)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Inspect learned patterns
# These maps use the calibration-fitted CSP only. They do not establish a causal neural generator.

# %%
csp=search.best_estimator_.named_steps['csp']
fig, axes = plt.subplots(1, csp.n_components, figsize=(3*csp.n_components, 3))
for component, ax in enumerate(np.atleast_1d(axes)):
    mne.viz.plot_topomap(csp.patterns_[component], epochs.info, axes=ax, show=False)
    ax.set_title(f'CSP pattern {component+1}')
fig.suptitle('Calibration-fitted patterns · arbitrary component scale')
plt.show()
print('Training epochs:',int(train.sum()),'test epochs:',int(test.sum()))

# %% [markdown]
# ## Exercises and checks
# 1. Compare CSP+LDA with the log-bandpower baseline under exactly the same sessions.
# 2. Why is fitting CSP before the split a serious leak?
# 3. Repeat for another subject and report both scores, not only the better one.
# 4. Extend to Competition IV 2b with `BNCI2014_004`; verify its labels and session protocol first.

# %% [markdown]
# ## Next steps and sources
# [Competition 2a via MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) · [MNE CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html).
# 
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
