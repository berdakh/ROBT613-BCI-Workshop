# %% [markdown]
# # 13 · Neural networks and autoencoders for EEG
# 
# **ROBT613 · Brain–Computer Interfaces** | 120–180 min
# 
# ## Goal
# Train a small temporal CNN and a denoising autoencoder using training-only normalization; compare what their losses actually measure.
# 
# **Data:** EEGBCI subject 1, runs 4/8/12; optional PyTorch
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
# A temporal convolution computes $h_{k,t}=\phi(b_k+\sum_{c,\tau}W_{kc\tau}X_{c,t-\tau})$. Weight sharing reduces parameter count relative to a fully connected network. Cross-entropy $L=-N^{-1}\sum_i\log p_\theta(y_i|X_i)$ rewards correct class probabilities. Dropout and weight decay regularize but do not replace independent evaluation.
# 
# An autoencoder maps $z=f_\theta(X)$ and $\hat X=g_\phi(z)$, minimizing $\|X-\hat X\|^2$. A denoising autoencoder receives deliberately corrupted $\tilde X$ while targeting $X$. Real EEG used as a target is not artifact-free ground truth; low reconstruction error can preserve artifacts or suppress discriminative transients. A bottleneck is a modeling assumption, not evidence of neural source recovery.
# 
# Use one run for training, one for validation and one for testing. Fit channel scaling on training samples only. Select the best epoch from validation loss and evaluate the test run once. This small-data exercise is not intended to establish deep-learning superiority over CSP/LDA.

# %% [markdown]
# ## Load EEG and PyTorch
# CPU execution is sufficient. PyTorch is optional for the rest of the course.

# %%
from mne.datasets import eegbci
runs = [4, 8, 12]  # ALL are imagined left versus right fist, not hands versus feet
parts, groups = [], []
for run in runs:
    paths = eegbci.load_data(1, [run], path=DATA_ROOT, update_path=False)
    raw_run = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
    eegbci.standardize(raw_run)
    raw_run.set_montage('standard_1005')
    raw_run.set_eeg_reference('average', projection=False)
    raw_run.filter(8, 30, fir_design='firwin')
    events, _ = mne.events_from_annotations(raw_run, event_id={'T1': 1, 'T2': 2})
    ep = mne.Epochs(raw_run, events, {'left': 1, 'right': 2}, tmin=0.5,
                    tmax=3.5, baseline=None, preload=True, picks='eeg',
                    reject_by_annotation=True)
    parts.append(ep)
    groups.extend([run] * len(ep))
epochs = mne.concatenate_epochs(parts)
X = epochs.get_data(copy=True)
y = epochs.events[:, 2] - 1
groups = np.asarray(groups)
print('Epochs:', X.shape, 'run counts:', pd.Series(groups).value_counts().to_dict())

import importlib.util
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch'])
import torch
from torch import nn
torch.manual_seed(SEED)
torch.set_num_threads(2)
tr=groups==4; va=groups==8; te=groups==12
mu=X[tr].mean(axis=(0,2),keepdims=True)
sd=X[tr].std(axis=(0,2),keepdims=True).clip(1e-12)
Z=torch.tensor((X-mu)/sd,dtype=torch.float32)
Y=torch.tensor(y,dtype=torch.long)
tr_t=torch.tensor(tr); va_t=torch.tensor(va); te_t=torch.tensor(te)
print('Train/validation/test:',tr.sum(),va.sum(),te.sum())

# %% [markdown]
# ## Train a small CNN
# Global average pooling yields a fixed-sized feature vector. Validation selects a checkpoint, never a test result.

# %%
import copy
net=nn.Sequential(nn.Conv1d(X.shape[1],16,15,padding=7),nn.ELU(),
                  nn.AvgPool1d(4),nn.Dropout(.25),nn.Conv1d(16,16,9,padding=4),
                  nn.ELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(16,2))
opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.01)
loss_fn=nn.CrossEntropyLoss()
history=[]; best_loss=float('inf'); best=None
for epoch in range(30):
    net.train(); opt.zero_grad()
    loss=loss_fn(net(Z[tr_t]),Y[tr_t]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): val=loss_fn(net(Z[va_t]),Y[va_t]).item()
    history.append((loss.item(),val))
    if val<best_loss: best_loss=val; best=copy.deepcopy(net.state_dict())
net.load_state_dict(best); net.eval()
with torch.no_grad(): pred=net(Z[te_t]).argmax(1).numpy()
print('Test-run balanced accuracy:',balanced_accuracy_score(y[te],pred))
fig,ax=plt.subplots(); ax.plot(history)
ax.set(xlabel='Epoch',ylabel='Cross-entropy',title='CNN learning curves')
ax.legend(['Training','Validation']); plt.show()

# %% [markdown]
# ## Train a denoising autoencoder
# Only training epochs are used. The target is recorded EEG, not verified clean EEG. Evaluate added-noise reconstruction on the validation set, leaving the test set for the classifier result above.

# %%
ae=nn.Sequential(nn.Conv1d(X.shape[1],8,9,padding=4),nn.ELU(),
                 nn.Conv1d(8,X.shape[1],9,padding=4))
optimizer=torch.optim.Adam(ae.parameters(),lr=.003)
for _ in range(20):
    corrupted=Z[tr_t]+.2*torch.randn_like(Z[tr_t])
    optimizer.zero_grad(); loss=((ae(corrupted)-Z[tr_t])**2).mean()
    loss.backward(); optimizer.step()
ae.eval()
with torch.no_grad():
    noisy=Z[va_t]+.2*torch.randn_like(Z[va_t])
    reconstructed=ae(noisy)
    print('Validation MSE: noisy input',((noisy-Z[va_t])**2).mean().item(),
          'autoencoder',((reconstructed-Z[va_t])**2).mean().item())
print('A worse reconstruction score is a valid outcome; do not claim denoising succeeded automatically.')

# %% [markdown]
# ## Exercises and checks
# 1. Compare the CNN with CSP/LDA using exactly runs 4/8/12 as train/validation/test.
# 2. Compute the network parameter count and compare it with the number of training trials.
# 3. Why is optimizing autoencoder reconstruction insufficient to establish useful BCI features?
# 4. Repeat seeds and report every run; do not choose the best seed on test accuracy.
# 5. Design a nested experiment that trains the autoencoder only within each training fold.

# %% [markdown]
# ## Next steps and sources
# [PyTorch training basics](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) · [MNE EEGBCI](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html).
# 
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
