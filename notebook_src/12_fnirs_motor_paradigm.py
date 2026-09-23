# %% [markdown]
# # 12 · Beyond EEG: fNIRS motor responses
# 
# **ROBT613 · Brain–Computer Interfaces** | 90–120 min
# 
# ## Goal
# Convert optical intensity to hemoglobin changes and compare a slow hemodynamic paradigm with electrophysiological BCI.
# 
# **Data:** Native MNE fNIRS motor dataset, participant 1
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
# fNIRS measures changes in detected near-infrared light related to absorption by oxygenated and deoxygenated hemoglobin. It does not measure neuronal voltage. Optical density is $\Delta OD_\lambda=-\ln[I_\lambda(t)/I_{\lambda,0}]$. The modified Beer–Lambert model is $\Delta OD_\lambda=\sum_j\epsilon_{\lambda j}\Delta c_j\,d\,DPF_\lambda$. With multiple wavelengths, solve this linear system for relative HbO/HbR concentration changes.
# 
# Source–detector distance and differential pathlength assumptions affect the estimate. Scalp blood flow, motion and systemic physiology can dominate. A low-frequency bandpass does not guarantee removal of superficial signals; short-separation regression is a separate technique. The hemodynamic response develops over seconds, so EEG timing and filtering choices cannot simply be reused.
# 
# This native MNE recording contains executed finger tapping and a control condition. It is not a motor-imagery BCI benchmark. The lesson supports the syllabus’s broader physiological-signal objective while keeping that distinction explicit.

# %% [markdown]
# ## Load and convert intensity
# Use documented annotation labels and retain source geometry. PPF is an explicit teaching assumption.

# %%
from mne.preprocessing.nirs import (optical_density,beer_lambert_law,
                                      scalp_coupling_index,source_detector_distances)
path=mne.datasets.fnirs_motor.data_path(path=DATA_ROOT,update_path=False)
intensity=mne.io.read_raw_nirx(path/'Participant-1',preload=True,verbose=False)
intensity.annotations.rename({'1.0':'Control','2.0':'Left','3.0':'Right'})
intensity.annotations.delete(np.where(intensity.annotations.description=='15.0')[0])
distance=source_detector_distances(intensity.info)
intensity.pick(np.where(distance>.01)[0])
od=optical_density(intensity)
sci=scalp_coupling_index(od)
od.info['bads']=[ch for ch,s in zip(od.ch_names,sci) if s<.5]
print('Low-coupling channels:',od.info['bads'])
haemo=beer_lambert_law(od,ppf=.1)
haemo.filter(.05,.7,h_trans_bandwidth=.2,l_trans_bandwidth=.02)

# %% [markdown]
# ## Epoch the slow response
# Use a long post-event interval and report how channel-quality choices affect interpretation.

# %%
events,event_id=mne.events_from_annotations(haemo,event_id={'Control':1,'Left':2,'Right':3})
epochs=mne.Epochs(haemo,events,event_id,-5,15,baseline=(-5,0),preload=True)
fig,ax=plt.subplots()
for name in ['Control','Left','Right']:
    ev=epochs[name].average(picks='hbo')
    ax.plot(ev.times,ev.data.mean(0)*1e6,label=name)
ax.set(xlabel='Time after instruction (s)',ylabel='Mean HbO change (µmol/L)',title='Motor execution · descriptive fNIRS response')
ax.legend(); plt.show()

# %% [markdown]
# ## Exercises and checks
# 1. Compare HbO and HbR and explain why their responses need not be perfectly opposite.
# 2. Explain why a 1-second decision window is a poor default here.
# 3. Plan a hybrid EEG–fNIRS experiment and specify synchronization and leakage controls.
# 4. Identify which short-distance channels were excluded and explain how regression would differ from simple exclusion.

# %% [markdown]
# ## Next steps and sources
# [MNE fNIRS processing](https://mne.tools/stable/auto_tutorials/preprocessing/70_fnirs_processing.html).
# 
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
