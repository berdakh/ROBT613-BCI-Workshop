# %% [markdown]
# # 07 · Time–frequency analysis and ERD/ERS
# 
# **ROBT613 · Brain–Computer Interfaces** | 90–120 min
# 
# ## Goal
# Compute wavelet power, normalize it to a pre-cue interval and interpret sensorimotor desynchronization.
# 
# **Data:** MNE EEGBCI subject 1, run 4
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
# The Morlet wavelet is a complex sinusoid multiplied by a Gaussian: $\psi_f(t)=e^{j2\pi ft}e^{-t^2/(2\sigma_t^2)}$. Time-frequency power is $P(f,t)=|x*\psi_f|^2$. With $n_c$ cycles, $\sigma_t\approx n_c/(2\pi f)$: more cycles sharpen frequency resolution but blur timing.
# 
# Baseline-relative power is $10\log_{10}[P(f,t)/P_B(f)]$ in dB. Negative values indicate less power than baseline (ERD); positive values indicate more (ERS). This is a ratio of power, not voltage. Averaging power before normalizing is not generally equivalent to averaging trialwise ratios. We explicitly average trial power first.
# 
# Convolution uses data around each point. Pad epochs and interpret only their interior. A late cue-locked difference can be physiological, while a pre-cue difference can come from filtering/wavelet temporal spread or experimental anticipation. No inferential significance is implied by a color plot.

# %% [markdown]
# ## Load longer epochs for padding
# Keep a pre-cue reference and an interior analysis window.

# %%
from mne.datasets import eegbci
files = eegbci.load_data(1, [4], path=DATA_ROOT, update_path=False)
raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
eegbci.standardize(raw)
raw.set_montage('standard_1005')
print(raw)
print('Channel types:', set(raw.get_channel_types()))

raw.filter(2,40)
events,_=mne.events_from_annotations(raw,event_id={'T1':1,'T2':2})
epochs=mne.Epochs(raw,events,{'left':1,'right':2},-1,4,baseline=None,
                  picks=['C3','C4'],preload=True,reject_by_annotation=True)

# %% [markdown]
# ## Compute and normalize power
# The baseline is −0.8 to −0.2 s. We show −0.5 to 3.5 s to reduce edge interpretation.

# %%
freqs=np.arange(6,31,2)
power=epochs.compute_tfr(method='morlet',freqs=freqs,n_cycles=freqs/2,
                         return_itc=False,average=False,decim=4)
P=power.data
baseline=(power.times>=-.8)&(power.times<=-.2)
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,(name,event) in zip(axes,epochs.event_id.items()):
    mean=P[epochs.events[:,2]==event,0].mean(0)
    db=10*np.log10(mean/mean[:,baseline].mean(1,keepdims=True))
    im=ax.pcolormesh(power.times,freqs,db,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto')
    ax.set(xlim=(-.5,3.5),xlabel='Time after cue (s)',title=f'C3 · {name}')
axes[0].set_ylabel('Frequency (Hz)')
fig.colorbar(im,ax=axes,label='Power relative to baseline (dB)'); plt.show()

# %% [markdown]
# ## Exercises and checks
# 1. Replace `freqs/2` cycles by 3 and then 10. Explain the resolution tradeoff.
# 2. Compare C3 and C4 without claiming localization from two scalp electrodes.
# 3. Derive the percent ERD formula and convert −3 dB into a power ratio.
# 4. Explain why testing every pixel requires multiple-comparison control.

# %% [markdown]
# ## Next steps and sources
# [MNE time-frequency analysis](https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html).
# 
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
