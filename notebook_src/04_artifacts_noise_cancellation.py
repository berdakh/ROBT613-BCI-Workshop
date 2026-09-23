# %% [markdown]
# # 04 · Artifacts, referencing and noise cancellation
# 
# **ROBT613 · Brain–Computer Interfaces** | 90–120 min
# 
# ## Goal
# Detect bad channels, compare reference choices, and understand regression and ICA through a controlled contamination experiment.
# 
# **Data:** EEGBCI for QC/ICA; explicit controlled contamination for regression
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
# Common-average reference subtracts $\bar x(t)=C^{-1}\sum_c x_c(t)$ from each EEG channel. It reduces rank by one and assumes the sampled scalp average is useful. A bad channel can contaminate all channels, so inspect channels first. Marking, interpolation and rejection are distinct decisions; never interpolate most of a sparse montage and call it recovered neural data.
# 
# For an observed EEG matrix $X$ and nuisance reference $E$, least squares estimates $B=(E^TE)^{-1}E^TX$ and forms $X_{clean}=X-EB$. It removes all activity correlated with the reference, including genuine neural signal. Fit nuisance coefficients on calibration data only, then apply them unchanged to evaluation data.
# 
# ICA assumes $X=AS$, with approximately independent sources $S$. It estimates an unmixing matrix $W$ and removes selected source contributions before reconstruction. Independence is statistical, not a label saying “artifact.” Inspect scalp maps, time courses, spectra and EOG/ECG association. Fit ICA on a high-pass-filtered copy (commonly 1 Hz) and apply its spatial solution to compatible data with the same channels/reference. Do not remove components solely because their variance is large.

# %% [markdown]
# ## Inspect real channel quality
# EEGBCI has EEG channels but no dedicated EOG channel. We do not invent an EOG sensor or automatically remove an arbitrary ICA component.

# %%
from mne.datasets import eegbci
files = eegbci.load_data(1, [4], path=DATA_ROOT, update_path=False)
raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
eegbci.standardize(raw)
raw.set_montage('standard_1005')
print(raw)
print('Channel types:', set(raw.get_channel_types()))

values = raw.get_data()
quality = pd.DataFrame({'channel':raw.ch_names, 'std_uV':values.std(1)*1e6,
                        'peak_to_peak_uV':np.ptp(values,axis=1)*1e6})
print(quality.sort_values('peak_to_peak_uV',ascending=False).head(10))
raw.copy().compute_psd(fmax=70).plot(show=False)
plt.show()

# %% [markdown]
# ## Controlled reference regression
# Here clean signal and blink reference are known by construction. Calibration is the first half and evaluation the second half; the simulation is not a claim of clinical artifact removal.

# %%
fs=128; t=np.arange(fs*40)/fs
clean = 8e-6*np.sin(2*np.pi*10*t) + rng.normal(0,2e-6,len(t))
eog = sum(100e-6*np.exp(-0.5*((t-center)/.12)**2) for center in range(2,39,3))
observed = clean + .7*eog
cal = t<20
B = np.linalg.lstsq(eog[cal,None],observed[cal],rcond=None)[0]
corrected = observed - eog*B[0]
print('Held-out RMSE before / after (µV):',
      np.sqrt(np.mean((observed[~cal]-clean[~cal])**2))*1e6,
      np.sqrt(np.mean((corrected[~cal]-clean[~cal])**2))*1e6)
fig,ax=plt.subplots()
for z,name in [(observed,'Contaminated'),(corrected,'Corrected'),(clean,'Known clean')]:
    ax.plot(t,z*1e6,label=name,alpha=.8)
ax.set(xlim=(20,24),xlabel='Time (s)',ylabel='Voltage (µV)',title='SIMULATED held-out blink removal')
ax.legend(); plt.show()

# %% [markdown]
# ## Fit ICA and inspect, without automatic exclusion
# This demonstration uses a calibration segment only. Set exclusions only after reviewing evidence; the default removes nothing.

# %%
calibration = raw.copy().crop(tmin=0,tmax=min(60,raw.times[-1])).filter(1,40)
ica = mne.preprocessing.ICA(n_components=15,method='fastica',random_state=SEED,max_iter=1000)
ica.fit(calibration, picks='eeg', decim=3)
ica.plot_components(show=False)
plt.show()
ica.exclude = []  # Intentionally empty: inspection is a scientific decision.
reconstructed = ica.apply(raw.copy())
print('Excluded components:', ica.exclude)

# %% [markdown]
# ## Exercises and checks
# 1. Add an EEG-correlated component to the simulated EOG. What neural information is lost?
# 2. Mark a truly flat channel in a copy, interpolate it, and report the effect of re-referencing.
# 3. Explain why artifact rejection rates must be reported by condition.
# 4. What prevents you from labeling an ICA component as ocular in this recording?

# %% [markdown]
# ## Next steps and sources
# [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) · [EEG reference](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html).
# 
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
