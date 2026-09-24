# %% [markdown]
# # 06 · Motor imagery: a first decoder
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Decode imagined left versus right hand movement with spectral features and leave-one-run-out validation.
#
# **Data:** MNE EEGBCI subject 1, imagery runs 4/8/12
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# Can imagined left- and right-hand movement be distinguished by the power of sensorimotor rhythms? We first turn a sinusoid into a defensible numerical feature, then test a simple model on a different recording run.
#
# ### By the end you should be able to
#
# - Connect sinusoid amplitude, variance and power spectral density.
# - Integrate band power with the correct frequency-bin width.
# - Build a trial × feature table from channel × time epochs.
# - Explain why scaling and classifier fitting belong inside grouped validation.
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
# ### From motor physiology to a measurable feature
#
# Motor imagery involves imagining movement without performing it. Changes in sensorimotor mu and beta activity can be informative, but they are not a guaranteed switch present in every trial. C3 and C4 are useful sensor-level landmarks over opposite sides of the scalp. The expected relationship is a hypothesis to test, not a reason to relabel an inconvenient result. Eye movements, muscle tension or cue differences can also predict class.
#
# A zero-mean sinusoid $x(t)=A\sin(2\pi ft)$ has mean-square power $A^2/2$ over complete cycles. Doubling amplitude quadruples power. For arbitrary EEG, a power spectral density distributes variance across frequency. A density in V²/Hz must be integrated over frequency to yield V²:
#
# $$P_{[f_1,f_2]}\approx\sum_{k\in K}S_{xx}(f_k)\Delta f.$$
#
# For uniformly spaced frequencies, `sum(psd[mask]) * df` is a rectangular approximation. Trapezoidal integration is another approximation; state the convention and keep it consistent. A sum without the frequency step changes when spectral resolution changes.

# %% [markdown]
# ### Welch estimation and its tradeoff
#
# Welch's method divides the signal into overlapping windows, applies a taper, estimates each spectrum and averages the estimates. Averaging stabilizes the spectrum at the cost of temporal localization. Longer windows give more closely spaced frequency bins, but fewer windows are available for averaging in a fixed-length epoch. Zero padding adds interpolated spectral samples; it does not create new independent information.
#
# Mu and beta boundaries vary across studies and participants. The fixed 8–12 Hz and 13–30 Hz bands here establish a reproducible baseline. If you tune their edges, do so using training folds. Selecting bands from the held-out run converts that run into development data.

# %% [markdown]
# ### Why logarithms and a linear classifier?
#
# Band powers are nonnegative and often highly skewed. Taking $z=\log(P+\epsilon)$ compresses large values and turns multiplicative differences into additive differences. The small positive floor protects the logarithm; it is a numerical safeguard, not an extra source of power. A change of units adds a constant in log space, so the feature convention must be preserved at deployment.
#
# For logistic regression, the score is $s=w^Tz+b$ and the estimated probability is $\sigma(s)=1/(1+e^{-s})$. The model learns weights from labeled training trials. Standardization uses training means and standard deviations; the test run is transformed using those fixed quantities. A pipeline ensures that cross-validation refits both scaling and the model inside each fold.
#
# Our unit of generalization is a held-out run from one participant. It is weaker than testing a new participant or a new day. Keep that scope in the conclusion even if a score looks impressive.

# %% [markdown]
# ### Worked example · amplitude versus power
#
# Compare two complete-cycle signals differing only in amplitude. Predict the power ratio.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_t=np.arange(1000)/100
demo_small=np.sin(2*np.pi*10*demo_t)
demo_large=2*demo_small
print('Power ratio:',np.mean(demo_large**2)/np.mean(demo_small**2))
assert np.isclose(np.var(demo_large)/np.var(demo_small),4)

# %% [markdown]
# **Read the result.** Amplitude and power have different units and scale differently. A factor of two in voltage is not a factor of two in power.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · integrate a density
#
# Estimate the power of a known 10 Hz sinusoid by integrating its Welch spectrum.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_f,demo_psd=signal.welch(demo_small,fs=100,nperseg=200)
demo_power=np.sum(demo_psd)*(demo_f[1]-demo_f[0])
print('Time-domain mean square:',np.mean(demo_small**2),'Integrated PSD:',demo_power)
assert np.isclose(demo_power,.5,atol=.01)

# %% [markdown]
# **Read the result.** Agreement provides a unit and normalization check before computing physiological band features.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · construct a feature row
#
# Build mu and beta powers for three artificial channels. Read the output shape aloud and associate each value with its channel and band.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_signals=np.stack([demo_small,2*demo_small,np.sin(2*np.pi*20*demo_t)])
demo_f,demo_psd=signal.welch(demo_signals,fs=100,nperseg=200,axis=-1)
demo_features=np.stack([demo_psd[:,(demo_f>=lo)&(demo_f<=hi)].sum(axis=1)*(demo_f[1]-demo_f[0]) for lo,hi in [(8,12),(13,30)]],axis=1)
print(pd.DataFrame(demo_features,index=['channel A','channel B','channel C'],columns=['mu','beta']))

# %% [markdown]
# **Read the result.** The channel × band matrix becomes one flattened feature row for one trial. Preserve the column order when applying a trained model.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Checkpoint · log power differences
#
# Take two powers with a known ratio. Show why their log difference depends on the ratio rather than the absolute scale.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_p=np.array([1.,4.])
print('Natural-log difference:',np.diff(np.log(demo_p))[0])
print('Power ratio in dB:',10*np.log10(demo_p[1]/demo_p[0]))
assert np.isclose(np.diff(np.log(demo_p))[0],np.log(4))

# %% [markdown]
# **Read the result.** Natural log and decibels are related but not identical numerical features. Name the transformation instead of writing simply “normalized power.”
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# Sensorimotor rhythms include mu (roughly 8–13 Hz) and beta (roughly 13–30 Hz) activity. Motor imagery can change their power; activity over contralateral sensorimotor cortex often desynchronizes. These are tendencies, not a guaranteed pattern for every participant. Eye, muscle and cue-related signals can also predict the task.
#
# Band power integrates a PSD: $P_{c,[a,b]}=\int_a^b\hat P_c(f)df$. A feature $z_c=\log(P_c+\epsilon)$ reduces skew. Logistic regression maps $p(y=1|z)=\sigma(w^Tz+b)$ and minimizes cross-entropy plus a penalty on $w$. Standardization must learn its mean and scale from training data only.
#
# Runs 4, 8 and 12 in PhysioNet EEGBCI all represent imagined left/right fist. Runs 6, 10 and 14 instead use hands/feet; mixing them under one T1/T2 label would change the task. We test transfer to a held-out run of the same participant, not to a new person.

# %% [markdown]
# ## Load runs and preserve groups
# Continuous filtering is performed independently within each run before epoching.
#
# Each run is filtered independently before epoch extraction. Run identifiers are attached to every trial and retained after concatenation. This preserves the grouping needed to evaluate transfer beyond the recording block used for fitting.

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

# %% [markdown]
# ### Inspect and interpret
#
# Verify all three run identifiers appear and that both classes are represented in each. Read the concatenated shape and confirm labels and groups have one entry per epoch.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Extract mu and beta power
# Use C3, Cz and C4 to keep the first model interpretable; this channel set is declared in advance.
#
# Welch spectra are estimated along time for selected sensors. Integrating each predeclared band and taking log power forms an interpretable feature vector. The plot is an inspection tool, not a replacement for held-out evaluation.

# %%
picks=[epochs.ch_names.index(ch) for ch in ['C3','Cz','C4']]
f,psd=signal.welch(X[:,picks,:],fs=epochs.info['sfreq'],nperseg=256,axis=-1)
features=np.concatenate([np.log(np.maximum(np.trapezoid(psd[:,:,(f>=a)&(f<=b)],f[(f>=a)&(f<=b)],axis=-1),1e-30))
                         for a,b in [(8,13),(13,30)]],axis=1)
print('Feature matrix:',features.shape)
fig,ax=plt.subplots()
for label,name in [(0,'Left'),(1,'Right')]:
    ax.plot(f,10*np.log10(psd[y==label,0].mean(0)),label=name)
ax.set(xlim=(5,35),xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 V²/Hz)',title='C3 · descriptive class spectra')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Name every feature column and its transformation. Check whether individual-trial distributions overlap even if class means differ. Strong overlap is normal in noisy physiological data.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Evaluate run transfer
# Three folds are descriptive repeated measurements, not three independent participants.
#
# The scaler and logistic model are fitted afresh for each grouped fold. Leaving out a complete run asks whether the learned feature relationship survives a new block of recording.

# %%
model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
cv=GroupKFold(n_splits=3)
result=cross_validate(model,features,y,groups=groups,cv=cv,
                      scoring=['balanced_accuracy','roc_auc'],return_train_score=False)
print(pd.DataFrame({k:v for k,v in result.items() if k.startswith('test_')}))
assert np.isfinite(result['test_balanced_accuracy']).all()

# %% [markdown]
# ### Inspect and interpret
#
# Report the individual run scores and their spread. Do not treat three folds from one person as three independent participants.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Power feature
#
# Implement mean_square on the final axis, returning one number per channel/trial. Explain when it equals variance.

# %%
def mean_square(values):
    # TODO: reduce the final (time) axis.
    return None

# %%
answer=mean_square(np.array([[1.,-1.],[2.,-2.]]))
if answer is not None:
    assert np.allclose(answer,[1.,4.]); print('Power checks passed.')
else: print('Exercise pending: implement mean_square.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Band-power audit
#
# For one real epoch, compare integrated PSD with time-domain variance. Account for the filter and the limited integration band.

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
# ### Exercise 3 · Channel comparison
#
# Plot class distributions of C3 and C4 mu features using training data. Label units and overlap; avoid judging separability from means alone.

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
# ### Exercise 4 · Ablation
#
# Compare mu-only, beta-only and combined features using exactly the same grouped splits. Report each run score, not only the mean.

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
# ### Exercise 5 · Confound audit
#
# State two non-neural mechanisms that could distinguish imagery labels and propose a measurement or control for each.

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
# ### Exercise 6 · Exit ticket
#
# Describe exactly which future observations the reported evaluation represents, and which it does not.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [EEGBCI dataset and run mapping](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html) · [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
