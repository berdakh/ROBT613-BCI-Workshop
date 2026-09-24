# %% [markdown]
# # 04 · Artifacts, referencing and noise cancellation
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Detect bad channels, compare reference choices, and understand regression and ICA through a controlled contamination experiment.
#
# **Data:** EEGBCI for QC/ICA; explicit controlled contamination for regression
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# A frontal channel contains large slow deflections. Is the correct response a high-pass filter, a new reference, trial rejection, regression or ICA? We separate the evidence for an artifact from the operation used to reduce it.
#
# ### By the end you should be able to
#
# - Distinguish temporal filtering, spatial referencing, rejection and nuisance regression.
# - Calculate how a bad channel contaminates an average reference.
# - Explain what ICA can and cannot identify from independence.
# - Evaluate cleaning on data not used to fit the cleaning model.
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
# ### Diagnose before correcting
#
# EEG is a mixture of activity of interest and other electrical contributions. “Noise” is defined relative to the question: an eye movement is useful in an eye-tracking experiment but can confound a motor-imagery decoder. Large amplitude alone does not prove that a component is non-neural. Start with channel names/types, time traces, spectra and the relation to task events. Ask whether the anomaly is confined to one sensor, one interval or a recurring pattern across channels.
#
# | Observation | Candidate explanation | First investigation |
# |---|---|---|
# | One almost constant channel | Poor contact or disconnected lead | Trace and channel variance |
# | Sharp narrow spectral peak | Mains interference | PSD and local mains frequency |
# | Large frontal slow waves | Blinks or eye motion | Frontal traces and measured EOG if available |
# | High-frequency bursts | Muscle activity or movement | Time-frequency distribution and task timing |
# | Abrupt shared step | Movement or recording discontinuity | Annotations and acquisition notes |
#
# A candidate explanation is not a diagnosis. A preprocessing record should state the evidence, decision and quantity removed. Filtering attenuates frequencies; it cannot separate two signals in the same band. Rejection removes observations. Regression subtracts the part predictable from measured nuisance channels. ICA changes the coordinate system so that a recurring mixture can sometimes be isolated more clearly.

# %% [markdown]
# ### Referencing is a linear transformation
#
# For a column vector of $C$ channel measurements, common-average referencing is $x_r=Rx$ with
#
# $$R=I-\frac{1}{C}\mathbf1\mathbf1^T.$$
#
# The channels sum to zero afterward, so one degree of freedom is lost. This is expected, not a software failure. If one channel has a large artifact, subtracting its contribution to the average spreads that artifact into all other channels. Inspect and mark bad sensors before choosing the reference. Scalp topographies are sensor-level summaries; this course does not infer anatomical generators.

# %% [markdown]
# ### Regression and ICA require assumptions
#
# The least-squares nuisance model $X=EB+U$ assumes that a stable linear combination of nuisance measurements explains contamination. If $E$ contains task-related brain activity, the same subtraction can remove the response we want. Low residual variance is therefore insufficient evidence of successful cleaning. Use a controlled mixture when you need ground truth, and inspect task-related changes when processing real recordings.
#
# ICA instead seeks statistically independent component time courses. Its component numbering, signs and scales are arbitrary. Evidence such as a frontal map, blink-shaped time course and correlation with a genuine EOG channel supports an ocular interpretation. None of these should be replaced with a rule such as “remove component zero.” The example below deliberately leaves exclusions empty because this recording has no dedicated EOG channel. Students practice building an evidence record rather than accepting an automatic label.

# %% [markdown]
# ### Worked example · a bad sensor changes every reference
#
# Predict the mean of each time column before running. The third sensor has an artificial offset; inspect how subtracting the average affects the two initially quiet sensors.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_channels=np.array([[1.,2.,1.],[2.,1.,2.],[90.,90.,90.]])
demo_ref=demo_channels-demo_channels.mean(axis=0,keepdims=True)
print('Before:\n',demo_channels,'\nAverage referenced:\n',demo_ref)
assert np.allclose(demo_ref.mean(axis=0),0)

# %% [markdown]
# **Read the result.** A zero channel average is a mathematical property, not evidence that the recording is clean. The contaminated channel changes the reference for every channel.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · rank after average reference
#
# Construct the reference matrix explicitly. Compare its rank with the number of channels and apply it twice.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_C=4
demo_R=np.eye(demo_C)-np.ones((demo_C,demo_C))/demo_C
print('Rank:',np.linalg.matrix_rank(demo_R),'of',demo_C)
assert np.allclose(demo_R@demo_R,demo_R)

# %% [markdown]
# **Read the result.** The second application changes nothing: this is a projection. Covariance-based methods must account for the resulting rank.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · regression can remove wanted signal
#
# Let a nuisance reference contain a component perfectly correlated with a desired oscillation. Even though subtraction reduces variance, examine its effect on the desired oscillation.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_t=np.arange(1280)/128
demo_wanted=np.sin(2*np.pi*10*demo_t)
demo_nuisance=demo_wanted+.3*np.sin(2*np.pi*2*demo_t)
demo_observed=demo_wanted+.5*demo_nuisance
demo_b=np.linalg.lstsq(demo_nuisance[:,None],demo_observed,rcond=None)[0]
demo_residual=demo_observed-demo_nuisance*demo_b[0]
print('Variance before / after:',np.var(demo_observed),np.var(demo_residual))
print('Wanted-wave coefficient before / after:',np.dot(demo_observed,demo_wanted)/np.dot(demo_wanted,demo_wanted),np.dot(demo_residual,demo_wanted)/np.dot(demo_wanted,demo_wanted))

# %% [markdown]
# **Read the result.** The desired coefficient should be one in the uncontaminated signal. A much smaller residual coefficient shows why minimizing amplitude is the wrong objective.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Checkpoint · correlation is not identity
#
# A mixture can correlate strongly with an artifact reference and still contain useful activity. Calculate the association and describe what extra evidence would be needed before removal.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_reference=np.linspace(-1,1,100)
demo_useful=np.sin(np.linspace(0,8*np.pi,100))
demo_mixture=3*demo_reference+demo_useful
print('Reference correlation:',np.corrcoef(demo_mixture,demo_reference)[0,1])
print('Useful activity variance:',np.var(demo_useful))

# %% [markdown]
# **Read the result.** Correlation measures association. It neither labels every sample as artifact nor guarantees that all useful variance is preserved by subtraction.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# Common-average reference subtracts $\bar x(t)=C^{-1}\sum_c x_c(t)$ from each EEG channel. It reduces rank by one and assumes the sampled scalp average is useful. A bad channel can contaminate all channels, so inspect channels first. Marking, interpolation and rejection are distinct decisions; never interpolate most of a sparse montage and call it recovered neural data.
#
# For an observed EEG matrix $X$ and nuisance reference $E$, least squares estimates $B=(E^TE)^{-1}E^TX$ and forms $X_{clean}=X-EB$. It removes all activity correlated with the reference, including genuine neural signal. Fit nuisance coefficients on calibration data only, then apply them unchanged to evaluation data.
#
# ICA assumes $X=AS$, with approximately independent sources $S$. It estimates an unmixing matrix $W$ and removes selected source contributions before reconstruction. Independence is statistical, not a label saying “artifact.” Inspect scalp maps, time courses, spectra and EOG/ECG association. Fit ICA on a high-pass-filtered copy (commonly 1 Hz) and apply its spatial solution to compatible data with the same channels/reference. Do not remove components solely because their variance is large.

# %% [markdown]
# ## Inspect real channel quality
# EEGBCI has EEG channels but no dedicated EOG channel. We do not invent an EOG sensor or automatically remove an arbitrary ICA component.
#
# The channel table ranks variation and extreme excursions as screening aids. Neither column is an automatic bad-channel label. EEGBCI lacks dedicated EOG here, so channel-type inspection prevents us from pretending an ocular reference was measured.

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
# ### Inspect and interpret
#
# Select a suspicious channel and inspect its waveform and spectrum before making a decision. Note whether the issue is sustained, intermittent or shared across sensors.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Controlled reference regression
# Here clean signal and blink reference are known by construction. Calibration is the first half and evaluation the second half; the simulation is not a claim of clinical artifact removal.
#
# The controlled mixture supplies known ground truth, making an RMSE comparison meaningful. Only the first half fits the nuisance coefficient; the second half assesses whether that coefficient transfers. This is a calibration/evaluation separation for preprocessing itself.

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
# ### Inspect and interpret
#
# Compare corrected and known-clean traces on the held-out interval. If you change the nuisance relationship after calibration, predict the direction of failure before rerunning.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Fit ICA and inspect, without automatic exclusion
# This demonstration uses a calibration segment only. Set exclusions only after reviewing evidence; the default removes nothing.
#
# ICA is fitted on a filtered calibration segment. The spatial decomposition can then be inspected, but exclusions remain empty until evidence supports a decision. This avoids teaching component index as an artifact label.

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
# ### Inspect and interpret
#
# Record the component evidence you would request: sensor map, time course, spectrum and genuine nuisance-channel association if available. An empty exclusion list means the applied reconstruction removes no chosen component.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Implement the reference
#
# Implement average_reference for a channel × time array without modifying the input.

# %%
def average_reference(values):
    # TODO: channel axis is zero; preserve input.
    return None

# %%
answer=average_reference(demo_channels)
if answer is not None:
    assert answer.shape==demo_channels.shape
    assert np.allclose(answer.mean(axis=0),0)
    assert np.allclose(answer[0]-answer[1],demo_channels[0]-demo_channels[1])
    print('Reference checks passed.')
else: print('Exercise pending: implement average_reference.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · QC report
#
# Select three channels from the real QC table. Compare their traces and spectra; write evidence for keeping or marking each, without relying only on rank in a table.

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
# ### Exercise 3 · Regression stress test
#
# Change the contamination coefficient between calibration and evaluation in the controlled example. Report held-out RMSE and explain the stationarity assumption.

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
# ### Exercise 4 · ICA evidence sheet
#
# Choose one component and record its sensor map, temporal behavior and spectrum. State whether the evidence warrants removal or remains inconclusive.

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
# ### Exercise 5 · Compare decisions
#
# Contrast rejecting a blink-contaminated epoch with removing an ICA component. Identify what each operation changes and how it can bias class balance.

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
# Why can a cleaner-looking trace produce a worse or misleading BCI? Give one signal-loss mechanism and one confounding mechanism.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) · [EEG reference](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
