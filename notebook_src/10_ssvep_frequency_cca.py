# %% [markdown]
# # 10 · SSVEP: spectral peaks and canonical correlation
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Decode 12 versus 15 Hz visual stimulation using spectral signal-to-noise ratio and a sinusoidal-reference CCA decoder.
#
# **Data:** Native MNE SSVEP dataset, participant 02
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# A participant attends to one of two flickering targets. Can a few seconds of occipital EEG identify the attended frequency? We compare spectral evidence with a multichannel correlation method and examine the assumptions behind a fast decision.
#
# ### By the end you should be able to
#
# - Relate observation duration to frequency discrimination.
# - Build sine/cosine reference matrices with harmonics.
# - Explain canonical correlation as a comparison of two multivariate signals.
# - Distinguish benchmark trial classification from a usable asynchronous interface.
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
# ### The response follows a periodic stimulus
#
# A steady-state visual evoked potential (SSVEP) is a response associated with periodic visual stimulation. Energy can appear at the stimulation frequency and its harmonics. An SSVEP interface assigns candidate frequencies to selectable targets, then compares the recorded response with those frequencies. This differs from P300 decoding, which detects a transient response to individual flashes, and from motor imagery, which uses changes in internally modulated rhythms.
#
# The demonstration uses an existing MNE SSVEP recording. It does not present flashing stimuli. Target frequencies, event codes and channel names come from the dataset protocol; they must not be guessed from the strongest peak in the test data. Occipital channels are selected before evaluating labels.

# %% [markdown]
# ### Duration sets a frequency scale
#
# For an observation window of duration $T$, Fourier bins are spaced approximately $\Delta f=1/T$. A longer interval helps distinguish nearby periodic components but delays a decision. A taper reduces leakage from a finite window at the cost of broadening the spectral peak. Zero padding interpolates the spectrum without extending the observation duration.
#
# A simple score compares target-bin power with nearby background power. The exact choice of neighboring bins matters: exclude the target and a small guard region, and avoid including another target or harmonic in the background estimate. A large ratio supports frequency-specific energy, but muscle activity, display artifacts and spontaneous alpha can also affect the spectrum.

# %% [markdown]
# ### CCA combines channels and reference waveforms
#
# For a candidate frequency $f$, construct reference columns
#
# $$Y_f(t)=[\sin(2\pi ft),\cos(2\pi ft),\ldots,\sin(2\pi Hft),\cos(2\pi Hft)].$$
#
# Sine and cosine allow an arbitrary phase at each harmonic. Canonical correlation analysis chooses linear combinations of EEG channels and reference columns that maximize their correlation:
#
# $$\rho_f=\max_{a,b}\frac{a^T\Sigma_{XY_f}b}{\sqrt{a^T\Sigma_{XX}a}\sqrt{b^T\Sigma_{Y_fY_f}b}}.$$
#
# The predicted target is the candidate with the largest correlation. Basic sinusoidal-reference CCA fits this association within each trial without using that trial's label. This is different from fitting a supervised model to all labeled trials. If channel selection, harmonic count or thresholds are chosen using accuracy, however, those choices require a separate calibration partition.

# %% [markdown]
# ### Interpret success at the right scale
#
# Two known targets and a small set of cued trials form a constrained classification problem. A practical interface must also handle no-control periods, gaze shifts, artifacts, uncertain evidence, latency and feedback. Reporting forced-choice accuracy alone hides false activations when the user intends nothing. A reject option can trade fewer wrong commands for longer or incomplete selections; select its threshold on calibration data and report both accepted fraction and accuracy among accepted decisions.

# %% [markdown]
# ### Worked example · duration and frequency spacing
#
# Calculate the bin spacing for short and long windows. Identify which durations provide a bin separation smaller than 1 Hz.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
for duration in [.5,1,2,4]:
    print(f'{duration:g} s -> {1/duration:g} Hz bin spacing')

# %% [markdown]
# **Read the result.** Bin spacing is a useful scale, not a guarantee that two noisy signals can be resolved perfectly.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · phase-invariant references
#
# Represent a phase-shifted 12 Hz signal as a weighted sum of sine and cosine references.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_t=np.arange(256)/128
demo_Y=np.column_stack([np.sin(2*np.pi*12*demo_t),np.cos(2*np.pi*12*demo_t)])
demo_wave=np.sin(2*np.pi*12*demo_t+.7)
demo_coef=np.linalg.lstsq(demo_Y,demo_wave,rcond=None)[0]
print('Reference weights:',demo_coef)
assert np.allclose(demo_Y@demo_coef,demo_wave)

# %% [markdown]
# **Read the result.** Using sine alone would unnecessarily assume a fixed phase. The two-column basis spans every phase at this frequency.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · harmonics are distinct features
#
# Construct two harmonics and inspect their shapes. The samples are rows, matching scikit-learn’s observation convention.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_ref=np.column_stack([wave(2*np.pi*h*12*demo_t) for h in [1,2] for wave in [np.sin,np.cos]])
print('Reference shape:',demo_ref.shape)
print('Matrix rank:',np.linalg.matrix_rank(demo_ref))

# %% [markdown]
# **Read the result.** A harmonic reference uses multiples of the fundamental. Check that all requested harmonics remain below the sampling Nyquist frequency.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Checkpoint · accuracy after rejection
#
# Apply two confidence thresholds to fixed illustrative decisions. Report the accepted fraction as well as accepted accuracy.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_conf=np.array([.2,.4,.6,.8,.9])
demo_correct=np.array([False,True,False,True,True])
for threshold in [.3,.7]:
    accepted=demo_conf>=threshold
    print('Threshold:',threshold,'Coverage:',accepted.mean(),'Accepted accuracy:',demo_correct[accepted].mean())

# %% [markdown]
# **Read the result.** Higher accepted accuracy can result from refusing more trials. A usable interface needs both quantities and a policy for rejected selections.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# SSVEP uses periodic visual stimulation to produce narrowband responses near the stimulus frequency and its harmonics. This differs from the transient ERP and self-paced motor-imagery paradigms. This recording is a frequency-tagging experiment, not a many-character online speller.
#
# A local spectral SNR is $SNR(f)=P(f)/\mathrm{mean}_{g\in\mathcal N(f)}P(g)$, where neighboring bins exclude the center and guard bins. Guard bins avoid counting spectral leakage as background. A spectral maximum can be contaminated by spontaneous alpha, so occipital channels and harmonics help.
#
# Canonical correlation chooses $a,b$ to maximize $\rho=\mathrm{corr}(X^Ta,Y_f^Tb)$, where $Y_f$ contains $\sin(2\pi hft)$ and $\cos(2\pi hft)$ for harmonics $h$. The frequency with the largest canonical correlation is selected. Fitting CCA to an individual test epoch and its known reference is part of this label-free decoder, not supervised training on its true label. Selecting channels, harmonics or window length using test accuracy would still leak.

# %% [markdown]
# ## Load native MNE SSVEP data
# The event mapping follows the dataset’s documented stimulus codes. Keep one decision per original trial.
#
# Read the documented BrainVision recording, attach sensor geometry and map the two stimulus codes to their known frequencies. The epoch starts after stimulus onset to focus on a sustained response.

# %%
path=mne.datasets.ssvep.data_path(path=DATA_ROOT,update_path=False)
fname=path/'sub-02'/'ses-01'/'eeg'/'sub-02_ses-01_task-ssvep_eeg.vhdr'
raw=mne.io.read_raw_brainvision(fname,preload=True,verbose=False)
raw.set_montage('easycap-M1')
raw.set_eeg_reference('average',projection=False)
raw.filter(1,40)
events,_=mne.events_from_annotations(raw,event_id={'Stimulus/S255':12,'Stimulus/S155':15})
epochs=mne.Epochs(raw,events,{'12 Hz':12,'15 Hz':15},1,5,baseline=None,
                  picks=['O1','Oz','O2'],preload=True)
X=epochs.get_data(copy=True)
y=epochs.events[:,2]
print('Original-trial decisions:',len(y), 'classes:',np.unique(y))

# %% [markdown]
# ### Inspect and interpret
#
# Confirm the retained channels, frequencies and duration. Do not reinterpret a code using whichever peak is strongest in a particular test trial.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Measure spectral evidence
# A four-second window has approximately 0.25 Hz native resolution.
#
# Average spectra over the selected occipital channels, then compare candidate-frequency bins with nearby background bins. The guard region keeps target leakage out of the immediate noise estimate.

# %%
f,P=signal.welch(X,fs=epochs.info['sfreq'],nperseg=X.shape[-1],axis=-1)
P=P.mean(axis=1)
snr=[]
for target in [12,15]:
    idx=np.argmin(abs(f-target))
    neighbors=np.r_[idx-5:idx-1,idx+2:idx+6]
    snr.append(P[:,idx]/P[:,neighbors].mean(axis=1))
snr=np.asarray(snr).T
pred=np.array([12,15])[snr.argmax(1)]
print('Fixed SNR rule balanced accuracy:',balanced_accuracy_score(y,pred))
fig,ax=plt.subplots()
for target in [12,15]:
    ax.plot(f,10*np.log10(P[y==target].mean(0)),label=f'{target} Hz stimulation')
ax.set(xlim=(5,35),ylim=(-145,-95),xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 V²/Hz)',title='Occipital SSVEP spectra')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Locate both target frequencies and their possible harmonics. Examine score margins as well as forced-choice labels; a correct low-margin decision is still uncertain.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Implement CCA classification
# Use two harmonics and identical reference length for each trial.
#
# For each candidate, construct sine/cosine harmonics over exactly the epoch’s time grid. CCA finds maximally associated EEG and reference combinations within that trial, then the strongest candidate wins.

# %%
from sklearn.cross_decomposition import CCA
def cca_score(epoch, frequency, sfreq, harmonics=2):
    t=np.arange(epoch.shape[-1])/sfreq
    reference=np.stack([fn(2*np.pi*h*frequency*t)
                        for h in range(1,harmonics+1) for fn in (np.sin,np.cos)],axis=1)
    cca=CCA(n_components=1,max_iter=1000)
    u,v=cca.fit_transform(epoch.T,reference)
    return abs(np.corrcoef(u[:,0],v[:,0])[0,1])
score=np.array([[cca_score(epoch,freq,epochs.info['sfreq']) for freq in [12,15]] for epoch in X])
pred=np.array([12,15])[score.argmax(1)]
print('Fixed CCA rule balanced accuracy:',balanced_accuracy_score(y,pred))
ConfusionMatrixDisplay.from_predictions(y,pred); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Compare spectral and CCA predictions on the same trial set. If both are perfect on this small benchmark, discuss why that does not establish performance during rest or among many targets.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Reference builder
#
# Implement harmonic_reference returning samples × (2 × harmonics), alternating sine and cosine.

# %%
def harmonic_reference(times, frequency, harmonics=2):
    # TODO: one sine/cosine pair per harmonic.
    return None

# %%
answer=harmonic_reference(np.arange(128)/128,12,2)
if answer is not None:
    assert answer.shape==(128,4)
    assert np.allclose(answer[:,0],np.sin(2*np.pi*12*np.arange(128)/128))
    assert np.allclose(answer[0],[0,1,0,1]); print('Reference checks passed.')
else: print('Exercise pending: implement harmonic_reference.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Window-length study
#
# Compare 1, 2 and 4 s windows on the same trials using fixed channels and harmonics. Plot accuracy against observation duration.

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
# ### Exercise 3 · Spectral audit
#
# For a correctly and an incorrectly decoded trial, show target frequencies and background bins. If there are no errors, use the smallest score-margin trials.

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
# ### Exercise 4 · Harmonic ablation
#
# Compare one versus two harmonics without selecting the winner on the final test labels. Explain how you would reserve calibration data.

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
# ### Exercise 5 · No-control design
#
# Propose a dataset and evaluation for false activations during rest. State why the provided two-class benchmark cannot establish this result.

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
# Compare the roles of event timing, calibration labels and observation duration in P300, imagery and SSVEP decoding.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [MNE SSVEP tutorial and provenance](https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html) · [CCA API](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
