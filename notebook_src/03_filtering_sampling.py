# %% [markdown]
# # 03 · Filtering, sampling and spectral inspection
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Design and inspect EEG filters, distinguish offline and causal processing, and measure attenuation on real EEG.
#
# **Data:** MNE EEGBCI subject 1, run 4
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# A colleague says, “I filtered the EEG at 30 Hz.” You cannot reproduce that statement yet: was it a low-pass or bandpass, what was the transition width, and did the filter use future samples? This lesson turns an informal preprocessing statement into an inspectable signal-processing method.
#
# ### By the end you should be able to
#
# - Relate convolution, impulse response and frequency response.
# - Measure passband behavior, transition width and causal delay.
# - Demonstrate aliasing and explain anti-alias filtering before resampling.
# - Choose different passbands for ERP and sensorimotor-power tasks.
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
# ### Filtering is a transformation with consequences
#
# A filter is not a general “clean EEG” button. It changes amplitude and sometimes phase as a function of frequency. For a sinusoidal input at frequency $f$, a linear time-invariant filter scales its amplitude by $|H(f)|$ and shifts phase by $\arg H(f)$. When a transient contains many frequencies, those changes alter its shape. This is why ERP latency and morphology require careful filter choices.
#
# For a finite impulse response filter, each output is a weighted sum of recent inputs. A moving average has equal weights and suppresses fast changes, but its frequency response has sidelobes rather than an ideal brick wall. Designed FIR filters choose weights to control these tradeoffs. Sharper transitions generally require longer filters. A long filter can extend farther in time than a short teaching epoch.
#
# An IIR filter uses feedback from previous outputs and can achieve a sharp magnitude response with fewer coefficients. Numerical implementation matters: second-order sections are preferable to a single high-order polynomial for many practical designs. Forward–backward filtering applies a filter in both temporal directions; it cancels phase in the interior but requires future data and still has boundary behavior. Zero phase does not mean zero consequences.

# %% [markdown]
# ### Sampling changes what frequencies mean
#
# The sampled sinusoid $\sin(2\pi fn/f_s)$ is indistinguishable from certain sinusoids differing by integer multiples of $f_s$. Frequencies above Nyquist can fold into lower frequencies. Once aliasing has occurred at acquisition, a digital low-pass cannot reconstruct the lost distinction. Before downsampling existing data, attenuate frequencies above the new Nyquist limit; simply retaining every second sample is not a complete resampling method.

# %% [markdown]
# ### A reproducible filter report
#
# Record filter family, passband, transition bands, phase mode, sampling rate, treatment of boundaries and whether filtering preceded epoching. For separate runs, maintain discontinuity annotations or process each run independently. For a temporal train/test split within one run, account for filter support around the boundary. A pipeline can leak samples even when the filter never uses class labels.

# %% [markdown]
# ### Worked example · convolution by hand
#
# A three-sample moving average uses weights [1/3, 1/3, 1/3]. For the sequence [0, 0, 3, 0, 0], predict the full convolution. The response to a single impulse reveals the filter itself.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_input=np.array([0.,0.,3.,0.,0.])
demo_kernel=np.ones(3)/3
demo_output=np.convolve(demo_input,demo_kernel,mode='full')
print('Input:',demo_input,'\nOutput:',demo_output)
assert np.allclose(demo_output,[0,0,1,1,1,0,0])

# %% [markdown]
# **Read the result.** The energy of the sharp event spreads across neighboring samples. Smoothing is a change in temporal resolution, not just removal of noise.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · expose aliasing
#
# At a sample rate of 100 Hz, a cosine at 70 Hz and one at 30 Hz produce the same samples. Compute the two sequences and compare. Cosine avoids the sign reversal that appears in the analogous sine example.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_fs=100
demo_t=np.arange(100)/demo_fs
high=np.cos(2*np.pi*70*demo_t)
aliased=np.cos(2*np.pi*30*demo_t)
print('Maximum difference:',np.max(np.abs(high-aliased)))
assert np.allclose(high,aliased,atol=1e-12)

# %% [markdown]
# **Read the result.** No downstream classifier can determine which original analog cosine generated these samples without additional information. Acquisition anti-alias filtering is essential.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · FIR length versus delay
#
# Hold the passband fixed and change the number of taps. Use the same axes. Calculate the delay for a symmetric causal FIR and compare the transition shapes.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
fig,ax=plt.subplots()
for taps in [33,65,129]:
    h=signal.firwin(taps,[8,30],fs=160,pass_zero=False)
    f_demo,H_demo=signal.freqz(h,fs=160)
    ax.plot(f_demo,20*np.log10(np.maximum(abs(H_demo),1e-8)),
            label=f'{taps} taps; delay {(taps-1)/(2*160):.2f} s')
ax.set(xlim=(0,50),ylim=(-80,5),xlabel='Frequency (Hz)',ylabel='Gain (dB)',title='FIR design tradeoff')
ax.legend();plt.show()

# %% [markdown]
# **Read the result.** A longer filter improves frequency selectivity while spreading information over a longer interval. Decide whether the task can tolerate that support.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · downsampling with and without anti-alias protection
#
# Mix 10 Hz and 70 Hz activity at 200 Hz, then reduce to 100 Hz. Compare direct slicing with a resampling method that includes anti-alias filtering.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_fs=200;demo_t=np.arange(2000)/demo_fs
mixed=np.sin(2*np.pi*10*demo_t)+.8*np.sin(2*np.pi*70*demo_t)
unsafe=mixed[::2]
safer=signal.resample_poly(mixed,1,2)
fig,ax=plt.subplots()
for values,name in [(unsafe,'Slicing only'),(safer,'Anti-alias resampling')]:
    f_demo,p_demo=signal.welch(values,fs=100,nperseg=500)
    ax.plot(f_demo,p_demo,label=name)
ax.set(xlabel='Frequency (Hz)',ylabel='PSD (arbitrary units²/Hz)',title='SIMULATION · 70 Hz folds toward 30 Hz')
ax.legend();plt.show()

# %% [markdown]
# **Read the result.** The extra lower-frequency peak after naive slicing is an artifact of resampling, not a new physiological rhythm.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# A finite impulse response filter computes $y[n]=\sum_{k=0}^{L-1}h[k]x[n-k]$. Its frequency response is $H(f)=\sum_k h[k]e^{-j2\pi fk/f_s}$. The transition band, not just cutoff, controls required length. A symmetric FIR has group delay $(L-1)/(2f_s)$ when used causally.
#
# Forward–backward IIR filtering cancels phase offline but uses future samples and squares the magnitude response. It cannot be copied into a live BCI. Filtering short epochs separately creates edge artifacts; filter each continuous run before epoching and preserve run boundaries. Do not filter across a train/test temporal boundary without a guard region at least covering the effective filter support.
#
# Welch PSD averages windowed periodograms: $\hat P(f)=K^{-1}\sum_k|\mathcal F\{w x_k\}|^2/(f_s\sum_n w[n]^2)$. Units are $V^2/Hz$. Frequency resolution is approximately $f_s/N_{FFT}$ but zero padding does not create new information. Downsampling needs anti-alias filtering. Notch only a measured line component; a 50/60 Hz notch is redundant after a sufficiently strong 30 Hz low-pass.

# %% [markdown]
# ## Load and inspect a real run
# Choose the filter for the intended paradigm. The 8–30 Hz band here is for motor imagery, not P300.
#
# Read one continuous imagery run and standardize channel names so C3 refers to the expected sensor. Keep the original Raw object as the baseline for comparison; filtering a copy makes before/after inspection reproducible.

# %%
from mne.datasets import eegbci
files = eegbci.load_data(1, [4], path=DATA_ROOT, update_path=False)
raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
eegbci.standardize(raw)
raw.set_montage('standard_1005')
print(raw)
print('Channel types:', set(raw.get_channel_types()))

# %% [markdown]
# ### Inspect and interpret
#
# Verify the sampling rate and duration. Identify the Nyquist frequency and check that every requested filter cutoff lies below it.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Compare spectra
# Use the same PSD parameters before and after filtering. Decibels below are referenced to one V²/Hz.
#
# Use identical Welch settings and the same sensor for both PSD curves. A band-pass filter should attenuate out-of-band energy without creating a new physiological interpretation for the remaining peaks. The logarithmic display uses a consistent reference.

# %%
filtered = raw.copy().filter(8,30)
channel = raw.ch_names.index('C3')
fig, ax = plt.subplots()
for obj,name in [(raw,'Before'),(filtered,'8–30 Hz')]:
    f,p = signal.welch(obj.get_data()[channel], fs=obj.info['sfreq'], nperseg=1024)
    ax.plot(f,10*np.log10(np.maximum(p,1e-30)),label=name)
ax.set(xlim=(0,75),xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 V²/Hz)',title='Subject 1, run 4, C3')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Locate the passband, transition regions and attenuated frequencies. Explain why the output PSD is not the filter response itself: it combines the input spectrum with the filter action.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Inspect FIR and causal delay
# An impulse exposes timing that a smooth waveform can hide.
#
# An impulse response and a frequency response describe the same linear filter in different domains. The symmetric FIR has a calculable causal delay; the offline zero-phase operation used by MNE handles phase differently. Resampling changes the time grid and requires anti-alias protection.

# %%
fs = raw.info['sfreq']
h = signal.firwin(129,[8,30],pass_zero=False,fs=fs)
f,H = signal.freqz(h,fs=fs)
fig,axes = plt.subplots(1,2,figsize=(11,4))
axes[0].plot(f,20*np.log10(np.maximum(abs(H),1e-8)))
axes[0].set(xlim=(0,70),ylim=(-90,5),xlabel='Frequency (Hz)',ylabel='Gain (dB)',title='129-tap FIR')
axes[1].plot(np.arange(len(h))/fs,h)
axes[1].set(xlabel='Time (s)',ylabel='Impulse response',title=f'Causal delay: {(len(h)-1)/(2*fs):.3f} s')
plt.tight_layout(); plt.show()
resampled = filtered.copy().resample(100)
assert resampled.info['sfreq'] == 100

# %% [markdown]
# ### Inspect and interpret
#
# Compare the printed delay with the impulse-response center. State which operation could run on an incoming stream and which requires future samples. Verify the new sampling rate after resampling.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Delay calculation
#
# Implement `fir_delay`. Check that doubling the sampling rate halves delay in seconds for the same tap count.

# %%
def fir_delay(n_taps, sampling_rate):
    # TODO: delay in seconds for a symmetric causal FIR.
    return None

# %%
answer=fir_delay(129,160)
if answer is not None:
    assert np.isclose(answer,.4); print('Delay check passed.')
else:
    print('Exercise pending: implement fir_delay.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Filter report
#
# Write a complete report for the real-data filter used in this notebook. Locate MNE’s filter design output and identify details absent from “8–30 Hz”.

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
# ### Exercise 3 · Task-specific design
#
# Design one candidate P300 filter and one motor-imagery filter. Plot their frequency responses and justify the bands in terms of the measured phenomena.

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
# ### Exercise 4 · Notch decision
#
# Inspect line-frequency power before adding a notch. Explain whether a notch remains necessary after your low-pass and what evidence would change your choice.

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
# ### Exercise 5 · Boundary experiment
#
# Filter a short impulse-containing epoch alone and inside a longer padded signal. Compare the retained interior; explain the edge differences.

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
# Explain to a teammate why zero-phase offline filtering cannot be copied unchanged into a real-time decoder. Include future samples and delay in the answer.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [MNE filtering background](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html) · [SciPy signal](https://docs.scipy.org/doc/scipy/reference/signal.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
