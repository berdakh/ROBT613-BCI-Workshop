# %% [markdown]
# # 14 · From offline analysis to causal replay
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Process recorded EEG in chunks with persistent filter state and verify that chunking does not change the causal result.
#
# **Data:** Recorded EEGBCI stream; no live hardware
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# An offline decoder sees an entire recording. A live interface receives one chunk at a time. We replay a recording under that constraint and inspect the state, timing and information boundaries needed for a credible online pipeline.
#
# ### By the end you should be able to
#
# - Distinguish causal filtering from zero-phase offline processing.
# - Carry filter state across chunks and test chunk-size invariance.
# - Timestamp a trailing feature window without using future samples.
# - Separate replay correctness from closed-loop BCI performance.
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
# ### Causality is an information constraint
#
# A causal output at time $t$ depends only on samples available at or before $t$. An offline zero-phase filter can use later samples to correct phase delay. That is valuable for descriptive analysis, but it cannot be copied into an online system that has not received those samples yet. An online filter has a response delay and startup behavior that must be included in the interface design.
#
# For an IIR filter,
#
# $$y[n]=\sum_{k=0}^{M}b_kx[n-k]-\sum_{k=1}^{P}a_ky[n-k],$$
#
# assuming a normalized leading denominator coefficient. The dependence on past inputs and outputs is stored in filter state. Second-order sections provide a numerically useful representation. When a chunk ends, retain the returned state and pass it into the next chunk. Resetting at every chunk creates artificial transients at each boundary.

# %% [markdown]
# ### An invariant worth testing
#
# Given identical initial state and ordered samples, filtering a recording in chunks should agree with filtering it in one causal pass, up to numerical tolerance. This is an engineering invariant, not a statistical performance measure. It can catch a state-management bug without requiring any labels.
#
# Chunk size affects delivery cadence and overhead. It should not change the mathematical filtered stream when state is handled correctly. Real acquisition adds dropped samples, timestamp jitter and disconnections, which a simple array replay does not reproduce. Document those limitations before calling the result real-time validation.

# %% [markdown]
# ### Feature windows have their own clock
#
# A trailing window ending at sample $n$ uses samples $n-W+1$ through $n$. Its decision cannot be available before sample $n$ arrives. A two-second window updated every quarter second yields many overlapping feature estimates, not independent new two-second observations. If those estimates enter a train/test split, neighboring windows require careful grouping or separation.
#
# Total latency includes acquisition buffering, the evidence window, filter response, computation and any decision confirmation. Some terms overlap rather than simply add; define the timestamp convention and measure actual end-to-end delay. Reporting only classifier inference time leaves out much of the user's wait.

# %% [markdown]
# ### From a score to a command
#
# A usable interface needs a policy for low confidence and for no-control periods. A single noisy threshold crossing can trigger an unwanted action. Hysteresis uses different entry and exit thresholds; dwell time requires sustained evidence; a refractory period prevents repeated commands. These policies change false activations, missed commands and delay, and their thresholds require calibration.
#
# This notebook deliberately implements causal filtering and trailing features, not a live command classifier. Offline replay confirms code behavior on recorded data. Closed-loop testing additionally asks how users adapt to feedback, whether feedback changes their signals and whether the complete system remains usable. The distinction should remain explicit in a project report.

# %% [markdown]
# ### Worked example · preserve state
#
# Compare a one-pass causal filter with two chunks sharing the final state.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_stream=np.sin(np.arange(1000)*.1)+.2*np.cos(np.arange(1000)*.9)
demo_sos=signal.butter(4,[8,30],btype='bandpass',fs=160,output='sos')
demo_full=signal.sosfilt(demo_sos,demo_stream)
demo_state=np.zeros((len(demo_sos),2))
demo_a,demo_state=signal.sosfilt(demo_sos,demo_stream[:333],zi=demo_state)
demo_b,demo_state=signal.sosfilt(demo_sos,demo_stream[333:],zi=demo_state)
assert np.allclose(np.r_[demo_a,demo_b],demo_full)
print('Chunked and one-pass causal outputs agree.')

# %% [markdown]
# **Read the result.** Both paths use zero initial state. Comparing different initialization conventions would test a different question.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · reset creates a boundary error
#
# Repeat the second chunk without carrying state. Measure the resulting difference.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_reset=np.r_[signal.sosfilt(demo_sos,demo_stream[:333]),signal.sosfilt(demo_sos,demo_stream[333:])]
print('Maximum reset error:',np.max(np.abs(demo_reset-demo_full)))

# %% [markdown]
# **Read the result.** The error is introduced by chunk handling, even though the input samples and filter coefficients are unchanged.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · timestamp the first decision
#
# A trailing two-second window at 128 Hz needs 256 samples. Compute the timestamp of its last sample under a zero-based clock.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_fs=128;demo_W=2*demo_fs
demo_first_end=demo_W-1
print('Last-sample timestamp:',demo_first_end/demo_fs,'s')
print('Nominal window duration:',demo_W/demo_fs,'s')

# %% [markdown]
# **Read the result.** Sample timestamps and acquisition duration differ by one sample interval under this convention. State which quantity appears on the plot.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Checkpoint · overlapping evidence
#
# Calculate how much data two neighboring feature windows share.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_W=256;demo_step=32
print('Shared fraction:',(demo_W-demo_step)/demo_W)
print('Updates per second:',128/demo_step)

# %% [markdown]
# **Read the result.** Four updates per second do not mean four independent pieces of evidence. Fast visual feedback and statistical independence are different properties.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# An online decoder has access only to samples already acquired. A causal IIR filter follows $y[n]=\sum_k b_kx[n-k]-\sum_{k=1}a_ky[n-k]$. Its state summarizes past inputs and outputs; resetting that state at every chunk causes repeated transients.
#
# Total response latency includes acquisition buffering, filter delay, feature-window duration, inference and actuator delay: $T_{total}\approx T_{buffer}+T_{filter}+T_{window}+T_{compute}+T_{actuator}$. A fast classifier cannot compensate for a four-second feature window. Offline zero-phase results are not online performance estimates.
#
# The replay below uses no real-time clock, hardware connection or robot. It verifies the signal-processing implementation against a single-pass causal reference. A deployed BCI additionally needs synchronized event timestamps, calibration, an idle/reject state, drift monitoring and an independent stop control.

# %% [markdown]
# ## Load a recorded stream
# Keep the raw recording in its native sampling rate and inspect one channel for the numerical check.
#
# Keep the native sampling rate and isolate one recorded sensor stream. The filter coefficients are fixed before replay, and all operations are constrained to the arriving samples.

# %%
from mne.datasets import eegbci
files = eegbci.load_data(1, [4], path=DATA_ROOT, update_path=False)
raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
eegbci.standardize(raw)
raw.set_montage('standard_1005')
print(raw)
print('Channel types:', set(raw.get_channel_types()))

stream=raw.get_data(picks=['C3'])[0]
fs=raw.info['sfreq']
sos=signal.butter(4,[8,30],btype='bandpass',fs=fs,output='sos')

# %% [markdown]
# ### Inspect and interpret
#
# Name the filter band and order. Explain which state must persist between calls and how a restart should be documented.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Replay with persistent state
# The equality check would fail if the state were reset inside the loop.
#
# The reference is a one-pass causal filter, not a zero-phase offline signal. Chunk processing passes the returned state forward and is checked for numerical equality before any feature is interpreted.

# %%
reference=signal.sosfilt(sos,stream)
state=np.zeros((sos.shape[0],2))
chunks=[]
for start in range(0,len(stream),32):
    filtered,state=signal.sosfilt(sos,stream[start:start+32],zi=state)
    chunks.append(filtered)
replay=np.concatenate(chunks)
assert np.allclose(replay,reference,rtol=1e-10,atol=1e-14)
print('Maximum chunking error:',np.max(abs(replay-reference)))
fig,ax=plt.subplots()
time=np.arange(len(stream))/fs
ax.plot(time[:1000],stream[:1000]*1e6,label='Raw',alpha=.5)
ax.plot(time[:1000],replay[:1000]*1e6,label='Causal 8–30 Hz')
ax.set(xlabel='Time (s)',ylabel='Voltage (µV)',title='Recorded EEG causal replay')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Explain what the equality assertion establishes and what it cannot establish about real acquisition timing. Compare the boundary error if state is reset.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Measure feature availability
# Each feature uses a trailing window only. This is a feature stream, not a validated online classifier.
#
# Every log-power estimate uses a complete trailing window. The hop sets display/update cadence; the window sets how much past evidence supports one estimate. No classification accuracy is claimed by this feature plot.

# %%
window=int(2*fs); hop=int(.25*fs)
ends=np.arange(window,len(replay)+1,hop)
log_power=np.array([np.log(np.var(replay[end-window:end])+1e-30) for end in ends])
fig,ax=plt.subplots(); ax.plot((ends-1)/fs,log_power)
ax.set(xlabel='Time feature becomes available (s)',ylabel='Log variance (V²)',title='Causal trailing-window features')
plt.show()
print('Window duration:',window/fs,'s; update interval:',hop/fs,'s')

# %% [markdown]
# ### Inspect and interpret
#
# Identify when the first complete feature is available and how strongly neighboring windows overlap. Describe the extra calibration and decision policy needed for commands.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Trailing feature
#
# Implement trailing_variance returning one variance per complete trailing window, with the specified step.

# %%
def trailing_variance(values, window, step):
    # TODO: use only complete windows; ddof=0.
    return None

# %%
answer=trailing_variance(np.arange(6.),3,2)
if answer is not None:
    assert np.allclose(answer,[2/3,2/3]); print('Trailing-window checks passed.')
else: print('Exercise pending: implement trailing_variance.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Chunk invariant
#
# Test causal filtering with chunk sizes 1, 31, 128 and a full stream. Assert equality under a common initial state.

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
# ### Exercise 3 · Startup inspection
#
# Plot the beginning of the causal output and discuss a warm-up policy. Do not silently remove startup samples from latency accounting.

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
# ### Exercise 4 · Decision policy
#
# Implement a simple two-threshold hysteresis rule on simulated scores. Report command count, false triggers and delay for a known simulated target interval.

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
# ### Exercise 5 · Replay limitations
#
# List three phenomena absent from array replay and propose a test for each before classroom use with live acquisition.

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
# Explain why a centered moving average and a trailing moving average have different online information requirements.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [SciPy SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
