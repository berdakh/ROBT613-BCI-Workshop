# %% [markdown]
# # 05 · Segmentation, baseline and quality control
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Build event-locked epochs from continuous EEG, audit discarded trials and preserve run identity.
#
# **Data:** MNE EEGBCI subject 1, run 4
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## The question for today
#
# A classifier reports excellent accuracy, but many more trials were discarded from one class than the other. We trace each trial from its event marker through epoch boundaries, baseline correction and quality control before trusting its label.
#
# ### By the end you should be able to
#
# - Convert an event sample and time interval into epoch boundaries.
# - Explain the difference between baseline correction and high-pass filtering.
# - Audit dropped trials by class and run.
# - Recognize overlap and artifact rejection as possible sources of evaluation bias.
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
# ### Events are the link between measurement and task
#
# An event array has three columns: sample index, previous event value and event code. The code is a lookup key, not a physiological measurement. Its interpretation comes from the dataset protocol. The same annotation name can mean different tasks in different runs; this is why the motor-imagery notebooks select documented run numbers explicitly.
#
# For sampling rate $f_s$ and event sample $s_0$, relative time $t$ maps approximately to $s_0+\operatorname{round}(t f_s)$. MNE also tracks `first_samp`, which matters after cropping or when a file starts at a nonzero acquisition sample. Do not assume that an array position is always an absolute event index. Inspect `raw.first_samp`, the event samples and `epochs.times` together.

# %% [markdown]
# ### What one epoch contains
#
# An epoch is a short multichannel segment aligned to a scientifically meaningful event. Negative times are before that event; positive times are after it. The interval should follow the hypothesis. A P300 response occurs hundreds of milliseconds after a flash, while imagery rhythm changes can extend over seconds. A broad interval increases available context but also increases overlap, artifacts and computational cost.
#
# With both endpoints included, an interval from $a$ to $b$ has approximately
#
# $$T=\operatorname{round}((b-a)f_s)+1$$
#
# samples when the endpoints lie on the sample grid. The actual `epochs.times` array is the authority. A request that extends beyond the recording cannot be satisfied by inventing data; MNE records why such an epoch was dropped.

# %% [markdown]
# ### Baseline subtraction and rejection
#
# Baseline correction subtracts the mean of a selected pre-event interval from every time point of each channel and trial:
#
# $$x'_{nct}=x_{nct}-\frac{1}{|B|}\sum_{u\in B}x_{ncu}.$$
#
# It aligns an offset but does not remove arbitrary drift. The baseline must be relevant to the comparison; a preceding task response can make a supposedly neutral baseline unequal between conditions. Baseline subtraction also introduces dependence between all samples and the baseline estimate.
#
# Peak-to-peak rejection checks $\max_t x_{nct}-\min_t x_{nct}$ against a threshold for each channel. It is sensitive to large excursions but cannot detect every artifact. The threshold is in volts for EEG: confusing volts with microvolts can reject everything or nothing. Our real-data demonstration uses a deliberately explicit threshold and prints retained class counts so the consequence is visible. It is an illustration, not a universal recommended value.
#
# Keep the unmodified epoch collection, the selection indices and the drop log. If rejection differs strongly by class or run, ask whether the task itself caused movement or whether the sensor quality changed over time. More data cleaning can make the surviving sample less representative of future use.

# %% [markdown]
# ### Worked example · sample boundaries
#
# Calculate a segment around an event at sample 1000, with 100 Hz sampling and limits −0.2 to 0.8 s. Predict the number of samples before evaluating.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_event=1000;demo_fs=100
demo_start=demo_event+round(-.2*demo_fs)
demo_stop=demo_event+round(.8*demo_fs)
demo_indices=np.arange(demo_start,demo_stop+1)
print(demo_start,demo_stop,len(demo_indices))
assert len(demo_indices)==101

# %% [markdown]
# **Read the result.** Python slices normally exclude their stop index; MNE epoch time bounds include the final sample when available. This is a common one-sample discrepancy.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Worked example · baseline subtraction
#
# Use an offset trace and a two-sample baseline. Compare the response relative to zero before and after correction.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_trace=np.array([5.,7.,9.,8.,6.])
demo_corrected=demo_trace-demo_trace[:2].mean()
print('Corrected:',demo_corrected)
assert np.isclose(demo_corrected[:2].mean(),0)
assert np.isclose(np.ptp(demo_trace),np.ptp(demo_corrected))

# %% [markdown]
# **Read the result.** Subtracting a constant changes the offset but not peak-to-peak range. Baseline correction and amplitude rejection therefore answer different questions.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · unequal rejection
#
# Create a small audit table. Calculate retention within each class, rather than only the total number of surviving epochs.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_audit=pd.DataFrame({'class':['left']*5+['right']*5,'keep':[1,1,1,1,1,1,0,0,1,0]})
print(demo_audit.groupby('class')['keep'].agg(['sum','count','mean']))

# %% [markdown]
# **Read the result.** The retained data have a different class composition from the original data. Report per-class retention with the modeling results.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ### Guided experiment · overlapping windows
#
# Two windows can share raw samples even when they have different row indices. Compute their intersection.
#
# **Before running:** state your prediction and the assumption behind it.

# %%
demo_window_a=set(range(0,200))
demo_window_b=set(range(100,300))
print('Shared samples:',len(demo_window_a & demo_window_b))
print('Fraction of each window shared:',len(demo_window_a & demo_window_b)/200)

# %% [markdown]
# **Read the result.** Randomly splitting these windows leaks part of the same recording into both sets. Use blocks or runs and, where necessary, a gap between partitions.
#
# **Pause and explain:** point to one computed value that supports this interpretation.

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# A cue at sample $s_i$ becomes an epoch $X_i[:,n]=x[:,s_i+n]$ over a specified interval. Event latency is part of the measurement: stimulus display delay and clock drift cannot be corrected by classification. MNE accounts for `raw.first_samp`; do not subtract it twice.
#
# Baseline correction is $X_i'(c,t)=X_i(c,t)-|B|^{-1}\sum_{u\in B}X_i(c,u)$. It assumes the reference interval is appropriate and can propagate baseline noise across the full trial. Motor imagery decoding often uses bandpassed post-cue power without ERP baseline subtraction. Peak-to-peak rejection removes an epoch when $\max_t X_i(c,t)-\min_t X_i(c,t)>\theta_c$.
#
# Rejection changes the analyzed population. Report counts before and after, by class and run, and separate boundary drops from amplitude drops. Overlapping sliding windows from one trial must remain in one split group. Preserve subject, session, run and original trial identifiers before any concatenation.

# %% [markdown]
# ## Parse annotations explicitly
# The T1/T2 mapping depends on the run. Run 4 is imagined left/right fist.
#
# Explicit event mapping prevents rest or a different task from entering the two imagery classes. The epoch collection is created without amplitude rejection so that the effect of the later threshold remains measurable.

# %%
from mne.datasets import eegbci
files = eegbci.load_data(1, [4], path=DATA_ROOT, update_path=False)
raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
eegbci.standardize(raw)
raw.set_montage('standard_1005')
print(raw)
print('Channel types:', set(raw.get_channel_types()))

events,event_id=mne.events_from_annotations(raw,event_id={'T1':1,'T2':2})
print(event_id, events[:5])
raw.filter(1,30)
base = mne.Epochs(raw,events,{'left':1,'right':2},-.5,3.5,
                  baseline=None,preload=True,picks='eeg',reject_by_annotation=True)
print('Candidate events:',len(events),'in-bounds epochs:',len(base))

# %% [markdown]
# ### Inspect and interpret
#
# Count candidate epochs by condition before rejection and inspect their time limits. Confirm that negative times represent the pre-cue interval.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Reject and audit
# The threshold is a declared teaching choice, not a universal EEG quality standard.
#
# Apply rejection to a copy and compare retained counts against the original collection. The declared threshold is deliberately visible. `drop_log` documents why candidates were removed; the selection indices connect survivors back to their source events.

# %%
threshold = 500e-6
checked = base.copy().drop_bad(reject={'eeg':threshold})
rows=[]
for name,event in base.event_id.items():
    before=np.sum(base.events[:,2]==event)
    after=np.sum(checked.events[:,2]==event)
    rows.append({'class':name,'before':int(before),'after':int(after),'removed':int(before-after)})
print(pd.DataFrame(rows))
print('Drop reasons:', checked.drop_log[:8])
assert len(checked)<=len(base)
fig,ax=plt.subplots()
ax.hist(np.ptp(base.get_data(),axis=-1).max(axis=1)*1e6,bins=15)
ax.axvline(threshold*1e6,color='red',label='Threshold')
ax.set(xlabel='Maximum channel peak-to-peak (µV)',ylabel='Epoch count',title='Rejection diagnostic')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Compare retained fractions by class, not just total count. If all epochs disappear, stop and investigate units, threshold and recording quality instead of proceeding with an empty classifier input.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Verify a baseline numerically
# Use this for understanding ERP baselines, not as a mandatory motor-imagery step.
#
# Baseline subtraction is demonstrated separately from the main imagery pipeline. The numerical check verifies the selected baseline mean, illustrating an invariant that can catch an incorrect time mask or axis.

# %%
baseline_epochs=base.copy().apply_baseline((-.5,0))
mask=(baseline_epochs.times>=-.5)&(baseline_epochs.times<=0)
assert np.allclose(baseline_epochs.get_data()[:,:,mask].mean(-1),0,atol=1e-12)
print('Baseline mean is numerically zero.')

# %% [markdown]
# ### Inspect and interpret
#
# Explain why zero baseline mean does not imply zero post-event drift. Identify which values changed and which peak-to-peak ranges should remain unchanged.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Independent practice
#
# Work through the tasks in order. Exercise 1 includes a small implementation check; passing it verifies the stated example, not every possible input. For the investigations, save a labeled figure or table and a short explanation. Use copies of data objects when changing preprocessing, and preserve any held-out evaluation partition.
#
# **Submission:** your completed notebook, the requested outputs, and a brief exit-ticket response. The notebook runs before exercises are completed; “pending” means your work is still required.

# %% [markdown]
# ### Exercise 1 · Baseline function
#
# Implement baseline_center for a trial × channel × time array and a Boolean time mask.

# %%
def baseline_center(values, mask):
    # TODO: subtract one baseline mean per trial and channel.
    return None

# %%
demo_cube=np.arange(24.).reshape(2,3,4)
demo_mask=np.array([True,True,False,False])
answer=baseline_center(demo_cube,demo_mask)
if answer is not None:
    assert answer.shape==demo_cube.shape
    assert np.allclose(answer[...,demo_mask].mean(axis=-1),0)
    assert np.allclose(answer[...,3]-answer[...,2],1)
    print('Baseline checks passed.')
else: print('Exercise pending: implement baseline_center.')

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ### Exercise 2 · Event audit
#
# Print the annotation dictionary and condition counts before epoching. Explain every event code included and excluded.

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
# ### Exercise 3 · Threshold curve
#
# Try at least four rejection thresholds on copies of the same original epochs. Plot retained fraction by condition. Choose a threshold based on evidence, not the highest test score.

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
# ### Exercise 4 · Baseline sensitivity
#
# Compare no baseline with two plausible intervals. Show the effect on an ERP and explain which claim depends on the baseline.

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
# ### Exercise 5 · Drop-log investigation
#
# Inspect at least one dropped trial or document why none were dropped. Distinguish data-boundary loss from amplitude rejection.

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
# Write a short reproducibility record including event mapping, interval, baseline, threshold, retained counts and split grouping.

# %% [markdown]
# **Your response:**
#
# - Prediction or rationale: _write here_
# - Evidence from your result: _write here_
# - Interpretation and limitation: _write here_

# %% [markdown]
# ## Next steps and sources
# [MNE epochs](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html) · [Epoch rejection](https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
