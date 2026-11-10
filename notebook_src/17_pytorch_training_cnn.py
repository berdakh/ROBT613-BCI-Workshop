# %% [markdown]
# # 17 · Deep learning with PyTorch II: training networks and a CNN for EEG
#
# **ROBT613 · Brain–Computer Interfaces** | PyTorch track, part 2 of 3 | Academic tutorial and independent exercises
#
# ## Goal
# Train the Lesson 16 network with gradient descent, train a convolutional network on raw motor-imagery epochs, choose models on a validation run, test once, and save a reusable checkpoint.
#
# **Data:** PhysioNet EEGBCI, subjects 1–5, imagery runs 4 (train), 8 (validation), 12 (test)
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required. This lesson adapts the classic "Training neural networks" MNIST tutorial to EEG: the fully connected digit classifier becomes a band-power classifier and the image CNN becomes a temporal–spatial EEG CNN.

# %% [markdown]
# ## Paradigm background and experimental design
#
# ### Motor imagery as a learning problem
#
# Imagining a left or right fist movement reduces mu and beta power over the opposite sensorimotor cortex. Lesson 16 summarized this as log band power and passed it through an untrained network. Training adjusts the weights so the network's probability for the correct class rises on trials whose labels it has seen. The scientific question is whether that improvement carries over to a recording block the network has never seen.
#
# ### Acquisition provenance and instructional protocol
#
# PhysioNet EEGBCI: 64 scalp channels, 160 Hz, BCI2000. Runs 4, 8 and 12 contain cued left/right fist imagery, about 15 trials per class per run. Subjects 1–5 are pooled; each run is filtered to 8–30 Hz and resampled to 80 Hz, and epochs cover 0.5–3.5 s after the cue.
#
# **Acquisition reference:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
#
# | Run | Role | What it may influence |
# |---|---|---|
# | 4 | Training | Normalization statistics and network weights |
# | 8 | Validation | Choice of epoch (checkpoint) and comparison between models |
# | 12 | Test | Nothing; used once at the end to report performance |
#
# Pooling five participants increases the number of training trials, but the evaluation is still within-participant across runs: every participant appears in all three runs. A claim about new participants would need a leave-subject-out design.
#
# ### Measurement model and interpretation
#
# Scalp EEG measures potential differences in volts that mix neural, physiological and environmental sources. A network can exploit eye or muscle activity that co-varies with the cue. A test score shows how well the selected model transfers to run 12 under this protocol. On its own it does not show which physiological source the model uses.
#
# ### Mathematical definitions for this lesson
#
# For trials $(X_i,y_i)$ the cross-entropy loss is $\mathcal L(\theta)=-\frac1N\sum_i\log p_\theta(y_i\mid X_i)$. Gradient descent updates every parameter by $\theta\leftarrow\theta-\eta\,\partial\mathcal L/\partial\theta$ with learning rate $\eta$. Backpropagation computes these derivatives with the chain rule, from the loss back to each layer. A 2-D convolution with $C_{in}$ input maps, $C_{out}$ output maps and a $k_h\times k_w$ kernel has $C_{out}(C_{in}k_hk_w+1)$ parameters.
#
# **Methodological reading:** [Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization](https://doi.org/10.1002/hbm.23730); [Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://doi.org/10.1088/1741-2552/aace8c).

# %% [markdown]
# ## How to study this notebook
#
# This is both the classroom lesson and the independent-study workbook. Everything needed for the exercises—questions, hints, executable solutions, checks and explanations—is here. Work from top to bottom in a fresh runtime.
#
# 1. Read the question and calculate a small example on paper.
# 2. Write your prediction before running the next code cell.
# 3. Complete the task in its workspace.
# 4. Continue to the worked solution and compare the reasoning as well as the numerical result.
# 5. Change one parameter and explain what the result means.
#
# **For a live class:** pause at each "Independent exercise" heading. The solution follows in the same notebook.
#
# **Prerequisites:** Lesson 16 (tensors, `nn.Module`, softmax) and Lesson 09 (validation design).

# %% [markdown]
# ## Analytical objectives
#
# ### Learning outcomes
#
# - Explain loss, gradient descent, learning rate and backpropagation, and use autograd to compute a gradient.
# - Write a training loop with `zero_grad`, `backward` and `step`, using a `DataLoader` for mini-batches.
# - Use a validation run to choose a checkpoint and compare models; touch the test run once.
# - Build a convolutional network whose layers act along time and across channels, and trace tensor shapes through it.
# - Save and reload a model together with everything needed to use it on new EEG.
#
# **How to work:** predict each result before running it, execute one cell at a time, and write a short interpretation.

# %% [markdown]
# ## Setup
# The first cell installs the tested core versions in Colab. Colab already includes PyTorch; locally, `pip install torch` once. A failed download is an error, never silently replaced by synthetic data.

# %%
# Colab: install before importing numerical libraries. Restart if pip requests it.
import sys, subprocess, os, importlib.util
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'mne==1.13.2', 'moabb==1.7.2'])
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch'])
import copy, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
SEED = 613
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = Path(os.environ.get('BCI_DATA', './mne_data')).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MNE_DATA'] = str(DATA_ROOT)
mne.set_log_level('WARNING')
plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
import importlib.metadata as metadata
print({p: metadata.version(p) for p in ['mne', 'numpy', 'scikit-learn', 'torch']})
print('Device:', device, '| dataset cache:', DATA_ROOT)

# %% [markdown]
# ## Visual route through the lesson
#
# Follow the arrows before running the analysis. For each box, say what the input represents, what changes, and what must be preserved.

# %%
# Drawing code for the lesson map; no analysis data are transformed here.
from matplotlib.patches import FancyBboxPatch
map_steps=['Run 4 trials\nfit scaling + weights', 'Mini-batches\nforward → loss', 'backward → step\nupdate weights', 'Run 8 validation\nkeep best epoch', 'Run 12 test once\nsave checkpoint']
fig,map_ax=plt.subplots(figsize=(12,3.1),constrained_layout=True)
map_ax.set(xlim=(-.1,12),ylim=(-.3,2.4));map_ax.axis('off')
for map_i,map_label in enumerate(map_steps):
    map_x=map_i*2.4
    map_ax.add_patch(FancyBboxPatch((map_x,.45),2.05,1.1,
        boxstyle='round,pad=0.08',facecolor='#edf3f7',edgecolor='#35688a',linewidth=1.5))
    map_ax.text(map_x+1.025,1.02,map_label,ha='center',va='center',fontsize=10)
    map_ax.text(map_x+1.025,1.83,str(map_i+1),ha='center',weight='bold',color='#35688a')
    if map_i<4:map_ax.annotate('',xy=(map_x+2.3,1),xytext=(map_x+2.13,1),arrowprops=dict(arrowstyle='->',lw=1.5))
map_ax.text(5.9,-.08,'Steps 2–4 repeat every epoch. Only run 4 changes the weights; run 12 is read once.',ha='center',fontsize=10)
map_ax.set_title('Lesson 17 · training with an honest evaluation boundary',fontsize=14,pad=12)
plt.show()

# %% [markdown]
# **Read the map:** training loops over steps 2–4 many times. Validation chooses which epoch's weights to keep but never updates them. The test run is used after every choice has been made. The map as text: Run 4 fits scaling and weights → mini-batches give a loss → backward and step update weights → run 8 keeps the best epoch → run 12 is tested once and the model is saved.

# %% [markdown]
# ## Symbols and a calculation by hand
#
# $\theta$: trainable parameters; $\mathcal L$: loss; $\eta$: learning rate; $g=\partial\mathcal L/\partial\theta$: gradient.
#
# ### Derive the operation before calling the library
#
# Minimize $\mathcal L(w)=(w-3)^2$ starting from $w=0$ with $\eta=0.1$. The gradient is $2(w-3)=-6$, so the first step gives $w=0-0.1(-6)=0.6$. The next gradient is $2(0.6-3)=-4.8$, giving $w=1.08$. Each step removes 20% of the remaining distance to the minimum at 3.
#
# Backpropagation applies the chain rule. For $y=3(x+1)^2$ and $z=\text{mean}(y)$ over four elements, $\partial z/\partial x=\frac14\cdot6(x+1)$. At $x=1$ this is $3$, the value PyTorch's autograd returns below.

# %% [markdown]
# ### Your paper calculation
#
# Compute the third gradient-descent step for $\mathcal L(w)=(w-3)^2$. Then repeat the first step with $\eta=1.1$ and describe what happens to the distance from the minimum.
#
# **My calculation:** _write your intermediate steps here before continuing._

# %% [markdown]
# ### Loss, gradients and the training loop
#
# The loss measures how badly the network predicts the training labels. Its gradient points in the direction of steepest increase, so stepping against it lowers the loss. In PyTorch every training step has the same three calls: `optimizer.zero_grad()` clears gradients left from the previous step, `loss.backward()` computes new gradients for every parameter, and `optimizer.step()` changes the parameters. Gradients accumulate by default, so forgetting `zero_grad` adds old gradients to new ones.
#
# Mini-batch gradient descent uses a random subset of training trials for each step. One pass over all training trials is an epoch. Shuffling the training trials each epoch decorrelates consecutive batches; validation and test trials are not shuffled because their order does not matter and should stay traceable.
#
# ### Why a validation run, and why only one test
#
# Training loss keeps falling as the network memorizes run 4. Validation loss on run 8 eventually stops falling, which shows where further training no longer transfers. Keeping the weights from the epoch with the lowest validation loss is called early stopping. Because run 8 guided that choice, its score is optimistic; run 12, untouched until the end, gives the honest estimate.
#
# ### From image convolutions to EEG convolutions
#
# The MNIST CNN slides 5 × 5 kernels over image rows and columns. An EEG trial is a channels × time grid, but its two axes mean different things. The network below therefore uses two kinds of convolution. A temporal convolution with a $1\times25$ kernel acts like a learned band-pass filter on each channel. A spatial convolution with a $64\times1$ kernel then combines all channels, like a learned spatial filter (compare CSP in Lesson 08). Average pooling over time summarizes power-like activity, and a linear layer maps the result to two logits. This design follows the "shallow" ConvNet family of Schirrmeister et al. (2017).
#
# ### GPU and reproducibility
#
# `model.to(device)` moves parameters to a GPU when one is available; every input tensor must then move to the same device. This lesson runs in seconds on a CPU. Fixed seeds make runs repeatable on one machine, but CPU and GPU kernels, and library versions, can still change the last digits.

# %% [markdown]
# ## Visual intuition · The learning rate decides whether descent converges
#
# **Independent exercise:** Before running, predict which of the three learning rates overshoots the minimum and which never reaches it.

# %%
vis_rates=[0.1,0.9,1.05]
fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
vis_grid=np.linspace(-1,7,200)
axes[0].plot(vis_grid,(vis_grid-3)**2,color='0.6')
for vis_lr,vis_color in zip(vis_rates,['#35688a','#b87714','#a23b3b']):
    vis_w=[0.0]
    for _ in range(12): vis_w.append(vis_w[-1]-vis_lr*2*(vis_w[-1]-3))
    vis_w=np.array(vis_w)
    axes[0].plot(vis_w[:6],(vis_w[:6]-3)**2,'o-',color=vis_color,ms=4,label=f'η = {vis_lr}')
    axes[1].plot(np.abs(vis_w-3),'o-',color=vis_color,ms=4,label=f'η = {vis_lr}')
axes[0].set(xlim=(-1,7),ylim=(0,20),xlabel='Weight w',ylabel='Loss (w − 3)²',title='First five steps')
axes[1].set(yscale='log',xlabel='Step',ylabel='|w − 3|',title='Distance from the minimum')
axes[1].legend()
plt.show()

# %% [markdown]
# ### Worked interpretation
#
# With $\eta=0.1$ the weight approaches 3 steadily. With $\eta=0.9$ it jumps past the minimum each step but still converges. With $\eta=1.05$ each step overshoots by more than it corrects and the distance grows. Real networks have many parameters and a curved loss surface, but the same trade-off holds: too small is slow, too large diverges.

# %% [markdown]
# ## Guided practice 1 · gradient descent by hand
#
# Run five steps of gradient descent on $(w-3)^2$ from $w=0$ with $\eta=0.1$.
#
# **Independent exercise on paper:** predict the first two values of $w$.
#
# **My prediction:** _write here._
#
# **Hint:** the gradient is $2(w-3)$.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_w=0.0
for demo_step in range(5):
    demo_grad=2*(demo_w-3)
    demo_w=demo_w-0.1*demo_grad
    print(f'Step {demo_step+1}: gradient {demo_grad:6.3f} → w = {demo_w:.4f}')

# %% [markdown]
# ### Why this result makes sense
#
# Each step multiplies the remaining distance by $1-2\eta=0.8$, so the gradient shrinks as the minimum gets closer and the steps become smaller automatically.
#
# **Check your understanding:** predict the value after step 5 from $3(1-0.8^5)$ and compare.

# %% [markdown]
# ## Guided practice 2 · check a derivative numerically
#
# Verify $\partial z/\partial x=3$ for $z=\text{mean}(3(x+1)^2)$ at $x=1$ with a finite difference.
#
# **Independent exercise on paper:** what change in $z$ do you expect if one of the four $x$ values increases by $10^{-6}$?
#
# **My prediction:** _write here._
#
# **Hint:** the change is approximately the derivative times the step.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
def demo_z(x):
    return np.mean(3*(x+1)**2)
demo_x=np.ones(4)
demo_eps=1e-6
demo_bumped=demo_x.copy(); demo_bumped[0]+=demo_eps
print('Finite-difference derivative:',round((demo_z(demo_bumped)-demo_z(demo_x))/demo_eps,4))

# %% [markdown]
# ### Why this result makes sense
#
# The mean divides by four, and $6(x+1)=12$ at $x=1$, so the derivative is 3. Finite differences are a useful check of hand-derived gradients but are far too slow for networks with thousands of parameters; backpropagation computes all derivatives in one backward pass.
#
# **Check your understanding:** repeat at $x=0$ and predict the result first.

# %% [markdown]
# ## Guided practice 3 · forgetting to clear gradients
#
# Simulate two training steps that both have gradient $-6$, once clearing the stored gradient between steps and once accumulating it.
#
# **Independent exercise on paper:** what gradient is used in the second step if it is not cleared?
#
# **My prediction:** _write here._
#
# **Hint:** accumulation adds the new gradient to the stored one.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_stored=0.0
for demo_step in range(2):
    demo_stored=demo_stored+(-6.0)
    print('Without zero_grad, step',demo_step+1,'uses gradient',demo_stored)
print('With zero_grad, every step uses gradient',-6.0)

# %% [markdown]
# ### Why this result makes sense
#
# Without clearing, the second step uses $-12$, so the update doubles and keeps growing. This is the most common silent bug in a PyTorch training loop.
#
# **Check your understanding:** how large is the accumulated gradient after ten steps?

# %% [markdown]
# ## Guided practice 4 · count the parameters of the EEG CNN
#
# Count parameters for the two convolutions used below: temporal $1\to8$ maps with a $1\times25$ kernel, then spatial $8\to16$ maps with a $64\times1$ kernel.
#
# **Independent exercise on paper:** apply $C_{out}(C_{in}k_hk_w+1)$ to each layer.
#
# **My prediction:** _write here._
#
# **Hint:** the spatial kernel spans all 64 channels of each of the 8 input maps.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_temporal=8*(1*1*25+1)
demo_spatial=16*(8*64*1+1)
print('Temporal convolution:',demo_temporal,'| spatial convolution:',demo_spatial)

# %% [markdown]
# ### Why this result makes sense
#
# Most parameters are in the spatial layer, because it connects every channel of every temporal map. Splitting time and space keeps the total far below a single $64\times25$ kernel per output map, which would need $16(64\cdot25+1)=25{,}616$ parameters. The network below drops the spatial bias (`bias=False`, 8,192 parameters) because the batch normalization that follows supplies its own offset.
#
# **Check your understanding:** recount with 4 temporal maps.

# %% [markdown]
# ## Practice 1 · A gradient-descent update
#
# Implement `gd_step(w, grad, lr)` that returns the updated parameter array. It must work for NumPy arrays of any shape.
#
# **Independent exercise:** complete the function below before reading its solution. The template deliberately returns None so that an unfinished attempt does not interrupt the rest of the lesson.

# %%
def gd_step(w, grad, lr):
    # TODO: move w against the gradient by the learning rate.
    return None

# %% [markdown]
# ### Hint
#
# One line: subtract the learning rate times the gradient. Do not modify `w` in place.

# %% [markdown]
# ### Worked solution
#
# Compare this implementation with your attempt. The next cell checks the reference answer on a concrete numerical case.

# %%
def gd_step(w, grad, lr):
    return w - lr * grad

# %%
answer=gd_step(np.array([0.0,1.0]),np.array([-6.0,2.0]),0.1)
if answer is not None:
    assert np.allclose(answer,[0.6,0.8]); print('Update check passed.')
else: print('Exercise pending: implement gd_step.')

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use PyTorch's autograd and then the real EEGBCI recordings. Read each stage's explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# Features are log mu (8–13 Hz) and log beta (13–30 Hz) power per channel from Welch spectra, standardized with run-4 statistics, as in Lesson 16. Raw epochs for the CNN are standardized per channel with run-4 statistics. Both networks are trained with Adam and cross-entropy on run 4; the epoch with the lowest run-8 loss is kept. A shrinkage LDA on the same features is the linear baseline. All three models are scored once on run 12 with balanced accuracy.

# %% [markdown]
# ## Autograd: gradients computed for you
# PyTorch records every operation on a tensor that has `requires_grad=True`, then applies the chain rule backwards when you call `.backward()`. The MNIST tutorial wrapped tensors in `Variable`; current PyTorch puts this directly on tensors.

# %% [markdown]
# ### Step 1.1 · record operations on a tensor
#
# **1.** Create a 2 × 2 tensor of ones that tracks gradients.
#
# **2.** Compute $y=3(x+1)^2$ and $z=\text{mean}(y)$; each result remembers the operation that made it in `grad_fn`.

# %%
x = torch.ones(2, 2, requires_grad=True)
y_demo = 3 * (x + 1) ** 2
z = y_demo.mean()
print('y:', y_demo)
print('y.grad_fn:', y_demo.grad_fn, '| z.grad_fn:', z.grad_fn)
print('x.grad before backward:', x.grad)

# %% [markdown]
# ### Step 1.2 · differentiate with backward
#
# **1.** `z.backward()` computes $\partial z/\partial x$ and stores it in `x.grad`.

# %%
z.backward()
print('x.grad after backward:', x.grad)

# %% [markdown]
# ### Inspect and interpret
#
# Every entry is 3, the value derived by hand and checked by finite differences. Autograd did not approximate it; it applied the chain rule exactly to the recorded operations.

# %% [markdown]
# ### Step 1.3 · one optimizer step on a controlled regression problem
#
# **1.** Simulate data with a known answer: $y=2x+3$ plus small noise. This is a controlled example, not EEG.
#
# **2.** Fit `nn.Linear(1, 1)` with mean squared error for 100 steps. Each step makes the three calls of every training loop.

# %%
torch.manual_seed(SEED)
reg_x = torch.randn(20, 1)
reg_y = 2 * reg_x + 3 + 0.1 * torch.randn(20, 1)
reg_model = nn.Linear(1, 1)
reg_opt = optim.SGD(reg_model.parameters(), lr=0.1)
for step in range(100):
    reg_opt.zero_grad()
    reg_loss = F.mse_loss(reg_model(reg_x), reg_y)
    reg_loss.backward()
    reg_opt.step()
print(f'Learned weight {reg_model.weight.item():.3f} (true 2), bias {reg_model.bias.item():.3f} (true 3), final loss {reg_loss.item():.4f}')

# %% [markdown]
# ### Inspect and interpret
#
# The model recovers the true slope and intercept up to the noise. The EEG problem below has no known answer, which is why it needs held-out runs.

# %% [markdown]
# ## Load EEG trials and define the run roles
# Five participants are downloaded once (about 40 MB) and cached. Each run is filtered and resampled separately.

# %% [markdown]
# ### Step 2.1 · load, filter and epoch each run
#
# **1.** Fix the subjects, runs and analysis window before looking at any result.
#
# **2.** For each run: load, standardize names, set montage and average reference, filter to 8–30 Hz, resample to 80 Hz and epoch the two imagery cues.

# %%
from mne.datasets import eegbci
SUBJECTS = [1, 2, 3, 4, 5]
RUNS = {'train': 4, 'validation': 8, 'test': 12}
parts, subject_ids, run_ids = [], [], []
for subject in SUBJECTS:
    for run in RUNS.values():
        path = eegbci.load_data(subject, [run], path=DATA_ROOT, update_path=False)[0]
        raw_run = mne.io.read_raw_edf(path, preload=True, verbose=False)
        eegbci.standardize(raw_run)
        raw_run.set_montage('standard_1005')
        raw_run.set_eeg_reference('average', projection=False)
        raw_run.filter(8, 30, fir_design='firwin')
        raw_run.resample(80)
        events, _ = mne.events_from_annotations(raw_run, event_id={'T1': 1, 'T2': 2})
        ep = mne.Epochs(raw_run, events, {'left': 1, 'right': 2}, tmin=0.5, tmax=3.5,
                        baseline=None, preload=True, picks='eeg', reject_by_annotation=True)
        parts.append(ep)
        subject_ids += [subject] * len(ep)
        run_ids += [run] * len(ep)
epochs = mne.concatenate_epochs(parts)
X = epochs.get_data(copy=True)
y = epochs.events[:, 2] - 1
subject_ids, run_ids = np.asarray(subject_ids), np.asarray(run_ids)
sfreq = epochs.info['sfreq']
print('Epochs (N, C, T):', X.shape, '| sampling rate:', sfreq, 'Hz')

# %% [markdown]
# ### Step 2.2 · make the three partitions explicit
#
# **1.** Build one boolean mask per role from the run IDs.
#
# **2.** Check that the masks do not overlap and that each contains both classes.

# %%
masks = {role: run_ids == run for role, run in RUNS.items()}
assert not (masks['train'] & masks['validation']).any() and not (masks['train'] & masks['test']).any()
print(pd.DataFrame({role: np.bincount(y[m], minlength=2) for role, m in masks.items()},
                   index=['left', 'right']))

# %% [markdown]
# ### Step 2.3 · band-power features scaled with training statistics
#
# **1.** Compute log mu and log beta power per channel, as in Lesson 16.
#
# **2.** Standardize every trial with the training-run mean and standard deviation.

# %%
freqs, psd = signal.welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
features = np.concatenate([np.log10(psd[:, :, (freqs >= 8) & (freqs <= 13)].mean(-1)),
                           np.log10(psd[:, :, (freqs > 13) & (freqs <= 30)].mean(-1))], axis=1)
f_mean, f_std = features[masks['train']].mean(0), features[masks['train']].std(0)
features_z = ((features - f_mean) / f_std).astype(np.float32)
print('Features (N, 2C):', features_z.shape)

# %% [markdown]
# ## Mini-batches with `TensorDataset` and `DataLoader`
# A `TensorDataset` pairs each feature row with its label. A `DataLoader` serves shuffled mini-batches from it.

# %% [markdown]
# ### Step 3.1 · wrap tensors and create loaders
#
# **1.** Create one dataset per partition from float32 features and int64 labels.
#
# **2.** Shuffle only the training loader, with a seeded generator so batch order is repeatable.

# %%
def make_dataset(inputs, mask):
    return TensorDataset(torch.tensor(inputs[mask]), torch.tensor(y[mask], dtype=torch.long))

feat_sets = {role: make_dataset(features_z, m) for role, m in masks.items()}
train_loader = DataLoader(feat_sets['train'], batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
batch_x, batch_y = next(iter(train_loader))
print('Batches per epoch:', len(train_loader), '| one batch:', tuple(batch_x.shape), tuple(batch_y.shape))

# %% [markdown]
# ### Inspect and interpret
#
# The last batch of an epoch can be smaller than 32. Labels are int64 class indices, as `CrossEntropyLoss` requires.

# %% [markdown]
# ## One training step, inspected
# Before looping, look at one step in detail: weights before, the loss, the gradient, and weights after.

# %% [markdown]
# ### Step 4.1 · define the Lesson 16 network, a loss and an optimizer
#
# **1.** Re-create the fully connected network: 128 → 64 → 32 → 2, with dropout to discourage memorization.
#
# **2.** `CrossEntropyLoss` takes logits directly; do not apply softmax first.
#
# **3.** Plain SGD makes the update rule visible: new weight = old weight − learning rate × gradient.

# %%
class Network(nn.Module):
    def __init__(self, n_features, n_hidden1=64, n_hidden2=32, n_classes=2, drop_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_classes)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

torch.manual_seed(SEED)
net = Network(features_z.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
print(net)

# %% [markdown]
# ### Step 4.2 · forward, loss, backward, step
#
# **1.** Keep a copy of one weight row before the step.
#
# **2.** Clear gradients, compute logits and loss, back-propagate, and update.

# %%
weights_before = net.fc1.weight[0, :5].detach().clone()
optimizer.zero_grad()
logits = net(batch_x)
loss = criterion(logits, batch_y)
loss.backward()
grad_row = net.fc1.weight.grad[0, :5].clone()
optimizer.step()
print('Logits shape:', tuple(logits.shape), '| loss:', round(loss.item(), 4), '(ln 2 =', round(np.log(2), 4), ')')
print('Weights before:', weights_before)
print('Gradient:      ', grad_row)
print('Weights after: ', net.fc1.weight[0, :5].detach())

# %% [markdown]
# ### Inspect and interpret
#
# The first loss is close to ln 2 ≈ 0.693, the cross-entropy of a 50/50 guess. Each weight moved by −0.01 × its gradient, a tiny change; training needs many such steps.

# %% [markdown]
# ## Practice 2 · Verify the SGD update rule
#
# **Independent exercise:** Using `weights_before`, `grad_row` and the learning rate, show that the new weights equal $w-\eta g$.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# This is `gd_step` from Practice 1, applied to tensors.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
expected = weights_before - 0.01 * grad_row
assert torch.allclose(net.fc1.weight[0, :5].detach(), expected)
print('SGD update verified:', expected)

# %% [markdown]
# ### Interpret and check
#
# `optimizer.step()` with plain SGD is exactly the update derived by hand. Adam, used below, rescales each parameter's step using running averages of its past gradients, but it still reads the gradients that `backward` stored.

# %% [markdown]
# ## Training for real
# One reusable function trains any model on run 4, measures run 8 after every epoch, and returns the weights from the epoch with the lowest validation loss.

# %% [markdown]
# ### Step 5.1 · a training function with early stopping
#
# **1.** Each epoch loops over shuffled training batches: zero_grad, forward, loss, backward, step.
#
# **2.** After each epoch the model is switched to evaluation mode (dropout off) and validation loss and balanced accuracy are computed without gradients.
#
# **3.** A deep copy of the weights is kept whenever validation loss improves.

# %%
def evaluate(model, dataset):
    """Return loss, balanced accuracy and predictions for a whole partition."""
    inputs, labels = dataset.tensors
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))
        loss = criterion(logits, labels.to(device)).item()
        pred = logits.argmax(dim=1).cpu().numpy()
    return loss, balanced_accuracy_score(labels.numpy(), pred), pred

def train_model(model, train_set, val_set, epochs=60, lr=1e-3, weight_decay=1e-3, batch_size=32):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    history, best_loss, best_state = [], float('inf'), None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            opt.step()
            running += loss.item() * len(labels)
        val_loss, val_bacc, _ = evaluate(model, val_set)
        history.append({'epoch': epoch + 1, 'train_loss': running / len(train_set),
                        'val_loss': val_loss, 'val_bacc': val_bacc})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    best = history.loc[history.val_loss.idxmin()]
    print(f"Best epoch {int(best.epoch)}: validation loss {best.val_loss:.3f}, balanced accuracy {best.val_bacc:.3f}")
    return history

# %% [markdown]
# ### Step 5.2 · train the band-power network
#
# **1.** Start from fresh weights with a fixed seed.
#
# **2.** Train for 60 epochs and keep the best validation checkpoint.

# %%
torch.manual_seed(SEED)
mlp = Network(features_z.shape[1])
start = time.time()
mlp_history = train_model(mlp, feat_sets['train'], feat_sets['validation'])
print(f'Training time: {time.time() - start:.1f} s')

# %% [markdown]
# ### Step 5.3 · read the learning curves
#
# **1.** Plot training and validation loss, and validation balanced accuracy, per epoch.
#
# **2.** Mark the epoch whose weights were kept.

# %%
def plot_history(history, title):
    best_epoch = history.loc[history.val_loss.idxmin(), 'epoch']
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    ax_loss.plot(history.epoch, history.train_loss, label='Training (run 4)')
    ax_loss.plot(history.epoch, history.val_loss, label='Validation (run 8)')
    ax_acc.plot(history.epoch, history.val_bacc, color='#b87714')
    for ax in (ax_loss, ax_acc):
        ax.axvline(best_epoch, color='0.5', ls='--', lw=1)
    ax_acc.axhline(0.5, color='0.7', lw=1)
    ax_loss.set(xlabel='Epoch', ylabel='Cross-entropy', title=f'{title}: loss')
    ax_acc.set(xlabel='Epoch', ylabel='Balanced accuracy', ylim=(0.3, 1), title='Validation accuracy (grey: chance)')
    ax_loss.legend()
    plt.show()

plot_history(mlp_history, 'Band-power network')

# %% [markdown]
# ### Inspect and interpret
#
# Training loss falls steadily. Validation loss falls less, levels off or rises: the gap is the part of learning that did not transfer from run 4 to run 8. The dashed line marks the kept checkpoint, which need not be the last epoch.

# %% [markdown]
# ## Practice 3 · Compare with a linear baseline on validation
#
# **Independent exercise:** Fit a shrinkage LDA on the training-run features and report its validation balanced accuracy next to the network's best value. Do not use run 12.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` handles 128 features from about 150 trials.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda.fit(features_z[masks['train']], y[masks['train']])
lda_val = balanced_accuracy_score(y[masks['validation']], lda.predict(features_z[masks['validation']]))
print(f"Validation balanced accuracy: LDA {lda_val:.3f} | network {mlp_history.val_bacc[mlp_history.val_loss.idxmin()]:.3f}")

# %% [markdown]
# ### Interpret and check
#
# With about 150 training trials, a regularized linear model is often as good as, or better than, a network with 10,000 parameters on the same features. If the network does not win here, that is a valid result: it shows the features, not the classifier, limit performance.

# %% [markdown]
# ## A convolutional network on raw epochs
# Instead of hand-made band power, the CNN receives the filtered EEG itself, shaped like a one-channel image: (trials, 1, channels, time).

# %% [markdown]
# ### Step 6.1 · normalize per channel and add the image axis
#
# **1.** Compute each channel's mean and standard deviation over training trials and time.
#
# **2.** Apply them to all trials, convert to float32 and insert a singleton "image channel" axis.

# %%
ch_mean = X[masks['train']].mean(axis=(0, 2), keepdims=True)
ch_std = X[masks['train']].std(axis=(0, 2), keepdims=True)
X_z = ((X - ch_mean) / ch_std).astype(np.float32)[:, np.newaxis]
raw_sets = {role: make_dataset(X_z, m) for role, m in masks.items()}
print('CNN input (N, 1, C, T):', X_z.shape)

# %% [markdown]
# ### Step 6.2 · define the convolutional network
#
# **1.** A temporal convolution (1 × 25 kernel, 8 maps) learns band-pass-like filters; padding keeps the time length.
#
# **2.** A spatial convolution (64 × 1 kernel, 16 maps) combines all channels, followed by batch normalization and ELU.
#
# **3.** Average pooling over 8 samples (0.1 s), dropout, flattening and a linear layer give two logits.
#
# **4.** The hyperparameters are stored on the model so a checkpoint can rebuild it.

# %%
class EEGConvNet(nn.Module):
    def __init__(self, n_channels, n_times, n_temporal=8, n_spatial=16, kernel=25, pool=8, drop_prob=0.5):
        super().__init__()
        self.hparams = dict(n_channels=n_channels, n_times=n_times, n_temporal=n_temporal,
                            n_spatial=n_spatial, kernel=kernel, pool=pool, drop_prob=drop_prob)
        self.temporal = nn.Conv2d(1, n_temporal, (1, kernel), padding=(0, kernel // 2))
        self.spatial = nn.Conv2d(n_temporal, n_spatial, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_spatial)
        self.pool = nn.AvgPool2d((1, pool))
        self.drop = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_spatial * (n_times // pool), 2)

    def forward(self, x):
        x = self.temporal(x)                        # (N, 8, C, T): filtered copies of each channel
        x = F.elu(self.bn(self.spatial(x)))         # (N, 16, 1, T): spatial combinations
        x = self.drop(self.pool(x))                 # (N, 16, 1, T // 8): temporal summary
        return self.fc(x.flatten(start_dim=1))      # (N, 2): logits

torch.manual_seed(SEED)
cnn = EEGConvNet(n_channels=X.shape[1], n_times=X.shape[2])
print(cnn)
print('Trainable parameters:', sum(p.numel() for p in cnn.parameters() if p.requires_grad))

# %% [markdown]
# ## Practice 4 · Trace shapes through the CNN
#
# **Independent exercise:** Pass two trials through each layer of `cnn` in order and print the shape after each. Explain why the spatial convolution reduces the channel axis to 1.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Call the layers one at a time in the order used in `forward`, in evaluation mode so dropout and batch normalization behave deterministically.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
cnn.eval()
probe = torch.tensor(X_z[:2])
with torch.no_grad():
    print('Input           ', tuple(probe.shape))
    probe = cnn.temporal(probe); print('Temporal conv   ', tuple(probe.shape))
    probe = F.elu(cnn.bn(cnn.spatial(probe))); print('Spatial conv    ', tuple(probe.shape))
    probe = cnn.pool(probe); print('Average pooling ', tuple(probe.shape))
    probe = cnn.fc(probe.flatten(start_dim=1)); print('Logits          ', tuple(probe.shape))

# %% [markdown]
# ### Interpret and check
#
# The spatial kernel is as tall as the montage (64 channels) and is not padded, so it fits exactly once along that axis. The temporal padding of 12 samples on each side keeps 241 samples; pooling by 8 leaves 30.

# %% [markdown]
# ### Step 6.3 · train the CNN with the same function
#
# **1.** The same training function, loss and validation rule apply; only the model and the input tensors change.

# %%
torch.manual_seed(SEED)
cnn = EEGConvNet(n_channels=X.shape[1], n_times=X.shape[2])
start = time.time()
cnn_history = train_model(cnn, raw_sets['train'], raw_sets['validation'], epochs=60)
print(f'Training time: {time.time() - start:.1f} s on {device}')
plot_history(cnn_history, 'EEG CNN')

# %% [markdown]
# ### Inspect and interpret
#
# The CNN has to learn its own filters from roughly 150 trials, so its curves are often noisier than the band-power network's. Compare the best validation values, not the final epoch.

# %% [markdown]
# ## Test once
# All choices are now fixed: features, architectures, epochs and checkpoints. Run 12 is scored once for each model.

# %% [markdown]
# ### Step 7.1 · score the three selected models on run 12
#
# **1.** Evaluate the LDA, the band-power network and the CNN on the test partition.
#
# **2.** Report the validation score next to the test score for each model.

# %%
_, mlp_test, _ = evaluate(mlp, feat_sets['test'])
_, cnn_test, cnn_pred = evaluate(cnn, raw_sets['test'])
lda_test = balanced_accuracy_score(y[masks['test']], lda.predict(features_z[masks['test']]))
results = pd.DataFrame({
    'validation (run 8)': [lda_val, mlp_history.val_bacc[mlp_history.val_loss.idxmin()], cnn_history.val_bacc[cnn_history.val_loss.idxmin()]],
    'test (run 12)': [lda_test, mlp_test, cnn_test]},
    index=['Shrinkage LDA, band power', 'Network, band power', 'CNN, raw epochs']).round(3)
print(results)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ConfusionMatrixDisplay.from_predictions(y[masks['test']], cnn_pred, display_labels=['left', 'right'],
                                        cmap='Blues', ax=ax, colorbar=False)
ax.set_title('CNN on the test run'); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Test scores are usually lower than validation scores, because validation helped choose each checkpoint. With about 150 test trials, a difference of a few percentage points between models is within chance variation; Lesson 09 shows how to quantify that.

# %% [markdown]
# ## Practice 5 · How far from chance is the test score?
#
# **Independent exercise:** With $n$ test trials and two balanced classes, accuracy under pure guessing has standard deviation $\sqrt{0.25/n}$. Compute the upper end of a 95% chance band (0.5 + 1.96 SD) and say which models exceed it.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# $n$ is the number of trials in run 12 across all five participants.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
n_test = int(masks['test'].sum())
chance_upper = 0.5 + 1.96 * np.sqrt(0.25 / n_test)
print(f'{n_test} test trials → 95% chance band up to {chance_upper:.3f}')
print(results['test (run 12)'] > chance_upper)

# %% [markdown]
# ### Interpret and check
#
# This normal approximation ignores that trials from the same participant are correlated, so the real chance band is somewhat wider. A permutation test (Lesson 09) avoids the approximation.

# %% [markdown]
# ## Saving and loading a model
# A trained EEG model is only reusable with everything that defines its input: architecture settings, weights, channel order, sampling rate, filter band and normalization statistics.

# %% [markdown]
# ### Step 8.1 · save a checkpoint dictionary
#
# **1.** Collect the hyperparameters, `state_dict`, normalization statistics and preprocessing description in one dictionary.
#
# **2.** Save it with `torch.save`.

# %%
checkpoint = {
    'hparams': cnn.hparams,
    'state_dict': cnn.state_dict(),
    'ch_names': epochs.ch_names,
    'sfreq': sfreq,
    'band_hz': [8, 30],
    'window_s': [0.5, 3.5],
    'ch_mean': torch.tensor(ch_mean.squeeze()),
    'ch_std': torch.tensor(ch_std.squeeze()),
    'classes': ['left', 'right'],
}
checkpoint_path = Path('model_checkpoints') / 'eeg_cnn_checkpoint.pt'
checkpoint_path.parent.mkdir(exist_ok=True)
torch.save(checkpoint, checkpoint_path)
print('Saved', checkpoint_path, f'({checkpoint_path.stat().st_size / 1024:.0f} kB)')

# %% [markdown]
# ### Step 8.2 · rebuild the model and check it gives the same predictions
#
# **1.** Load the dictionary (`weights_only=True` refuses arbitrary Python objects, a safety default in current PyTorch).
#
# **2.** Rebuild the network from the stored hyperparameters, load the weights and compare test predictions.

# %%
loaded = torch.load(checkpoint_path, weights_only=True)
restored = EEGConvNet(**loaded['hparams'])
restored.load_state_dict(loaded['state_dict'])
_, restored_test, restored_pred = evaluate(restored.to(device), raw_sets['test'])
assert np.array_equal(restored_pred, cnn_pred)
print('Restored model reproduces the test predictions; balanced accuracy', round(restored_test, 3))

# %% [markdown]
# ### Inspect and interpret
#
# The weights alone would not be enough: applied to EEG with a different channel order, sampling rate or scaling, the same weights compute something else without raising an error.

# %% [markdown]
# ## Recorded-signal inspection with MNE-Python
#
# The spatial convolution has one weight per channel for each temporal map. MNE can draw these weights on the scalp. They are filter weights, not activation patterns: a large weight can cancel noise rather than mark the source of the signal (Haufe et al., 2014).

# %%
spatial_w = cnn.spatial.weight.detach().cpu().numpy()[:, :, :, 0]   # (16 maps, 8 temporal, 64 channels)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
for k, ax in enumerate(axes):
    mne.viz.plot_topomap(spatial_w[k, 0], epochs.info, axes=ax, show=False, cmap='RdBu_r')
    ax.set_title(f'Spatial map {k + 1}\n(temporal filter 1)')
plt.show()

# %% [markdown]
# ### Figure interpretation and independent exercise
#
# Each map shows how one spatial filter weights the 64 channels after temporal filter 1. Weights near C3 and C4 with opposite signs would contrast the two hemispheres, as in Lesson 08's CSP filters.
#
# **Exercise.** State one reason these weights could look noisy even if the model decodes well, and one claim the maps cannot support.
#
# **Reference interpretation.** With 150 training trials and many parameters, weights on uninformative channels are poorly constrained, and filters can cancel noise rather than point to sources. The maps cannot localize the neural generators of motor imagery; that would require patterns (forward models) and source analysis, which this course does not cover.

# %% [markdown]
# ## Practice 6 · explain the complete method
#
# Independently explain how the CNN was trained, selected and evaluated. Include one way the validation run could leak into the final result and one claim the test score does not establish.
#
# **My explanation:** _write here._

# %% [markdown]
# ### Worked answer · compare your reasoning
#
# Run 4 provided normalization statistics and weight updates through mini-batch Adam on cross-entropy; after each epoch, run 8's loss decided whether to keep the weights; run 12 was scored once with the kept weights. Validation leaks if architectures or learning rates are tuned repeatedly on run 8 and the best run-8 score is reported as the result. The test score does not establish performance for new participants, online use with feedback, or that the model relies on sensorimotor rhythms rather than artifacts.

# %% [markdown]
# ## If your result is different
#
# If training loss does not fall, check that `zero_grad` is called and the learning rate is not too small. If it becomes `nan`, lower the learning rate. If validation accuracy stays at 0.5, check that labels are aligned with trials and that the model predicts both classes. Scores vary with seeds and library versions; record them with your results. If a download fails, retry when PhysioNet is reachable; do not replace real data with simulated values.

# %% [markdown]
# ## Can you now do this independently?
#
# - Derive a gradient-descent step and verify a gradient with autograd.
# - Write a training loop with `zero_grad`, `backward` and `step`, and explain what goes wrong without `zero_grad`.
# - Use a validation run for early stopping and keep the test run for one final score.
# - Build a temporal–spatial CNN for EEG and trace its tensor shapes.
# - Save and reload a model with the preprocessing information it needs.
#
# If one item is unclear, return to the associated practice before Lesson 18.

# %% [markdown]
# ## Next steps and sources
# Lesson 18 uses the same tools for unsupervised learning: a convolutional autoencoder that compresses and reconstructs EEG epochs.
#
# [PyTorch optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) · [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) · [Datasets and DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

# %% [markdown]
# ## References and further reading
#
# 1. [Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization](https://doi.org/10.1002/hbm.23730).
# 2. [Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://doi.org/10.1088/1741-2552/aace8c).
# 3. [Haufe et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging](https://doi.org/10.1016/j.neuroimage.2013.10.067).
# 4. [Kingma and Ba (2015). Adam: a method for stochastic optimization](https://arxiv.org/abs/1412.6980).
# 5. [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
#
# Numerical outcomes in this notebook refer only to the selected participants, runs and processing; they are not population performance estimates.
