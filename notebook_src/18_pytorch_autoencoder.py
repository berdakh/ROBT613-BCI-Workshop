# %% [markdown]
# # 18 · Deep learning with PyTorch III: convolutional autoencoders for EEG
#
# **ROBT613 · Brain–Computer Interfaces** | PyTorch track, part 3 of 3 | Academic tutorial and independent exercises
#
# ## Goal
# Build a convolutional autoencoder that compresses 64-channel EEG epochs 16-fold and reconstructs them, compare it with a linear baseline of the same size, and test whether the compressed code still carries motor-imagery information.
#
# **Data:** PhysioNet EEGBCI, subjects 1–5, imagery runs 4 (train), 8 (validation), 12 (test)
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required. This lesson adapts the classic "Autoencoders in PyTorch" MNIST tutorial and its exercise version to EEG: 2-D image convolutions become 1-D convolutions along time, with EEG channels as input maps.

# %% [markdown]
# ## Paradigm background and experimental design
#
# ### Why compress EEG?
#
# An autoencoder learns a short code from which its input can be reconstructed. For EEG this is used to reduce dimensionality before decoding, to learn features without labels, to detect unusual segments by their reconstruction error, and, with deliberately corrupted inputs, for denoising (Lesson 13). The training objective is reconstruction, not classification. Whether the code is useful for a BCI has to be tested separately, which this lesson does at the end.
#
# ### Motor imagery and sensorimotor rhythms
#
# Imagining a left or right fist movement reduces mu (8–13 Hz) and beta (13–30 Hz) power over the opposite sensorimotor cortex. These changes are small compared with the total EEG variance. An autoencoder that minimizes squared error spends its capacity on the largest-variance parts of the signal, which need not be the parts that distinguish the two tasks.
#
# ### Acquisition provenance and instructional protocol
#
# PhysioNet EEGBCI: 64 scalp channels, 160 Hz, BCI2000; runs 4, 8 and 12 contain cued left/right fist imagery. Each run is filtered to 4–35 Hz and resampled to 80 Hz; epochs cover 0.5–3.5 s after the cue and are cropped to 240 samples so that two halvings of the time axis give whole numbers.
#
# **Acquisition reference:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
#
# | Run | Autoencoder role | Classification probe role |
# |---|---|---|
# | 4 | Normalization and weights | Classifier training |
# | 8 | Checkpoint selection | Comparison of feature sets |
# | 12 | Final reconstruction error | One final score per feature set |
#
# ### Measurement model and interpretation
#
# The reconstruction target is the recorded EEG, which already contains eye, muscle and line-noise contributions. A low reconstruction error means the code preserves whatever dominates the recording, artifacts included. It does not mean the network has found the brain sources of the signal.
#
# ### Mathematical definitions for this lesson
#
# An encoder $E_\theta$ maps an epoch $X\in\mathbb R^{C\times T}$ to a code $Z=E_\theta(X)\in\mathbb R^{K\times T'}$ and a decoder $D_\phi$ returns $\hat X=D_\phi(Z)$. Training minimizes $\text{MSE}=\frac{1}{NCT}\sum_{i,c,t}(\hat X_{ict}-X_{ict})^2$. A 1-D convolution with kernel $k$, stride $s$ and padding $p$ maps length $L$ to $\lfloor(L+2p-k)/s\rfloor+1$. A transposed convolution maps $L$ to $(L-1)s-2p+k+a$, where the output padding $a$ resolves the rounding.
#
# **Methodological reading:** [Hinton and Salakhutdinov (2006). Reducing the dimensionality of data with neural networks](https://doi.org/10.1126/science.1127647); [Roy et al. (2019). Deep learning-based electroencephalography analysis: a systematic review](https://doi.org/10.1088/1741-2552/ab260c).

# %% [markdown]
# ## How to study this notebook
#
# This is both the classroom lesson and the independent-study workbook. Everything needed for the exercises—questions, hints, executable solutions, checks and explanations—is here. Work from top to bottom in a fresh runtime.
#
# 1. Read the question and calculate a small example on paper.
# 2. Write your prediction before running the next code cell.
# 3. Complete the task in its workspace. In Practice 2 you write the whole network yourself before seeing the solution.
# 4. Continue to the worked solution and compare the reasoning as well as the numerical result.
# 5. Change one parameter and explain what the result means.
#
# **For a live class:** pause at each "Independent exercise" heading. The solution follows in the same notebook.
#
# **Prerequisites:** Lessons 16 and 17 (tensors, `nn.Module`, training loops, validation).

# %% [markdown]
# ## Analytical objectives
#
# ### Learning outcomes
#
# - Compute output lengths of strided and transposed 1-D convolutions and design an encoder–decoder that restores the input shape.
# - Write an autoencoder with separate `encode` and `decode` methods and train it with mean squared error.
# - Judge reconstruction error against a zero predictor and a linear baseline that keeps the same number of values.
# - Test whether a compressed code carries task information, and explain why low reconstruction error does not guarantee it.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
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
map_steps=['Epoch\n64 ch × 240 samples', 'Encoder\nstrided Conv1d ×2', 'Code\n16 maps × 60 steps', 'Decoder\nConvTranspose1d ×2', 'Reconstruction\nMSE vs input']
fig,map_ax=plt.subplots(figsize=(12,3.1),constrained_layout=True)
map_ax.set(xlim=(-.1,12),ylim=(-.3,2.4));map_ax.axis('off')
for map_i,map_label in enumerate(map_steps):
    map_x=map_i*2.4
    map_ax.add_patch(FancyBboxPatch((map_x,.45),2.05,1.1,
        boxstyle='round,pad=0.08',facecolor='#f7f2ea' if map_i==2 else '#edf3f7',edgecolor='#35688a',linewidth=1.5))
    map_ax.text(map_x+1.025,1.02,map_label,ha='center',va='center',fontsize=10)
    map_ax.text(map_x+1.025,1.83,str(map_i+1),ha='center',weight='bold',color='#35688a')
    if map_i<4:map_ax.annotate('',xy=(map_x+2.3,1),xytext=(map_x+2.13,1),arrowprops=dict(arrowstyle='->',lw=1.5))
map_ax.text(5.9,-.08,'15,360 values → 960 values → 15,360 values. No labels are used to train this network.',ha='center',fontsize=10)
map_ax.set_title('Lesson 18 · compress and reconstruct an EEG epoch',fontsize=14,pad=12)
plt.show()

# %% [markdown]
# **Read the map:** the shaded code is the only path from input to output, so everything the decoder reproduces must pass through 960 numbers. Labels are used later, separately, to ask whether those 960 numbers distinguish left from right imagery. The map as text: epoch (64 × 240) → encoder → code (16 × 60) → decoder → reconstruction compared with the input.

# %% [markdown]
# ## Symbols and a calculation by hand
#
# $L$: sequence length; $k$: kernel size; $s$: stride; $p$: padding; $a$: output padding; $K$: number of code maps.
#
# ### Derive the operation before calling the library
#
# With $k=7$, $s=2$, $p=3$, a length-240 input gives $\lfloor(240+6-7)/2\rfloor+1=\lfloor119.5\rfloor+1=120$. A second such layer gives 60. Going back, a transposed convolution with the same settings gives $(60-1)\cdot2-6+7=119$ from 60, one short of 120; output padding $a=1$ (or `output_size=120`) makes it 120.
#
# The input has $64\times240=15{,}360$ values. The code has $16\times60=960$, a compression factor of 16. At 60 steps over 3 s, each code map is sampled at 20 Hz.

# %% [markdown]
# ### Your paper calculation
#
# Repeat the encoder lengths for an input of 241 samples (the uncropped epoch). Where does a non-integer appear, and what length does the decoder then produce without `output_size`?
#
# **My calculation:** _write your intermediate steps here before continuing._

# %% [markdown]
# ### From image autoencoders to EEG autoencoders
#
# The MNIST autoencoder treats a digit as a 1 × 28 × 28 image and uses `Conv2d`. An EEG epoch is better treated as 64 input maps (the channels) of 240 time samples each, with `Conv1d` sliding along time. The first layer's kernels therefore span all 64 channels at once and learn spatial and temporal structure together.
#
# ### Downsampling and upsampling
#
# The encoder halves the time axis twice with stride-2 convolutions, instead of max-pooling, as in the original tutorial. The decoder doubles it twice with transposed convolutions (`ConvTranspose1d`); passing `output_size` removes the rounding ambiguity. Nearest-neighbour upsampling followed by an ordinary convolution is an alternative that avoids the "checkerboard" artifacts transposed convolutions can produce (Odena et al., 2016).
#
# ### Dropout, train mode and evaluation mode
#
# Dropout randomly zeroes activations during training to reduce overfitting. `model.train()` turns it on and `model.eval()` turns it off. Forgetting `eval()` when measuring validation error adds randomness to the result; Practice 5 shows how much.
#
# ### What reconstruction error does and does not measure
#
# On channel-standardized data, predicting zeros everywhere gives an MSE of about 1, the average variance. Any useful model must do better. A fairer comparison is a linear method that keeps the same number of values per epoch. Here, four spatial principal components at every time sample keep exactly 960 values. If the autoencoder cannot beat that, its nonlinearity is not earning its parameters.

# %% [markdown]
# ## Visual intuition · Stride halves the time axis
#
# **Independent exercise:** Before running, predict how many output samples a stride-2 kernel produces from 16 input samples with padding that keeps the edges.

# %%
vis_t=np.arange(16)
vis_x=np.sin(2*np.pi*vis_t/8)
vis_kernel=np.array([.25,.5,.25])
vis_full=np.convolve(np.pad(vis_x,1),vis_kernel,mode='valid')
vis_stride=vis_full[::2]
fig,ax=plt.subplots(figsize=(10,3.4))
ax.plot(vis_t,vis_x,'o-',color='0.6',label='Input, 16 samples')
ax.plot(vis_t[::2],vis_stride,'s',color='#b87714',ms=9,label=f'Stride 2 output, {len(vis_stride)} samples')
ax.set(xlabel='Sample',ylabel='Value',title='A stride-2 convolution keeps every second output')
ax.legend()
plt.show()

# %% [markdown]
# ### Worked interpretation
#
# The smoothing kernel is evaluated at every second position, so 16 samples become 8. The shape of the slow wave survives; a wave faster than half the new sampling rate would not. The encoder therefore keeps slower structure more easily than fast detail.

# %% [markdown]
# ## Guided practice 1 · encoder lengths
#
# Apply the output-length formula twice to 240 samples with $k=7$, $s=2$, $p=3$.
#
# **Independent exercise on paper:** predict the two lengths.
#
# **My prediction:** _write here._
#
# **Hint:** use integer (floor) division.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_L=240
for demo_layer in range(2):
    demo_L=(demo_L+2*3-7)//2+1
    print('After layer',demo_layer+1,':',demo_L,'samples')

# %% [markdown]
# ### Why this result makes sense
#
# With this padding a stride-2 layer halves an even length exactly. Lengths 240 → 120 → 60 stay integers because 240 is divisible by 4.
#
# **Check your understanding:** start from 241 and compare.

# %% [markdown]
# ## Guided practice 2 · decoder lengths
#
# Apply the transposed-convolution formula from 60 back to 240.
#
# **Independent exercise on paper:** with $a=0$, what length follows 60?
#
# **My prediction:** _write here._
#
# **Hint:** $(L-1)s-2p+k+a$.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_L=60
for demo_target in [120,240]:
    demo_plain=(demo_L-1)*2-2*3+7
    print(f'{demo_L} → {demo_plain} without output padding; output_padding = {demo_target-demo_plain} gives {demo_target}')
    demo_L=demo_target

# %% [markdown]
# ### Why this result makes sense
#
# Many input lengths map to the same strided output (both 239 and 240 give 120), so the transposed layer cannot know which to restore. `output_size` tells it.
#
# **Check your understanding:** which input lengths give 60 after one stride-2 layer?

# %% [markdown]
# ## Guided practice 3 · compression factor
#
# Compare the number of values in an epoch with the number in the code, for 8, 16 and 32 code maps.
#
# **Independent exercise on paper:** which code size gives 16-fold compression?
#
# **My prediction:** _write here._
#
# **Hint:** the code length is 60 for all three.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
for demo_maps in [8,16,32]:
    print(f'{demo_maps:2d} code maps: {demo_maps*60:4d} values, compression ×{64*240/(demo_maps*60):.0f}')

# %% [markdown]
# ### Why this result makes sense
#
# Fewer code maps compress more and usually reconstruct worse. The right size depends on what the code is for; a reconstruction-optimal size need not be classification-optimal.
#
# **Check your understanding:** how many values would four spatial principal components keep at 80 Hz for 3 s?

# %% [markdown]
# ## Guided practice 4 · the zero predictor
#
# Show that predicting all zeros gives an MSE equal to the mean variance for standardized data.
#
# **Independent exercise on paper:** what is the MSE of zeros for data with mean 0 and standard deviation 1?
#
# **My prediction:** _write here._
#
# **Hint:** the squared error of zero is the squared value itself.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_X=rng.standard_normal((50,4,240))
print('MSE of predicting zeros:',round(np.mean(demo_X**2),3))

# %% [markdown]
# ### Why this result makes sense
#
# For standardized data the mean of the squares is the variance, 1. A reconstruction MSE of 0.5 therefore means the model explains half of the variance.
#
# **Check your understanding:** what MSE would predicting the per-channel mean give on the training data?

# %% [markdown]
# ## Practice 1 · A convolution length calculator
#
# Implement `conv_out_len(L, k, s, p)` that returns the output length of a 1-D convolution.
#
# **Independent exercise:** complete the function below before reading its solution. The template deliberately returns None so that an unfinished attempt does not interrupt the rest of the lesson.

# %%
def conv_out_len(L, k, s, p):
    # TODO: floor((L + 2p - k) / s) + 1
    return None

# %% [markdown]
# ### Hint
#
# Use integer division `//`, as in Guided practice 1.

# %% [markdown]
# ### Worked solution
#
# Compare this implementation with your attempt. The next cell checks the reference answer on a concrete numerical case.

# %%
def conv_out_len(L, k, s, p):
    return (L + 2 * p - k) // s + 1

# %%
answer=conv_out_len(240,7,2,3)
if answer is not None:
    assert answer==120 and conv_out_len(241,7,2,3)==121 and conv_out_len(28,3,2,1)==14; print('Length check passed.')
else: print('Exercise pending: implement conv_out_len.')

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real EEGBCI recordings. Read each stage's explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# Each channel is standardized with its run-4 mean and standard deviation. The autoencoder is trained on run-4 epochs with Adam and MSE; after each epoch, run-8 MSE decides whether to keep the weights. The baselines are the zero predictor and a four-component spatial PCA fitted on run 4, which keeps the same number of values. The probe classifies left versus right with shrinkage LDA on the log-variance of each code map, and compares it with the same classifier on PCA components and on all channels.

# %% [markdown]
# ## Load EEG trials
# Five participants are downloaded once (about 40 MB) and cached. Each run is filtered and resampled separately.

# %% [markdown]
# ### Step 1.1 · load, filter, resample and epoch each run
#
# **1.** Fix the subjects, runs and analysis window before looking at any result.
#
# **2.** For each run: load, standardize names, set montage and average reference, filter to 4–35 Hz, resample to 80 Hz and epoch the imagery cues.

# %%
from mne.datasets import eegbci
SUBJECTS = [1, 2, 3, 4, 5]
RUNS = {'train': 4, 'validation': 8, 'test': 12}
parts, run_ids = [], []
for subject in SUBJECTS:
    for run in RUNS.values():
        path = eegbci.load_data(subject, [run], path=DATA_ROOT, update_path=False)[0]
        raw_run = mne.io.read_raw_edf(path, preload=True, verbose=False)
        eegbci.standardize(raw_run)
        raw_run.set_montage('standard_1005')
        raw_run.set_eeg_reference('average', projection=False)
        raw_run.filter(4, 35, fir_design='firwin')
        raw_run.resample(80)
        events, _ = mne.events_from_annotations(raw_run, event_id={'T1': 1, 'T2': 2})
        ep = mne.Epochs(raw_run, events, {'left': 1, 'right': 2}, tmin=0.5, tmax=3.5,
                        baseline=None, preload=True, picks='eeg', reject_by_annotation=True)
        parts.append(ep)
        run_ids += [run] * len(ep)
epochs = mne.concatenate_epochs(parts)
run_ids = np.asarray(run_ids)
y = epochs.events[:, 2] - 1

# %% [markdown]
# ### Step 1.2 · crop to 240 samples and standardize per channel
#
# **1.** Keep the first 240 samples (3.0 s) so the encoder lengths are whole numbers.
#
# **2.** Standardize each channel with run-4 statistics and build one tensor dataset per run.

# %%
X = epochs.get_data(copy=True)[:, :, :240]
masks = {role: run_ids == run for role, run in RUNS.items()}
ch_mean = X[masks['train']].mean(axis=(0, 2), keepdims=True)
ch_std = X[masks['train']].std(axis=(0, 2), keepdims=True)
X_z = ((X - ch_mean) / ch_std).astype(np.float32)
datasets = {role: TensorDataset(torch.tensor(X_z[m])) for role, m in masks.items()}
n_channels, n_times = X_z.shape[1:]
print('Epochs (N, C, T):', X_z.shape, '| trials per role:', {r: int(m.sum()) for r, m in masks.items()})

# %% [markdown]
# ### Inspect and interpret
#
# The datasets contain EEG only; labels `y` are kept aside for the probe at the end. Training an autoencoder never needs them.

# %% [markdown]
# ## Practice 2 · Build the autoencoder yourself
#
# **Independent exercise:** This is the exercise version of the MNIST tutorial. Write a class `Autoencoder(nn.Module)` with:
#
# - an encoder of two `Conv1d` layers with kernel 7, stride 2 and padding 3, mapping 64 → 32 → 16 maps, with ELU after each and dropout after the first;
# - a decoder of two `ConvTranspose1d` layers mapping 16 → 32 → 64 maps, called with `output_size` to restore 120 and 240 samples, with ELU and dropout after the first and no activation after the last;
# - methods `encode(x)`, `decode(z)` and `forward(x)`.
#
# Then run `test_network` (defined in the worked solution) on your class. The skeleton below is commented out so that Run all still works.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Uncomment and complete the skeleton.
# class Autoencoder(nn.Module):
#     def __init__(self, n_channels=64, n_code=16, drop_prob=0.1):
#         super().__init__()
#         # TODO: encoder layers
#         # TODO: decoder layers
#         # TODO: dropout
#
#     def encode(self, x):
#         # TODO: forward pass through the encoder
#         return x
#
#     def decode(self, z):
#         # TODO: forward pass through the decoder; restore 120, then 240 samples
#         return z
#
#     def forward(self, x):
#         return self.decode(self.encode(x))

# %% [markdown]
# ### Hint
#
# Store the transposed layers as attributes and call them as `self.up1(z, output_size=(120,))`. Use Practice 1 and Guided practice 2 to check every length before running.

# %% [markdown]
# ### Worked solution
#
# Run the following definitions after attempting your own version. `test_network` replaces the MNIST tutorial's `helper.test_network`: it passes one batch forward and backward and checks the output shape.

# %%
class Autoencoder(nn.Module):
    def __init__(self, n_channels=64, n_code=16, drop_prob=0.1):
        super().__init__()
        # Encoder: each stride-2 layer halves time and reduces the number of maps.
        self.conv1 = nn.Conv1d(n_channels, 32, 7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, n_code, 7, stride=2, padding=3)
        # Decoder: each transposed layer doubles time; the last one returns all channels.
        self.up1 = nn.ConvTranspose1d(n_code, 32, 7, stride=2, padding=3)
        self.up2 = nn.ConvTranspose1d(32, n_channels, 7, stride=2, padding=3)
        self.drop = nn.Dropout(drop_prob)

    def encode(self, x):
        x = self.drop(F.elu(self.conv1(x)))
        return F.elu(self.conv2(x))

    def decode(self, z):
        n = z.shape[-1]
        z = self.drop(F.elu(self.up1(z, output_size=(2 * n,))))
        return self.up2(z, output_size=(4 * n,))

    def forward(self, x):
        return self.decode(self.encode(x))

def test_network(model, dataset, batch_size=16):
    """One forward and backward pass; fails loudly if shapes or gradients are wrong."""
    (batch,) = next(iter(DataLoader(dataset, batch_size=batch_size)))
    model.train()
    out = model(batch)
    assert out.shape == batch.shape, f'output {tuple(out.shape)} != input {tuple(batch.shape)}'
    F.mse_loss(out, batch).backward()
    assert all(p.grad is not None for p in model.parameters()), 'a parameter received no gradient'
    model.zero_grad()
    return True

torch.manual_seed(SEED)
ae = Autoencoder(n_channels=n_channels)
print(ae)
print('Trainable parameters:', sum(p.numel() for p in ae.parameters()))
print('test_network:', test_network(ae, datasets['train']))
with torch.no_grad():
    ae.eval()
    print('Code shape for 2 trials:', tuple(ae.encode(datasets['train'].tensors[0][:2]).shape))

# %% [markdown]
# ### Interpret and check
#
# The code is 16 × 60 per trial, as calculated by hand. Most parameters are in the first encoder layer and the last decoder layer, which connect to all 64 channels. If your own class failed `test_network`, read the assertion message: a shape mismatch usually comes from a missing `output_size`.

# %% [markdown]
# ## Train the autoencoder
# The loop is the Lesson 17 loop with two changes: the target is the input itself, and the loss is mean squared error.

# %% [markdown]
# ### Step 2.1 · training with validation checkpoints and progress views
#
# **1.** Each epoch: shuffled run-4 batches; zero_grad, reconstruct, MSE against the same batch, backward, step.
#
# **2.** After each epoch, run-8 MSE is computed in evaluation mode; the best weights are copied.
#
# **3.** Every 20 epochs, one validation trial's C3 signal is plotted with its reconstruction, like the tutorial's `show_every` views.

# %%
def reconstruction_mse(model, dataset):
    (inputs,) = dataset.tensors
    model.eval()
    with torch.no_grad():
        return F.mse_loss(model(inputs.to(device)), inputs.to(device)).item()

c3 = epochs.ch_names.index('C3')
view_trial = datasets['validation'].tensors[0][:1]
times = np.arange(n_times) / 80 + 0.5

def show_reconstruction(model, epoch):
    model.eval()
    with torch.no_grad():
        recon = model(view_trial.to(device)).cpu()
    plt.figure(figsize=(9, 2.6))
    plt.plot(times, view_trial[0, c3], color='0.5', label='Recorded (standardized)')
    plt.plot(times, recon[0, c3], color='#35688a', label='Reconstruction')
    plt.title(f'Validation trial, C3, after epoch {epoch}'); plt.xlabel('Time after cue (s)')
    plt.legend(loc='upper right'); plt.show()

torch.manual_seed(SEED)
ae = Autoencoder(n_channels=n_channels).to(device)
optimizer = optim.Adam(ae.parameters(), lr=3e-3)
loader = DataLoader(datasets['train'], batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
history, best_loss, best_state = [], float('inf'), None
start = time.time()
for epoch in range(1, 61):
    ae.train()
    running = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(ae(batch), batch)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(batch)
    val_mse = reconstruction_mse(ae, datasets['validation'])
    history.append({'epoch': epoch, 'train_mse': running / len(datasets['train']), 'val_mse': val_mse})
    if val_mse < best_loss:
        best_loss, best_state = val_mse, copy.deepcopy(ae.state_dict())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:2d}  train MSE {history[-1]['train_mse']:.3f}  validation MSE {val_mse:.3f}")
        show_reconstruction(ae, epoch)
ae.load_state_dict(best_state)
history = pd.DataFrame(history)
print(f'Training time {time.time() - start:.1f} s; best validation MSE {best_loss:.3f} at epoch {int(history.val_mse.idxmin()) + 1}')

# %% [markdown]
# ### Step 2.2 · learning curves
#
# **1.** Plot training and validation MSE per epoch, with the zero-predictor level for reference.

# %%
fig, ax = plt.subplots()
ax.plot(history.epoch, history.train_mse, label='Training (run 4)')
ax.plot(history.epoch, history.val_mse, label='Validation (run 8)')
ax.axhline(1.0, color='0.6', ls='--', label='Zero predictor ≈ 1')
ax.set(xlabel='Epoch', ylabel='MSE (standardized units²)', title='Autoencoder learning curves')
ax.legend(); plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Both curves should fall well below 1. Validation MSE is usually higher than training MSE because run 8 is a different recording block, and its channel scaling came from run 4. The reconstructions above typically capture the slower, larger oscillations before the fast detail.

# %% [markdown]
# ## Practice 3 · Is the autoencoder better than a linear code of the same size?
#
# **Independent exercise:** Fit a four-component spatial PCA on run-4 data (every time sample is one 64-channel observation), reconstruct run 8 from those components, and compare its MSE with the zero predictor and the autoencoder. Both codes keep 960 values per trial.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Reshape (N, C, T) to (N·T, C) with `transpose(0, 2, 1).reshape(-1, C)`; `PCA.inverse_transform` returns the reconstruction.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
def to_samples(arr):
    return arr.transpose(0, 2, 1).reshape(-1, arr.shape[1])

pca = PCA(n_components=4).fit(to_samples(X_z[masks['train']]))
def pca_mse(role):
    samples = to_samples(X_z[masks[role]])
    return np.mean((pca.inverse_transform(pca.transform(samples)) - samples) ** 2)

baselines = pd.DataFrame({
    'zero predictor': [np.mean(X_z[masks[r]] ** 2) for r in ['validation', 'test']],
    'spatial PCA, 4 components (960 values)': [pca_mse(r) for r in ['validation', 'test']],
    'autoencoder (960 values)': [reconstruction_mse(ae, datasets[r]) for r in ['validation', 'test']]},
    index=['run 8 (validation)', 'run 12 (test)']).round(3)
print(baselines.T)

# %% [markdown]
# ### Interpret and check
#
# Report all three numbers. If the autoencoder is not clearly below the PCA line, its extra parameters bought nothing for reconstruction. The test-run row is computed once, after the checkpoint was fixed on run 8.

# %% [markdown]
# ## Encoding trials
# The trained encoder turns each trial into a 16 × 60 code. The MNIST tutorial displays this code as an image.

# %% [markdown]
# ### Step 3.1 · look at the code of one trial
#
# **1.** Encode one left and one right validation trial.
#
# **2.** Display each code as a 16 × 60 image: rows are code maps, columns are time steps of 0.05 s.

# %%
ae.eval()
val_inputs = datasets['validation'].tensors[0]
val_labels = y[masks['validation']]
with torch.no_grad():
    codes = {name: ae.encode(val_inputs[np.flatnonzero(val_labels == cls)[0]][None])[0].numpy()
             for cls, name in enumerate(['left', 'right'])}
fig, axes = plt.subplots(1, 2, figsize=(11, 3), constrained_layout=True)
for ax, (name, code) in zip(axes, codes.items()):
    im = ax.imshow(code, aspect='auto', cmap='Reds', extent=[0.5, 3.5, 15.5, -0.5])
    ax.set(title=f'Code of a {name}-imagery trial', xlabel='Time after cue (s)', ylabel='Code map')
fig.colorbar(im, ax=axes, label='Activation')
plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Two single trials will differ for many reasons besides the task. Whether code maps systematically differ between classes is a question for many trials and a classifier, which the next practice answers.

# %% [markdown]
# ## Practice 4 · Does the code carry motor-imagery information?
#
# **Independent exercise:** Compute the log-variance of each of the 16 code maps for every trial, train shrinkage LDA on run 4, and report validation and test balanced accuracy. Compare with log-variance of 16 spatial PCA components and of all 64 channels.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Log-variance over time is a band-power feature (Lesson 06) computed on a different set of signals.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
with torch.no_grad():
    all_codes = ae.encode(torch.tensor(X_z)).numpy()
pca16 = PCA(n_components=16).fit(to_samples(X_z[masks['train']]))
pca16_signals = pca16.transform(to_samples(X_z)).reshape(len(X_z), n_times, 16).transpose(0, 2, 1)
feature_sets = {'autoencoder code, 16 maps': np.log(all_codes.var(axis=2) + 1e-8),
                'spatial PCA, 16 components': np.log(pca16_signals.var(axis=2)),
                'all 64 channels': np.log(X_z.var(axis=2))}
rows = {}
for name, feats in feature_sets.items():
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(feats[masks['train']], y[masks['train']])
    rows[name] = [balanced_accuracy_score(y[masks[r]], clf.predict(feats[masks[r]])) for r in ['validation', 'test']]
probe = pd.DataFrame(rows, index=['validation (run 8)', 'test (run 12)']).T.round(3)
print(probe)

# %% [markdown]
# ### Interpret and check
#
# The autoencoder was trained to reconstruct, not to separate classes. If its 16 maps decode worse than all 64 channels, the compression discarded task information that MSE did not reward. That is the expected risk of an unsupervised code, not a failure of the code. Compare validation rows to choose; read the test row once.

# %% [markdown]
# ## Practice 5 · Train mode versus evaluation mode
#
# **Independent exercise:** Reconstruct the validation run three times with `ae.train()` and three times with `ae.eval()`. Report the MSE each time and explain the difference.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Dropout is random only in training mode.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
for mode in ['train', 'eval']:
    getattr(ae, mode)()
    with torch.no_grad():
        values = [F.mse_loss(ae(val_inputs), val_inputs).item() for _ in range(3)]
    print(f'{mode:5s} mode MSE:', np.round(values, 4))
ae.eval()

# %% [markdown]
# ### Interpret and check
#
# In training mode dropout zeroes random activations, so each pass gives a different, usually higher, error. Evaluation mode is deterministic. Validation and test numbers must always be computed in evaluation mode, as the MNIST tutorial's `net.eval()` and `net.train()` calls do.

# %% [markdown]
# ## Recorded-signal inspection with MNE-Python
#
# Which frequencies does the autoencoder preserve? Put the reconstructions back into volts, wrap them as an MNE `EpochsArray` with the original channel information, and compare power spectra with the recorded validation epochs.

# %%
with torch.no_grad():
    recon_val = ae(val_inputs).numpy() * ch_std + ch_mean
recorded = mne.EpochsArray(X[masks['validation']], epochs.info, verbose=False)
reconstructed = mne.EpochsArray(recon_val, epochs.info, verbose=False)
fig, ax = plt.subplots(figsize=(9, 3.8))
for name, ep_obj, color in [('Recorded', recorded, '0.4'), ('Reconstructed', reconstructed, '#35688a')]:
    spectrum = ep_obj.compute_psd(fmin=2, fmax=38, picks=['C3', 'C4'])
    psd_vals, freqs = spectrum.get_data(return_freqs=True)
    ax.semilogy(freqs, psd_vals.mean(axis=(0, 1)), color=color, label=name)
ax.set(xlabel='Frequency (Hz)', ylabel='PSD (V²/Hz)', title='C3/C4 spectra, validation run')
ax.legend(); plt.show()

# %% [markdown]
# ### Figure interpretation and independent exercise
#
# Where the two curves overlap, the code preserved that frequency range; where the reconstructed curve falls below, it was lost.
#
# **Exercise.** If the reconstruction keeps 4–12 Hz but loses 15–30 Hz, what does that imply for decoding beta desynchronization from the code? Name one claim the figure cannot support.
#
# **Reference interpretation.** Beta-band task information would be weakened or lost, consistent with a lower probe score. The figure cannot show that the preserved power is neural rather than ocular or muscular, nor that the code maps correspond to brain sources.

# %% [markdown]
# ## Practice 6 · explain the complete method
#
# Independently explain what the autoencoder was trained to do, how it was evaluated, and why its code may or may not help a BCI. Include one baseline and one claim the results do not establish.
#
# **My explanation:** _write here._

# %% [markdown]
# ### Worked answer · compare your reasoning
#
# The network compressed each standardized run-4 epoch to 16 × 60 values and reconstructed it, minimizing MSE; run 8 chose the checkpoint and run 12 gave a final reconstruction error. Its MSE was compared with zeros and with a four-component spatial PCA of the same size. A separate LDA probe asked whether code-map log-variance separates left and right imagery. The results do not establish that the code isolates brain activity, that it would transfer to new participants, or that it is optimal for classification; a code learned to reconstruct can discard small task-related changes.

# %% [markdown]
# ## If your result is different
#
# If the decoder output is one sample short, pass `output_size` to every transposed layer. If MSE stays near 1, check that inputs are standardized and the learning rate is not too small. If validation MSE is much higher than training MSE, the code may be memorizing run 4; try fewer epochs or more dropout. If a download fails, retry when PhysioNet is reachable; do not replace real data with simulated values.

# %% [markdown]
# ## Can you now do this independently?
#
# - Compute the lengths of strided and transposed convolutions and design a decoder that restores the input.
# - Write an autoencoder with `encode` and `decode` methods and pass a `test_network` check.
# - Compare reconstruction error with a zero predictor and a linear code of the same size.
# - Probe a learned code for task information and explain why low MSE does not guarantee it.
#
# If one item is unclear, return to the associated practice.

# %% [markdown]
# ## Next steps and sources
# Lesson 13 trains a denoising autoencoder with injected noise. The capstone (Lesson 15) is a place to compare a learned code with hand-designed features under a full validation protocol.
#
# [PyTorch ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html) · [Odena et al. (2016), Deconvolution and checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/) · [MNE EpochsArray](https://mne.tools/stable/generated/mne.EpochsArray.html).

# %% [markdown]
# ## References and further reading
#
# 1. [Hinton and Salakhutdinov (2006). Reducing the dimensionality of data with neural networks](https://doi.org/10.1126/science.1127647).
# 2. [Vincent et al. (2010). Stacked denoising autoencoders](https://www.jmlr.org/papers/v11/vincent10a.html).
# 3. [Roy et al. (2019). Deep learning-based electroencephalography analysis: a systematic review](https://doi.org/10.1088/1741-2552/ab260c).
# 4. [Odena, Dumoulin and Olah (2016). Deconvolution and checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/).
# 5. [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
#
# Numerical outcomes in this notebook refer only to the selected participants, runs and processing; they are not population performance estimates.
