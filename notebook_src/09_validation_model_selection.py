# %% [markdown]
# # 09 · Honest validation and model selection
#
# **ROBT613 · Brain–Computer Interfaces** | Teaching session + independent lab
#
# ## Goal
# Compare classifiers with nested run-grouped validation, distinguish uncertainty units and diagnose leakage.
#
# **Data:** EEGBCI subject 1, three independent run groups
#
# Run cells from top to bottom in a fresh CPU runtime. No previous notebook state is required.

# %% [markdown]
# ## How to study this notebook
#
# This is both the classroom lesson and the independent-study workbook. Everything needed for the exercises—questions, hints, executable solutions, checks and explanations—is here. Work from top to bottom in a fresh runtime.
#
# 1. Read the question and calculate a small example on paper.
# 2. Write your prediction before running the next code cell.
# 3. Try the practice task in its workspace.
# 4. Continue to the worked solution and compare the reasoning, not just the number.
# 5. Change one parameter and explain what the result means.
#
# **For a live class:** pause at each “Try it” heading. The solution follows in the same notebook, so no separate answer document is required. Saved figures support reading without execution; downloading real data and rerunning cells requires internet on the first run. Code comments explain each analysis statement, and longer loops are explained before execution.
#
# **Prerequisites:** basic Python arrays, arithmetic and plotting. The symbol guide below defines the mathematical notation used here. These lessons stay at the sensor level; EEG source imaging is outside the course.

# %% [markdown]
# ## The question for today
#
# Two models differ by five percentage points. Was one better, or did the evaluation accidentally reward reuse of the same recording? We turn “train/test split” into an explicit scientific claim about future use.
#
# ### By the end you should be able to
#
# - Match run, session and participant grouping to the target use case.
# - Explain inner selection and outer evaluation in nested validation.
# - Calculate class-balanced metrics and distinguish scores from uncertainty.
# - Identify every learned preprocessing operation that must be refitted within folds.
#
# **How to work:** predict each result before running it, execute one cell at a time, and write a short interpretation. The first worked examples use controlled arrays; the later walkthrough and applied practice use the dataset stated above. Practice workspaces, hints and worked solutions are placed beside the relevant methods. Complete your attempt before continuing to the reference solution.

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
# ## Visual route through the lesson
#
# Follow the arrows before running the analysis. For each box, say what the input represents, what changes, and what must be preserved.

# %%
# Drawing code for the lesson map; no analysis data are transformed here.
from matplotlib.patches import FancyBboxPatch
map_steps=['Define future use\nrun / session / person', 'Outer partition\nreserve test group', 'Inner partition\nselect hyperparameters', 'Refit on outer train\nall learned steps', 'Outer score table\nindependent scope']
fig,map_ax=plt.subplots(figsize=(12,3.1),constrained_layout=True)
map_ax.set(xlim=(-.1,12),ylim=(-.3,2.4));map_ax.axis('off')
for map_i,map_label in enumerate(map_steps):
    map_x=map_i*2.4
    map_ax.add_patch(FancyBboxPatch((map_x,.45),2.05,1.1,
        boxstyle='round,pad=0.08',facecolor='#edf3f7',edgecolor='#35688a',linewidth=1.5))
    map_ax.text(map_x+1.025,1.02,map_label,ha='center',va='center',fontsize=10)
    map_ax.text(map_x+1.025,1.83,str(map_i+1),ha='center',weight='bold',color='#35688a')
    if map_i<4:map_ax.annotate('',xy=(map_x+2.3,1),xytext=(map_x+2.13,1),arrowprops=dict(arrowstyle='->',lw=1.5))
map_ax.text(5.9,-.08,'Read left to right. Keep units, observation identities and evaluation boundaries attached to the data.',ha='center',fontsize=10)
map_ax.set_title('Lesson 09 · from measurement to an interpretable result',fontsize=14,pad=12)
plt.show()

# %% [markdown]
# **Read the map:** the arrows represent processing order, not permission to fit on all observations. When a stage learns parameters, keep evaluation data outside that fit. The map is also available as text: Define future use: run / session / person → Outer partition: reserve test group → Inner partition: select hyperparameters → Refit on outer train: all learned steps → Outer score table: independent scope.

# %% [markdown]
# ## Symbols and a calculation by hand
#
# $G$: grouping variable; $h$: hyperparameters; $k$: outer fold; $s_k$: outer score; $K$: classes.
#
# ### Derive the operation before calling the library
#
# Within outer fold $k$, choose $\hat h_k=\arg\max_h\operatorname{CV}_{inner}(h,D_{train,k})$. Refit that configuration on $D_{train,k}$ and score it once on $D_{test,k}$. The test group takes no part in either choice or fit.
#
# For three runs A/B/C, one outer fold reserves C and performs inner A→B and B→A comparisons. It then refits on A+B and evaluates on C. Rotate the outer held-out run to obtain three descriptive scores. Balanced accuracy is $K^{-1}\sum_kTP_k/(TP_k+FN_k)$: an all-majority predictor on a binary 90/10 dataset has accuracy 0.9 but balanced accuracy $(1+0)/2=0.5$.

# %% [markdown]
# ### Your paper calculation
#
# Rewrite one equation with the numerical example above. Name the input units and output units, and identify the axis being reduced or transformed.
#
# **My calculation:** _write your intermediate steps here before continuing._

# %% [markdown]
# ### Choose the prediction problem before the split
#
# A random trial split asks whether a model generalizes to another trial under closely shared recording conditions. A held-out run asks a stronger question about a new recording block. A held-out session tests changes across sessions; a held-out participant tests transfer across people. None is a substitute for another. The partition is part of the experiment, not a final programming detail.
#
# Adjacent EEG windows can share samples, artifacts and slow changes. Even non-overlapping trials in the same run share impedance, fatigue, amplifier conditions and task order. Grouped cross-validation holds entire groups out. If the intended deployment has a temporal direction, ensure the evaluation also respects that direction; ordinary grouped folds do not by themselves simulate chronological deployment.

# %% [markdown]
# ### Learn only from the training partition
#
# A transformation is “learned” whenever it estimates parameters from data. Standardization estimates means and variances. PCA estimates directions. CSP estimates label-dependent spatial filters. Feature selection estimates which columns to keep. All belong inside the fold when evaluating the full pipeline. A fixed physical unit conversion does not estimate a parameter and can be applied consistently in advance.
#
# Hyperparameters include component count, regularization, bands selected by performance and decision thresholds selected by performance. Repeatedly trying them against a test set makes the test set part of the training process, even if no gradient was computed on it.

# %% [markdown]
# ### Two loops with different jobs
#
# For each outer fold, set aside the outer test group. Within the remaining groups, inner validation selects one candidate configuration. Refit that configuration on all outer training data, then evaluate once on the outer test group. The outer scores describe the model-selection procedure, not one model fitted to every observation.
#
# $$\hat h_k=\arg\max_h\operatorname{CV}_{inner}(h;D_{train,k}),\qquad s_k=\operatorname{score}(\operatorname{fit}(\hat h_k,D_{train,k}),D_{test,k}).$$
#
# The nested example has few groups, so estimates can be noisy. A table of individual run scores is more informative than a mean with many decimal places. Fold scores are not independent participants; treating them as such produces unjustified precision.

# %% [markdown]
# ### Metrics and uncertainty
#
# Balanced accuracy averages class recalls, $BA=K^{-1}\sum_k TP_k/(TP_k+FN_k)$. ROC AUC describes ranking for a binary score; it does not choose a useful operating threshold. A confusion matrix shows which classes are confused but depends on a decision rule. Report the class counts and the evaluation unit alongside any metric.
#
# Permutation tests require exchangeability under the null. Arbitrarily shuffling individual trials can break temporal or block structure. Bootstrap intervals likewise depend on the unit resampled: trials, runs and participants support different claims. This lesson teaches the reasoning and avoids presenting three folds as a precise population confidence interval.

# %% [markdown]
# ## Visual intuition · Separate outer testing from inner selection
#
# **Try it:** Trace the row where C is the outer test run. Which observations can select a component count?

# %%
from matplotlib.colors import ListedColormap
vis_roles=np.array([[2,1,1],[1,2,1],[1,1,2],[0,1,2],[1,0,2]])
vis_labels=['Outer fold 1','Outer fold 2','Outer fold 3','Fold 3 inner split 1','Fold 3 inner split 2']
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
ax.imshow(vis_roles,cmap=ListedColormap(['#e5b453','#b9d1e0','#465461']),vmin=0,vmax=2,aspect='auto')
for row in range(5):
    for col in range(3):ax.text(col,row,['Validate','Train','TEST: untouched'][vis_roles[row,col]],ha='center',va='center',color='white' if vis_roles[row,col]==2 else 'black')
ax.set(xticks=range(3),xticklabels=['Run A','Run B','Run C'],yticks=range(5),yticklabels=vis_labels,title='Partition map · the outer test run stays outside both inner splits')
plt.show()

# %% [markdown]
# ### Worked interpretation
#
# In outer fold 3, A and B exchange training/validation roles while C remains untouched. After selection, refit on A+B, then evaluate once on C. The two lower rows illustrate only fold 3, not two additional independent tests.

# %% [markdown]
# ## Guided practice 1 · imbalanced accuracy
#
# Construct a 90/10 dataset and predict only the majority class.
#
# **Try it on paper:** predict the output’s shape, sign or approximate value. State which assumption makes your prediction valid.
#
# **My prediction:** _write here._
#
# **Hint:** trace one sample, one channel or one trial through the calculation before considering the full array.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
from sklearn.metrics import accuracy_score
demo_y=np.array([0]*90+[1]*10)
demo_pred=np.zeros(100,dtype=int)
print('Accuracy:',accuracy_score(demo_y,demo_pred),'Balanced accuracy:',balanced_accuracy_score(demo_y,demo_pred))

# %% [markdown]
# ### Why this result makes sense
#
# Ninety percent accuracy can coexist with zero sensitivity to the minority class. The metric must reflect the scientific objective.
#
# **Check your understanding:** change one numerical parameter, predict the direction of change, and rerun. If the result disagrees, inspect units and axes before changing the method.

# %% [markdown]
# ## Guided practice 2 · inspect grouped folds
#
# Use three groups and print the identifiers in each partition. Assert that no group crosses a boundary.
#
# **Try it on paper:** predict the output’s shape, sign or approximate value. State which assumption makes your prediction valid.
#
# **My prediction:** _write here._
#
# **Hint:** trace one sample, one channel or one trial through the calculation before considering the full array.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_groups=np.repeat(['run A','run B','run C'],4)
demo_y=np.tile([0,1,0,1],3)
for train_idx,test_idx in GroupKFold(3).split(np.zeros((12,1)),demo_y,demo_groups):
    assert set(demo_groups[train_idx]).isdisjoint(demo_groups[test_idx])
    print('Train:',np.unique(demo_groups[train_idx]),'Test:',np.unique(demo_groups[test_idx]))

# %% [markdown]
# ### Why this result makes sense
#
# Checking group identities is more direct than assuming a splitter did what the experiment required.
#
# **Check your understanding:** change one numerical parameter, predict the direction of change, and rerun. If the result disagrees, inspect units and axes before changing the method.

# %% [markdown]
# ## Guided practice 3 · fitting the scaler
#
# Compare training-only and pooled means when the held-out data have shifted.
#
# **Try it on paper:** predict the output’s shape, sign or approximate value. State which assumption makes your prediction valid.
#
# **My prediction:** _write here._
#
# **Hint:** trace one sample, one channel or one trial through the calculation before considering the full array.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_train=np.array([[0.],[1.],[2.]])
demo_test=np.array([[10.],[11.]])
demo_scaler=StandardScaler().fit(demo_train)
print('Training mean:',demo_scaler.mean_)
print('Pooled mean:',np.vstack([demo_train,demo_test]).mean(axis=0))
print('Test transformed by training parameters:',demo_scaler.transform(demo_test).ravel())

# %% [markdown]
# ### Why this result makes sense
#
# The shifted test values should remain shifted. Using their mean during fitting partly adapts to the evaluation distribution and changes the question being measured.
#
# **Check your understanding:** change one numerical parameter, predict the direction of change, and rerun. If the result disagrees, inspect units and axes before changing the method.

# %% [markdown]
# ## Guided practice 4 · selection optimism
#
# Imagine twenty equally good candidates whose validation estimates contain noise. Compare the maximum observed estimate with their shared true value.
#
# **Try it on paper:** predict the output’s shape, sign or approximate value. State which assumption makes your prediction valid.
#
# **My prediction:** _write here._
#
# **Hint:** trace one sample, one channel or one trial through the calculation before considering the full array.

# %% [markdown]
# ### Worked solution · read one statement at a time

# %%
demo_rng=np.random.default_rng(613)
demo_estimates=.6+demo_rng.normal(0,.05,size=(1000,20))
print('Mean of all estimates:',demo_estimates.mean())
print('Mean selected maximum:',demo_estimates.max(axis=1).mean())

# %% [markdown]
# ### Why this result makes sense
#
# Selecting the largest noisy estimate inflates its apparent performance. An untouched outer evaluation is needed even when every candidate has identical true quality.
#
# **Check your understanding:** change one numerical parameter, predict the direction of change, and rerun. If the result disagrees, inspect units and axes before changing the method.

# %% [markdown]
# ## Practice 1 · Group assertion
#
# Implement groups_disjoint to return whether train and test contain no shared group identifiers.
#
# **Try it:** complete the function below before reading its solution. The template deliberately returns None so that an unfinished attempt does not interrupt the rest of the lesson.

# %%
def groups_disjoint(train_groups, test_groups):
    # TODO: return a Boolean.
    return None

# %% [markdown]
# ### Hint
#
# Use the equation above and keep the trial/channel axes intact unless the requested output removes them. Test the smallest example by hand first.

# %% [markdown]
# ### Worked solution
#
# Compare this implementation with your attempt. The next cell checks the reference answer on a concrete numerical case.

# %%
def groups_disjoint(train_groups,test_groups):
    return set(train_groups).isdisjoint(set(test_groups))

# %%
answer=groups_disjoint(['A','B'],['C'])
if answer is not None:
    assert answer
    assert not groups_disjoint(['A','B'],['B','C'])
    print('Group checks passed.')
else: print('Exercise pending: implement groups_disjoint.')

# %% [markdown]
# ## Apply the ideas to the complete pipeline
#
# The next stages use the real recording or the explicitly labeled simulation described in the lesson. Keep preprocessing, feature extraction and evaluation decisions visible. Read each stage’s explanation before running its code.

# %% [markdown]
# ### Method reference for the pipeline
#
# A test score estimates performance on the distribution represented by the held-out unit. Holding out trials, runs, sessions and people answers different questions. If adjacent windows share EEG samples, random window splitting can make generalization look much stronger than it is.
#
# Nested cross-validation separates model selection from evaluation: for each outer split, choose hyperparameters using only inner folds of the outer training set, refit, then score the outer test set once. For balanced accuracy $BA=K^{-1}\sum_k TP_k/(TP_k+FN_k)$. A confusion matrix reveals which classes contribute to a mean score.
#
# A bootstrap must resample independent units. Thousands of correlated epochs from one participant do not provide thousands of independent estimates of population performance. Three run scores are too few for a precise uncertainty estimate. Permutation tests must respect exchangeability (e.g. experimental blocks); shuffling arbitrary overlapping flash labels is not automatically valid.

# %% [markdown]
# ## Load an auditable benchmark
# This lesson deliberately keeps all three run IDs.
#
# Preserve run groups while loading and processing each run separately. The nested evaluation needs those identifiers to distinguish inner development from outer assessment.

# %% [markdown]
# ### Step 1.1 · trace the next operation
#
# **1.** Import the named tools used in this step.
#
# **2.** Make the analysis choice visible and fixed before inspecting evaluation performance.
#
# **3.** Store this intermediate result so the next operation can be traced and inspected.

# %%
# Import the named tools used in this step.
from mne.datasets import eegbci
# Make the analysis choice visible and fixed before inspecting evaluation performance.
runs = [4, 8, 12]
# Store this intermediate result so the next operation can be traced and inspected.
parts, groups = ([], [])

# %% [markdown]
# ### Step 1.2 · trace the next operation
#
# **1.** Process each run independently; carry its run ID into every resulting trial.

# %%
# Process each run independently; carry its run ID into every resulting trial.
for run in runs:
    # Retrieve this run independently so filtering does not cross a run boundary.
    paths = eegbci.load_data(1, [run], path=DATA_ROOT, update_path=False)
    # Load this run’s continuous recording and its sampling metadata.
    raw_run = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
    # Make sensor names consistent with the montage and later channel selections.
    eegbci.standardize(raw_run)
    # Attach sensor locations; no anatomical inverse model is used.
    raw_run.set_montage('standard_1005')
    # Use the same reference convention for every run.
    raw_run.set_eeg_reference('average', projection=False)
    # Filter the continuous run before extracting its trials.
    raw_run.filter(8, 30, fir_design='firwin')
    # Map only the documented imagery cues to the two class codes.
    events, _ = mne.events_from_annotations(raw_run, event_id={'T1': 1, 'T2': 2})
    # Extract the stated post-cue interval while retaining condition labels.
    ep = mne.Epochs(raw_run, events, {'left': 1, 'right': 2}, tmin=0.5, tmax=3.5, baseline=None, preload=True, picks='eeg', reject_by_annotation=True)
    # Retain this run’s epoch object for the later concatenation.
    parts.append(ep)
    # Append exactly one run identifier for every retained trial.
    groups.extend([run] * len(ep))

# %% [markdown]
# ### Step 1.3 · trace the next operation
#
# **1.** Join trial collections while retaining one aligned group label per trial.
#
# **2.** Expose the numerical array; EEG values are in volts and the final axis is time.
#
# **3.** Keep one label or grouping identifier per trial in exactly the same order as the EEG array.
#
# **4.** Make an explicit NumPy vector while preserving its current row order.
#
# **5.** Display a bounded diagnostic; read the units, shape or partition rather than treating output as a success label.

# %%
# Join trial collections while retaining one aligned group label per trial.
epochs = mne.concatenate_epochs(parts)
# Expose the numerical array; EEG values are in volts and the final axis is time.
X = epochs.get_data(copy=True)
# Keep one label or grouping identifier per trial in exactly the same order as the EEG array.
y = epochs.events[:, 2] - 1
# Make an explicit NumPy vector while preserving its current row order.
groups = np.asarray(groups)
# Display a bounded diagnostic; read the units, shape or partition rather than treating output as a success label.
print('Epochs:', X.shape, 'run counts:', pd.Series(groups).value_counts().to_dict())

# %% [markdown]
# ### Inspect and interpret
#
# Inspect group sizes and class balance. Consider whether the available number of groups is adequate for a large hyperparameter search.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Practice 2 · List the groups in one outer and inner split
#
# **Try it:** Print one outer partition and the two inner partitions within its training set. Verify that the outer test run never appears inside.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Inner indices refer to the outer-training array, not directly to the original full array.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
lab_outer_train,lab_outer_test=next(GroupKFold(3).split(X,y,groups))
print('Outer test:',np.unique(groups[lab_outer_test]))
for lab_it,lab_iv in GroupKFold(2).split(X[lab_outer_train],y[lab_outer_train],groups[lab_outer_train]):
    lab_tgroups=groups[lab_outer_train][lab_it]
    lab_vgroups=groups[lab_outer_train][lab_iv]
    print('Inner train:',np.unique(lab_tgroups),'inner validation:',np.unique(lab_vgroups))
    assert set(lab_tgroups).isdisjoint(groups[lab_outer_test])
    assert set(lab_vgroups).isdisjoint(groups[lab_outer_test])

# %% [markdown]
# ### Interpret and check
#
# The index mapping is a common source of mistakes. Inner indices select rows within outer-training data. Applying those indices directly to full X would silently select different observations and could invalidate the split.

# %% [markdown]
# ## Nested model comparison
# With three runs, each outer training fold contains two runs, so the inner loop uses two groups. All learned transformations remain inside the pipeline.
#
# The outer loop holds out one run. Inner grouped search chooses CSP component count using only the other two runs. Each row records the held-out result for a complete model-selection procedure.

# %% [markdown]
# ### Step 2.1 · trace the next operation
#
# **1.** Import the named tools used in this step.
#
# **2.** Hold entire recording groups out rather than mixing neighboring trials.
#
# **3.** Initialize the collection that will retain outputs in the same order as the inputs.

# %%
# Import the named tools used in this step.
from mne.decoding import CSP
# Hold entire recording groups out rather than mixing neighboring trials.
outer = GroupKFold(3)
# Initialize the collection that will retain outputs in the same order as the inputs.
rows = []

# %% [markdown]
# ### Step 2.2 · trace the next operation
#
# **1.** Reserve an outer run, select the model only inside remaining runs, then score the untouched run.

# %%
# Reserve an outer run, select the model only inside remaining runs, then score the untouched run.
for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
    # Fit representation and classifier together within the permitted training partition.
    model = make_pipeline(CSP(n_components=4, reg='ledoit_wolf', log=True), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    # Search only within the outer-training runs; the outer test remains untouched.
    search = GridSearchCV(model, {'csp__n_components': [2, 4]}, cv=GroupKFold(2), scoring='balanced_accuracy', n_jobs=1)
    # Supply only outer-training observations and their inner grouping labels.
    search.fit(X[tr], y[tr], groups=groups[tr])
    # Score the held-out group after selection, averaging recalls across classes.
    score = balanced_accuracy_score(y[te], search.predict(X[te]))
    rows.append({'held_out_run': int(groups[te][0]), 'balanced_accuracy': score, 'components': search.best_params_['csp__n_components']})

# %% [markdown]
# ### Step 2.3 · trace the next operation
#
# **1.** Store this intermediate result so the next operation can be traced and inspected.
#
# **2.** Display a bounded diagnostic; read the units, shape or partition rather than treating output as a success label.

# %%
# Store this intermediate result so the next operation can be traced and inspected.
results = pd.DataFrame(rows)
# Display a bounded diagnostic; read the units, shape or partition rather than treating output as a success label.
print(results)

# %% [markdown]
# ### Step 2.4 · trace the next operation
#
# **1.** Create axes; plotting changes the display, not the analyzed data.
#
# **2.** Apply the stated operation to the current object; use the surrounding explanation to check its role.
#
# **3.** Apply the stated operation to the current object; use the surrounding explanation to check its role.
#
# **4.** Apply the stated operation to the current object; use the surrounding explanation to check its role.
#
# **5.** Apply the stated operation to the current object; use the surrounding explanation to check its role.
#
# **6.** Render the completed figure and inspect labels, units and the comparison.

# %%
# Create axes; plotting changes the display, not the analyzed data.
fig, ax = plt.subplots()
# Apply the stated operation to the current object; use the surrounding explanation to check its role.
ax.bar(results.held_out_run.astype(str), results.balanced_accuracy)
# Apply the stated operation to the current object; use the surrounding explanation to check its role.
ax.axhline(0.5, color='gray', linestyle='--', label='Balanced binary chance')
# Apply the stated operation to the current object; use the surrounding explanation to check its role.
ax.set(ylim=(0, 1), xlabel='Held-out run', ylabel='Balanced accuracy', title='Nested run transfer · subject 1')
# Apply the stated operation to the current object; use the surrounding explanation to check its role.
ax.legend()
# Render the completed figure and inspect labels, units and the comparison.
plt.show()

# %% [markdown]
# ### Inspect and interpret
#
# Compare outer scores, chosen configurations and run-specific errors. A variable selected configuration may indicate that the calibration sample is too small to identify one stable optimum.
#
# **Record in your notes:** the relevant shape/count or metric, the units where applicable, and one limitation of the inference.

# %% [markdown]
# ## Practice 3 · Report scores without artificial precision
#
# **Try it:** Display the per-run scores and their mean/range. Write a conclusion limited to these runs.
#
# **My reasoning / hand calculation:** _write here._

# %%
# Your attempt goes here. Work on copies and preserve the evaluation split.

# %% [markdown]
# ### Hint
#
# Three run scores from one person are not three independent participant estimates.

# %% [markdown]
# ### Worked solution
#
# Run the following calculation after attempting your own version.

# %%
print(results[['held_out_run','balanced_accuracy','components']])
print('Mean:',results.balanced_accuracy.mean())
print('Range:',results.balanced_accuracy.min(),results.balanced_accuracy.max())

# %% [markdown]
# ### Interpret and check
#
# A valid conclusion describes variation in held-out-run performance for this participant under the stated selection procedure. It cannot establish population performance or a precise confidence interval across people. Do not choose the best outer fold and report it as the overall result.

# %% [markdown]
# ## Practice 4 · Choose the evaluation unit for deployment
#
# **Try it:** Match three intended uses to appropriate partitions: new block today, another day, a new participant.
#
# **My reasoning / hand calculation:** _write here._

# %% [markdown]
# **My answer:** _write a short explanation before continuing._

# %% [markdown]
# ### Hint
#
# The partition should withhold the source of variation you claim to generalize across.

# %% [markdown]
# ### Worked solution
#
# A new block calls for held-out runs or chronological blocks; another day calls for held-out sessions/days; a new person calls for held-out participants. If deployment is chronological, the training data must precede the evaluation data. Grouping alone does not enforce time direction.

# %% [markdown]
# ## Practice 5 · Respond to test-set reuse
#
# **Try it:** After seeing the final score, a team changes bands repeatedly until accuracy improves. What can the revised score legitimately be called?
#
# **My reasoning / hand calculation:** _write here._

# %% [markdown]
# **My answer:** _write a short explanation before continuing._

# %% [markdown]
# ### Hint
#
# The test observations have now influenced development.

# %% [markdown]
# ### Worked solution
#
# It is an exploratory result on reused evaluation data. Document the changes and obtain a new independent test before making a confirmatory claim. There is no correction that restores the original untouched status merely by renaming the partition.

# %% [markdown]
# ## Practice 6 · explain the complete method
#
# Without looking back, explain the measurement, transformation, feature or summary, and the evaluation boundary. Include one failure mode and one claim the result does not establish.
#
# **My explanation:** _write here._

# %% [markdown]
# ### Worked answer · compare your reasoning
#
# The split defines the future-use claim. Inner validation selects a configuration; outer testing assesses that selection procedure. Every learned transformation respects the boundary. Repeated use of a test result for design turns it into development data.

# %% [markdown]
# ## If your result is different
#
# If inner splits fail, count distinct groups inside the outer-training set. Reduce the search/fold count appropriately; never borrow the outer test group to make a fold possible.
#
# If a dataset download fails, read the error and retry when the public host is reachable; do not silently replace real data with simulated values. If a notebook cell refers to an undefined variable, restart the kernel and run the preceding cells in order. Numerical scores can vary slightly with library versions; record versions and compare the protocol before concluding that a method changed.

# %% [markdown]
# ## Can you now do this independently?
#
# - Explain each arrow in the lesson map and the units at its boundaries.
# - Reproduce the hand calculation and point to its corresponding code.
# - Interpret the figures without turning a descriptive pattern into an unsupported causal claim.
# - Complete a practice task before reading its worked solution.
# - State which choices were fixed and which were learned from calibration data.
#
# If one item is unclear, return to the associated figure or practice section before the next lesson.

# %% [markdown]
# ## Next steps and sources
# [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) · [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
#
# Record package versions, subject/run IDs, preprocessing, split unit, random seed, and all exclusions with your results. Do not interpret a single participant as a population estimate.
