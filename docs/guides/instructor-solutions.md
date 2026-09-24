# Instructor reference answers and discussion guide

This guide is separate from the student notebooks so that exercises can be attempted first. The small function answers are executable reference implementations; open-ended investigations are assessed against the reasoning and evidence criteria below, not a predetermined accuracy. Data-dependent results must come from each student’s actual run.

The course is sensor-level. EEG source imaging, forward models, inverse models and anatomical localization are outside its scope.

For each lesson, use roughly 15 minutes for framing, 25–40 minutes for worked examples, 30–45 minutes for the real-data walkthrough, and 30–60 minutes for exercises. Split longer lessons across meetings. Download large datasets before class.

## 00_start_here

[Open student notebook](../../notebooks/00_start_here.ipynb)

### Exercise 1 reference implementation

```python
def to_volts(values_uV):
    return np.asarray(values_uV)*1e-6
```

### Discussion and assessment

Look for an explicit distinction between changing metadata and resampling samples; a 10 Hz peak moves if only sfreq changes. Shape answers must name all axes. Copies should protect the original object from in-place operations.

### Requested evidence

1. **Unit conversion:** Implement `to_volts` below and test positive, negative and zero values. Explain why the plotting label alone cannot repair incorrectly stored units.
2. **Sampling calculation:** At 250 Hz, find the sample index corresponding to 0.8 s. State whether your convention rounds, floors or interpolates for times off the grid.
3. **Axis detective:** Starting from `toy_epochs`, compute one scalar per trial. Then compute one waveform per channel. Write the resulting shapes before running the code.
4. **MNE object inspection:** Print the sampling rate, channel names and first five times of the recording in this notebook. Locate where montage positions are stored.
5. **Averaging experiment:** Repeat the simulation for trial counts 1, 4, 16 and 64. Plot error against count with and without the shared artifact. Explain which part follows the square-root rule.
6. **Exit ticket:** Draw the Raw → Epochs → Evoked sequence. Label what information is added or discarded at each arrow. Submit the drawing and one example of a plausible but incorrect axis operation.

## 01_p300_signal_to_epochs

[Open student notebook](../../notebooks/01_p300_signal_to_epochs.ipynb)

### Exercise 1 reference implementation

```python
def window_mean(data,times,start,stop):
    mask=(times>=start)&(times<stop)
    if not mask.any(): raise ValueError('Empty feature window')
    return data[:,:,mask].mean(-1)
```

### Discussion and assessment

Count imbalance should motivate class-specific recall and balanced accuracy. Window features should use fixed masks, preserve trial order and avoid selecting latency ranges from the final session. Students should distinguish class-average visibility from single-trial classification.

### Requested evidence

1. **Implement a temporal feature:** Complete `window_mean` below. Use a half-open interval [start, stop). Preserve the trial and channel axes and reject an empty window with a useful error.
2. **Audit the real dataset:** Create a table of target/non-target counts for every session. Compute target fraction and majority-class accuracy for each session. Explain why the label proportions matter.
3. **Single trial versus average:** Plot five target trials at Pz alongside their average. Use microvolts and a common axis. Describe variability that the average hides.
4. **Feature comparison:** Using calibration sessions only, compare 0.25–0.40 s and 0.40–0.60 s means. Report class distributions rather than choosing the window with the prettiest full-dataset ERP.
5. **Timing sensitivity:** Shift the synthetic response by 100 ms and recompute the fixed-window feature. Explain the implication of an uncorrected stimulus-marker delay.
6. **Exit ticket:** In 150 words, distinguish an ERP contrast, flash detection and character decoding. List the additional metadata needed for the last task.

## 02_p300_classification_speller

[Open student notebook](../../notebooks/02_p300_classification_speller.ipynb)

### Exercise 1 reference implementation

```python
def balanced_from_counts(tn,fp,fn,tp):
    if tp+fn==0 or tn+fp==0: raise ValueError('Both classes required')
    return .5*(tp/(tp+fn)+tn/(tn+fp))
```

### Discussion and assessment

Require session-disjoint development and final evaluation. Compare LDA and logistic regression using training-session validation. AUC uses continuous target scores; threshold changes affect confusion matrices. Simulated character selection must not be reported as real spelling accuracy.

### Requested evidence

1. **Metrics from counts:** Implement `balanced_from_counts`. Verify it gives 0.5 for an all-non-target classifier when both classes exist, regardless of imbalance.
2. **Baseline comparison:** Fit a dummy classifier and compare its accuracy and balanced accuracy with LDA on the same held-out session. Explain each difference.
3. **Training-only model choice:** Compare LDA with logistic regression using only calibration sessions for the choice. Write the choice down before reporting the held-out score.
4. **Speller aggregation:** Extend the worked flash example to three characters and verify row/column decoding against known indices. State the shape of every array.
5. **Speed–accuracy tradeoff:** Compute idealized ITR for the simulated repetition curve using 12 flashes/repetition, 0.125 s/flash and 2 s overhead. Handle the 0·log(0) limit safely and label the result simulated.
6. **Exit ticket:** Explain why a high AUC does not imply a high character rate. Include thresholding, repetitions, metadata, errors and overhead in your answer.

## 03_filtering_sampling

[Open student notebook](../../notebooks/03_filtering_sampling.ipynb)

### Exercise 1 reference implementation

```python
def fir_delay(n_taps,sampling_rate):
    return (n_taps-1)/(2*sampling_rate)
```

### Discussion and assessment

Longer symmetric FIR filters have larger causal group delay and sharper transitions. Inspect impulse support as well as frequency response. Direct downsampling introduces an aliased 30 Hz component in the worked example. A low-pass may already attenuate mains; justify a notch from spectra.

### Requested evidence

1. **Delay calculation:** Implement `fir_delay`. Check that doubling the sampling rate halves delay in seconds for the same tap count.
2. **Filter report:** Write a complete report for the real-data filter used in this notebook. Locate MNE’s filter design output and identify details absent from “8–30 Hz”.
3. **Task-specific design:** Design one candidate P300 filter and one motor-imagery filter. Plot their frequency responses and justify the bands in terms of the measured phenomena.
4. **Notch decision:** Inspect line-frequency power before adding a notch. Explain whether a notch remains necessary after your low-pass and what evidence would change your choice.
5. **Boundary experiment:** Filter a short impulse-containing epoch alone and inside a longer padded signal. Compare the retained interior; explain the edge differences.
6. **Exit ticket:** Explain to a teammate why zero-phase offline filtering cannot be copied unchanged into a real-time decoder. Include future samples and delay in the answer.

## 04_artifacts_noise_cancellation

[Open student notebook](../../notebooks/04_artifacts_noise_cancellation.ipynb)

### Exercise 1 reference implementation

```python
def average_reference(values):
    return values-values.mean(axis=0,keepdims=True)
```

### Discussion and assessment

A good QC answer combines trace, PSD and task timing. Regression fails under changed nuisance coupling or when the nuisance reference contains wanted activity. ICA component index or variance alone is insufficient evidence. Removing components changes every retained trial; rejecting epochs changes the sample.

### Requested evidence

1. **Implement the reference:** Implement average_reference for a channel × time array without modifying the input.
2. **QC report:** Select three channels from the real QC table. Compare their traces and spectra; write evidence for keeping or marking each, without relying only on rank in a table.
3. **Regression stress test:** Change the contamination coefficient between calibration and evaluation in the controlled example. Report held-out RMSE and explain the stationarity assumption.
4. **ICA evidence sheet:** Choose one component and record its sensor map, temporal behavior and spectrum. State whether the evidence warrants removal or remains inconclusive.
5. **Compare decisions:** Contrast rejecting a blink-contaminated epoch with removing an ICA component. Identify what each operation changes and how it can bias class balance.
6. **Exit ticket:** Why can a cleaner-looking trace produce a worse or misleading BCI? Give one signal-loss mechanism and one confounding mechanism.

## 05_segmentation_quality_control

[Open student notebook](../../notebooks/05_segmentation_quality_control.ipynb)

### Exercise 1 reference implementation

```python
def baseline_center(values,mask):
    return values-values[...,mask].mean(axis=-1,keepdims=True)
```

### Discussion and assessment

Threshold curves should use the same starting epochs and report separate class fractions. Constant baseline subtraction leaves peak-to-peak range unchanged. Drop reasons must separate amplitude limits from boundary loss. Overlapping windows require grouped or temporally separated partitions.

### Requested evidence

1. **Baseline function:** Implement baseline_center for a trial × channel × time array and a Boolean time mask.
2. **Event audit:** Print the annotation dictionary and condition counts before epoching. Explain every event code included and excluded.
3. **Threshold curve:** Try at least four rejection thresholds on copies of the same original epochs. Plot retained fraction by condition. Choose a threshold based on evidence, not the highest test score.
4. **Baseline sensitivity:** Compare no baseline with two plausible intervals. Show the effect on an ERP and explain which claim depends on the baseline.
5. **Drop-log investigation:** Inspect at least one dropped trial or document why none were dropped. Distinguish data-boundary loss from amplitude rejection.
6. **Exit ticket:** Write a short reproducibility record including event mapping, interval, baseline, threshold, retained counts and split grouping.

## 06_motor_imagery_bandpower

[Open student notebook](../../notebooks/06_motor_imagery_bandpower.ipynb)

### Exercise 1 reference implementation

```python
def mean_square(values):
    return np.mean(np.asarray(values)**2,axis=-1)
```

### Discussion and assessment

Integrated PSD and time variance agree only for compatible bandwidth and normalization. Log-power distributions should retain all trials, not just class means. Ablations must reuse grouped folds. Movement and cue-related eye position are plausible confounds requiring independent controls.

### Requested evidence

1. **Power feature:** Implement mean_square on the final axis, returning one number per channel/trial. Explain when it equals variance.
2. **Band-power audit:** For one real epoch, compare integrated PSD with time-domain variance. Account for the filter and the limited integration band.
3. **Channel comparison:** Plot class distributions of C3 and C4 mu features using training data. Label units and overlap; avoid judging separability from means alone.
4. **Ablation:** Compare mu-only, beta-only and combined features using exactly the same grouped splits. Report each run score, not only the mean.
5. **Confound audit:** State two non-neural mechanisms that could distinguish imagery labels and propose a measurement or control for each.
6. **Exit ticket:** Describe exactly which future observations the reported evaluation represents, and which it does not.

## 07_time_frequency_erd

[Open student notebook](../../notebooks/07_time_frequency_erd.ipynb)

### Exercise 1 reference implementation

```python
def relative_db(power,baseline):
    return 10*np.log10(np.asarray(power)/baseline)
```

### Discussion and assessment

Higher fixed cycle counts broaden temporal support and improve frequency selectivity. Baseline changes alter relative ERD without necessarily changing task power. Total power can survive phase variability when evoked power cancels. Data-selected regions require selection within training folds.

### Requested evidence

1. **dB conversion:** Implement relative_db for positive power and baseline arrays using broadcasting.
2. **Resolution experiment:** Repeat the real map with two cycle schedules. Compare temporal smearing and frequency separation using the same color scale.
3. **Baseline sensitivity:** Compare two pre-event baseline intervals. Explain whether differences reflect the task or the reference estimate.
4. **Evoked versus total:** Compute the spectrum or time-frequency representation of an averaged waveform and compare with mean single-trial power. Explain the difference without calling either incorrect.
5. **ERD feature:** Define a pre-specified frequency/time/channel region and extract one ERD feature per trial. State how any data-driven region selection must be confined to training folds.
6. **Exit ticket:** Write a caption that includes channels, frequency range, cycles, baseline, transform and number of trials.

## 08_competition_csp

[Open student notebook](../../notebooks/08_competition_csp.ipynb)

### Exercise 1 reference implementation

```python
def variance_ratio(w,covariance_a,covariance_b):
    return (w@covariance_a@w)/(w@(covariance_a+covariance_b)@w)
```

### Discussion and assessment

CSP must be fitted inside every inner fold. Pattern maps remain sensor-level and have arbitrary sign/scale. Binary variance ratios provide intuition but the four-class implementation is not a binary left-versus-rest shortcut. Baseline comparisons need identical sessions and scoring.

### Requested evidence

1. **Variance-ratio objective:** Implement variance_ratio for a vector and two class covariance matrices.
2. **Component inspection:** Inspect selected patterns and name the reference, channel set and frequency band. Explain why a pattern is not anatomical localization.
3. **Leakage diagram:** Draw the order of fitting CSP, selecting component count and fitting LDA for one inner fold and the final session test.
4. **Baseline comparison:** Compare fixed band-power features with CSP using the same session split. Report balanced accuracy and class-specific errors.
5. **Regularization experiment:** Try a small regularization grid in inner folds only. Report the chosen value and whether the session-level conclusion changes.
6. **Exit ticket:** Explain why a high calibration score can coexist with weak session transfer, mentioning both model complexity and recording changes.

## 09_validation_model_selection

[Open student notebook](../../notebooks/09_validation_model_selection.ipynb)

### Exercise 1 reference implementation

```python
def groups_disjoint(train_groups,test_groups):
    return set(train_groups).isdisjoint(set(test_groups))
```

### Discussion and assessment

Every scaler, CSP fit and hyperparameter decision must be traced to its permitted observations. Selection optimism grows with the number of noisy candidates. Paired run differences are descriptive with only three groups; do not turn folds into independent participant confidence intervals.

### Requested evidence

1. **Group assertion:** Implement groups_disjoint to return whether train and test contain no shared group identifiers.
2. **Split specification:** Write three deployment scenarios and choose the correct grouping for each. Include one chronological scenario.
3. **Pipeline audit:** List all fit calls in the real nested example and state exactly which observations each can see.
4. **Search-size experiment:** Repeat the numerical optimism experiment with 2, 10 and 100 candidates. Plot selection bias against candidate count.
5. **Model comparison:** Compare two pipelines using identical outer splits. Report paired run differences and discuss why three runs cannot support broad population claims.
6. **Exit ticket:** Write a short reviewer response to a result that selected the best frequency band using the final test session.

## 10_ssvep_frequency_cca

[Open student notebook](../../notebooks/10_ssvep_frequency_cca.ipynb)

### Exercise 1 reference implementation

```python
def harmonic_reference(times,frequency,harmonics=2):
    return np.column_stack([wave(2*np.pi*h*frequency*times) for h in range(1,harmonics+1) for wave in [np.sin,np.cos]])
```

### Discussion and assessment

Longer windows generally provide more frequency evidence but also longer delay; observed accuracy need not be monotonic on a small sample. Harmonics must respect Nyquist. No-control performance requires no-control data. Calibration-selected rejection must report coverage and errors, not accepted accuracy alone.

### Requested evidence

1. **Reference builder:** Implement harmonic_reference returning samples × (2 × harmonics), alternating sine and cosine.
2. **Window-length study:** Compare 1, 2 and 4 s windows on the same trials using fixed channels and harmonics. Plot accuracy against observation duration.
3. **Spectral audit:** For a correctly and an incorrectly decoded trial, show target frequencies and background bins. If there are no errors, use the smallest score-margin trials.
4. **Harmonic ablation:** Compare one versus two harmonics without selecting the winner on the final test labels. Explain how you would reserve calibration data.
5. **No-control design:** Propose a dataset and evaluation for false activations during rest. State why the provided two-class benchmark cannot establish this result.
6. **Exit ticket:** Compare the roles of event timing, calibration labels and observation duration in P300, imagery and SSVEP decoding.

## 11_auditory_visual_erp

[Open student notebook](../../notebooks/11_auditory_visual_erp.ipynb)

### Exercise 1 reference implementation

```python
def global_field_power(values):
    return np.std(values,axis=0,ddof=0)
```

### Discussion and assessment

Equal-count comparisons should use reproducible subsampling and acknowledge remaining variability. Window means reduce peak-search flexibility but still require predefinition. Jitter lowers the average peak without reducing each trial amplitude. An attention BCI needs intention labels, not merely delivered-stimulus labels.

### Requested evidence

1. **GFP function:** Implement global_field_power for channel × time input using population standard deviation.
2. **Count matching:** Compare auditory and visual averages using equal trial counts chosen reproducibly. Report what changes and what remains stable.
3. **Window measurement:** Pre-specify a channel and latency window, then calculate single-trial mean amplitudes. Plot distributions instead of only the averages.
4. **Jitter experiment:** Increase simulated latency jitter and plot average peak amplitude. Explain why this does not prove a change in neural response strength.
5. **BCI redesign:** Propose an auditory attention experiment: choices, event markers, target labels, controls, calibration and held-out evaluation.
6. **Exit ticket:** Write one supported conclusion from the real ERP comparison and one tempting conclusion the data do not establish.

## 12_fnirs_motor_paradigm

[Open student notebook](../../notebooks/12_fnirs_motor_paradigm.ipynb)

### Exercise 1 reference implementation

```python
def optical_density_change(intensity,reference):
    return -np.log(np.asarray(intensity)/reference)
```

### Discussion and assessment

Positive intensity is required for the logarithm. Optical-density sign follows attenuation. Scaling the assumed pathlength inversely scales inferred concentration. Coupling quality is not proof of cortical origin. Executed movement, systemic physiology and slow response latency limit BCI claims.

### Requested evidence

1. **Optical-density function:** Implement optical_density_change for strictly positive intensity and reference values.
2. **Channel audit:** Inspect coupling values and optode distances. Explain the threshold and list retained channels rather than silently dropping them.
3. **Conversion sensitivity:** Repeat conversion with another explicitly stated pathlength factor. Compare scale and waveform shape, explaining which changed.
4. **Filter interpretation:** Plot the frequency response of the hemodynamic filter and relate it to task-block duration.
5. **BCI proposal:** Design a trial-level feature and a grouped validation scheme for an fNIRS motor BCI. State the latency cost and the difference between executed and imagined movement.
6. **Exit ticket:** Trace the units from measured intensity through optical density to relative HbO/HbR. Name two physiological confounds.

## 13_neural_networks_autoencoders

[Open student notebook](../../notebooks/13_neural_networks_autoencoders.ipynb)

### Exercise 1 reference implementation

```python
def mean_nll(correct_class_probabilities):
    return -np.log(correct_class_probabilities).mean()
```

### Discussion and assessment

Shape audits should show actual tensors at each layer. Cross-entropy receives logits. Checkpoint selection uses validation only. A baseline must receive the same data split. Autoencoder targets are recorded EEG, so improved MSE under injected noise does not establish biological artifact removal.

### Requested evidence

1. **Cross-entropy function:** Implement mean_nll for a vector of probabilities assigned to the correct class. Use values strictly between zero and one.
2. **Shape audit:** Annotate the real network with input/output dimensions at each layer and verify them with a dummy batch.
3. **Learning curves:** Plot training and validation loss together. Identify a checkpoint choice and distinguish underfitting from overfitting using evidence.
4. **Baseline comparison:** Compare the CNN with a fixed band-power model using the same train/validation/test runs. Do not claim superiority from a single favorable seed.
5. **Autoencoder stress test:** Vary synthetic corruption amplitude and compare reconstruction MSE with an unchanged-input baseline. State what the clean target actually represents.
6. **Exit ticket:** Explain why neither low reconstruction loss nor high training accuracy establishes a useful neural interface.

## 14_online_replay

[Open student notebook](../../notebooks/14_online_replay.ipynb)

### Exercise 1 reference implementation

```python
def trailing_variance(values,window,step):
    return np.array([np.var(values[end-window:end]) for end in range(window,len(values)+1,step)])
```

### Discussion and assessment

Carry the returned SOS state unchanged into the next chunk and use matching initialization for the reference. A centered average uses future data. Trailing windows overlap, so updates are dependent. Hysteresis should be evaluated against known simulated command intervals with both delay and false activations.

### Requested evidence

1. **Trailing feature:** Implement trailing_variance returning one variance per complete trailing window, with the specified step.
2. **Chunk invariant:** Test causal filtering with chunk sizes 1, 31, 128 and a full stream. Assert equality under a common initial state.
3. **Startup inspection:** Plot the beginning of the causal output and discuss a warm-up policy. Do not silently remove startup samples from latency accounting.
4. **Decision policy:** Implement a simple two-threshold hysteresis rule on simulated scores. Report command count, false triggers and delay for a known simulated target interval.
5. **Replay limitations:** List three phenomena absent from array replay and propose a test for each before classroom use with live acquisition.
6. **Exit ticket:** Explain why a centered moving average and a trailing moving average have different online information requirements.

## 15_capstone_reproducible_bci

[Open student notebook](../../notebooks/15_capstone_reproducible_bci.ipynb)

### Exercise 1 reference implementation

```python
def macro_recall(confusion):
    return np.mean(np.diag(confusion)/confusion.sum(axis=1))
```

### Discussion and assessment

Reward a narrow supported claim, complete manifest and prediction integrity over a high score. Comparisons must share partitions and calibration information. Error-driven redesign makes the previous test exploratory. Reproduction should record package and numerical differences instead of hiding them.

### Requested evidence

1. **Balanced accuracy from a matrix:** Implement macro_recall for a confusion matrix whose true classes are rows, assuming each class has at least one observation.
2. **Pre-analysis plan:** Write a one-page plan with question, data, split, primary metric, baseline, one comparison and a stopping rule for model development.
3. **Run the comparison:** Implement the planned alternative using the same partitions. Save configuration and trial predictions; state whether the result supports the hypothesis.
4. **Reproduction exchange:** Give the notebook and manifest to another student. Record what they needed to change and whether their outputs agree within expected numerical tolerance.
5. **Model card:** Write intended use, data scope, performance, failure cases, computational needs and limitations. Include why the result is not yet a deployed BCI.
6. **Exit presentation:** Prepare a five-minute explanation centered on one figure, one comparison and one limitation. Answer which independent dataset or participant group should be tested next.
