# Validation record

Validated on 2026-09-25 in a local CPU Python 3.13 environment with MNE 1.13.2 and MOABB 1.7.2. All 17 academic teaching notebooks were executed top-to-bottom against their declared datasets, including the new inline worked solutions, with saved outputs. Notebook 13 used PyTorch 2.14.0+cpu. The numerical integrity suite has 7 passing tests.

The self-study revision contains 64 guided practices and 96 applied/conceptual practices with worked answers inside the notebooks. The dataset-free checker verifies the embedded introductory calculations and all 16 Practice 1 reference functions. Full notebook execution verifies the computational applied solutions as well. Conceptual answers provide reasoning, not automatically graded correctness.

Student-attempt cells are intentionally blank or contain a function template returning None. The complete reference implementation follows each computational task, so top-to-bottom execution continues through its worked solution and checks. No separate answer guide is required.

The 16 new flowcharts, 16 new concept figures and five additional applied-exercise plots were visually inspected. The P300 score-distribution legend was moved outside the data area, and the imagery feature histograms use shared bin edges for a comparable overlay.

The Colab setup cells and public notebook links are provided; **execution inside Google Colab itself was not tested**. Colab may require a runtime restart after installing numerical dependencies. Upstream host availability is outside the course’s control.

New worked-example figures and the saved real-data figures were visually reviewed for labels, units, clipping and readability. The CSP pattern plot uses the public topomap API to avoid a multiclass plotting failure in MNE’s legacy CSP plotting convenience method. The competition notebook was rerun after its first download was incomplete. A rejection-threshold example was revised after inspection showed its original threshold removed every trial.

## Academic revision verification

The September 25 revision adds academic paradigm reviews, acquisition tables, mathematical definitions, references and a native MNE recorded-signal display to every existing lesson. The new MNE foundations tutorial covers software architecture and object semantics with four numerical/conceptual exercises and a synthesis exercise. The complete collection contains 88 saved figures. All 20 newly added figure outputs were inspected for readability, axes and modality-appropriate interpretation. The introductory continuous EEG display was rescaled after review.

The auditory/visual lesson selects its numbered EEG channel from metadata rather than assuming a standard channel name. The fNIRS display is explicitly identified as haemodynamic. Competition plots explain their trial-relative time origin, and pooled P300 plots are distinguished from target-specific responses. All 17 notebooks executed successfully; only the corrected notebooks were rerun after the complete run.

## Reproduce

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_notebooks.py
python -m pytest -q
python scripts/check_exercise_answers.py
python scripts/execute_notebooks.py
```

The execution script downloads real data and can take several minutes per new dataset. It writes `execution_report.json`, fails if any notebook fails, and preserves outputs only after a successful complete run. Rebuilding notebooks clears outputs intentionally.

| Notebook | Execution |
|---|---|
| 00_start_here.ipynb | passed |
| 00a_mne_python_foundations.ipynb | passed |
| 01_p300_signal_to_epochs.ipynb | passed |
| 02_p300_classification_speller.ipynb | passed |
| 03_filtering_sampling.ipynb | passed |
| 04_artifacts_noise_cancellation.ipynb | passed |
| 05_segmentation_quality_control.ipynb | passed |
| 06_motor_imagery_bandpower.ipynb | passed |
| 07_time_frequency_erd.ipynb | passed |
| 08_competition_csp.ipynb | passed |
| 09_validation_model_selection.ipynb | passed |
| 10_ssvep_frequency_cca.ipynb | passed |
| 11_auditory_visual_erp.ipynb | passed |
| 12_fnirs_motor_paradigm.ipynb | passed |
| 13_neural_networks_autoencoders.ipynb | passed |
| 14_online_replay.ipynb | passed |
| 15_capstone_reproducible_bci.ipynb | passed |
| 16_pytorch_tensors_first_network.ipynb | passed |
| 17_pytorch_training_cnn.ipynb | passed |
| 18_pytorch_autoencoder.ipynb | passed |

## PyTorch track (notebooks 16–18)

Added on 2026-11-09. The authoring environment could not reach PhysioNet, so the lessons were first run only on a synthetic stand-in that checked shapes, training loops, checkpoint save/load and figures.

**Real-data execution, 2026-11-10.** All three notebooks were executed top to bottom with `python scripts/execute_notebooks.py --only 16 17 18` on the real PhysioNet EEGBCI recordings (subjects 1–5, runs 4/8/12; 75 trials per run after epoching), in a CPU-only Linux environment (4-core Intel Xeon @ 2.10 GHz, no GPU) with Python 3.13, PyTorch 2.15.0, MNE 1.13.2, NumPy 2.5.3 and scikit-learn 1.9.1. Each notebook ran in under 35 s once the data were cached, and a second run reproduced the same scores. The 18 saved figures (4, 6 and 8) were visually inspected for labels, units and readability.

The interpretive text was written before real scores existed and was revised against the outputs:

- Run 4 has about 75 trials, not about 150 as first stated; trial counts in notebooks 16 and 17 were corrected.
- Notebook 16: the pooled training-run spectra show only small class differences at C3/C4, and the text now says so.
- Notebook 17: the band-power network's validation loss was still falling at the epoch limit, so its kept checkpoint is the last epoch; the text now covers that case. The network scored higher than shrinkage LDA on run 8, and the text now notes that this comparison is biased in the network's favour because run 8 chose its checkpoint. The CNN memorized run 4 (training loss near zero, validation loss near ln 2, validation accuracy near chance), and its interpretation now describes memorization instead of “noisier curves”. On run 12 no model exceeded the 95 % chance band; the text states that this is a common and legitimate outcome for pooled participants with few trials.
- Notebook 18: validation MSE lies below training MSE because the training value is averaged in training mode with dropout active; the interpretation now explains this. The autoencoder reconstructs only slightly better than a four-component spatial PCA of the same size, and the text now says what a small margin means. In the probe, 16 spatial PCA components decoded at or below chance, the autoencoder code above them, and all 64 channels best; the interpretation now uses the PCA row to illustrate that high-variance directions need not carry task information. The reconstructed spectrum lies below the recording at all frequencies with a wider gap in the beta band, and the figure text now says so.

Exact scores depend on library versions and hardware and are read from the saved outputs, not hard-coded in the text.
