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
