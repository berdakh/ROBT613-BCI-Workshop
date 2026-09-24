# Validation record

Validated on 2026-09-24 in a local CPU Python 3.13 environment with MNE 1.13.2 and MOABB 1.7.2. All 16 self-study notebooks were executed top-to-bottom against their declared datasets, including the new inline worked solutions, with saved outputs. Notebook 13 used PyTorch 2.14.0+cpu. The numerical integrity suite has 7 passing tests.

The self-study revision contains 64 guided practices and 96 applied/conceptual practices with worked answers inside the notebooks. The dataset-free checker verifies the embedded introductory calculations and all 16 Practice 1 reference functions. Full notebook execution verifies the computational applied solutions as well. Conceptual answers provide reasoning, not automatically graded correctness.

Student-attempt cells are intentionally blank or contain a function template returning None. The complete reference implementation follows each computational task, so top-to-bottom execution continues through its worked solution and checks. No separate answer guide is required.

The 16 new flowcharts, 16 new concept figures and five additional applied-exercise plots were visually inspected. The P300 score-distribution legend was moved outside the data area, and the imagery feature histograms use shared bin edges for a comparable overlay.

The Colab setup cells and public notebook links are provided; **execution inside Google Colab itself was not tested**. Colab may require a runtime restart after installing numerical dependencies. Upstream host availability is outside the course’s control.

New worked-example figures and the saved real-data figures were visually reviewed for labels, units, clipping and readability. The CSP pattern plot uses the public topomap API to avoid a multiclass plotting failure in MNE’s legacy CSP plotting convenience method. The competition notebook was rerun after its first download was incomplete. A rejection-threshold example was revised after inspection showed its original threshold removed every trial.

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
