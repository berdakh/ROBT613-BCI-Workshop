# Validation record

Validated on 2026-09-23 in a local CPU Python 3.13 environment with MNE 1.13.2 and MOABB 1.7.2. All 16 notebooks were executed top-to-bottom against their declared data, with saved outputs. Real datasets were downloaded; controlled demonstrations remained explicitly labeled. Notebook 13 used PyTorch 2.14.0+cpu. The numerical integrity suite has 7 passing tests.

The Colab setup cells and public notebook links are provided; **execution inside Google Colab itself was not tested**. Colab may require a runtime restart after installing numerical dependencies. Upstream host availability is outside the course’s control.

Saved figures were visually reviewed for labels, units, clipping and readability. The CSP pattern plot uses the public topomap API to avoid a multiclass plotting failure in MNE’s legacy CSP plotting convenience method. The competition notebook was rerun after its first download was incomplete. A rejection-threshold example was revised after inspection showed its original threshold removed every trial.

## Reproduce

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_notebooks.py
python -m pytest -q
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
