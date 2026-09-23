# Colab, local setup and troubleshooting

1. Open a Colab badge in the repository README.
2. Choose a Python CPU runtime. The first cell installs MNE 1.13.2 and MOABB 1.7.2 before imports.
3. Run all cells in order. If Colab requests a restart after upgrading a preloaded numerical dependency, restart once and rerun from the top.
4. Save a copy to your own Drive to retain changes. Dataset cache and files written under the runtime are lost on reset unless explicitly copied out.
5. Download the completed notebook and capstone result files before ending the session.

Every notebook embeds its own setup, loader and processing steps. No repository import, API token, GPU or previous notebook is needed. The first package installation and dataset download require internet. Notebook 13 adds PyTorch; other notebooks do not need it.

## Common problems

| Symptom | What to check |
|---|---|
| Download HTTP error or timeout | Upstream dataset host, network policy and available disk; rerun later without changing labels or provenance |
| Binary NumPy import mismatch | Restart Colab after installation; locally use a fresh environment |
| Missing channel name | Print `raw.ch_names`; EEGBCI names are standardized before applying a montage |
| Empty epochs | Print event mapping and epoch boundaries; do not assume all datasets use T1/T2 |
| All epochs rejected | Check volts versus microvolts, channel types and threshold before relaxing rejection |
| Long ICA runtime or convergence warning | Start with bounded calibration data; inspect rank and component count; do not automatically accept a nonconverged solution |
| CSP singular covariance | Check reference-induced rank, flat channels and regularization |
| Colab out of RAM | Restart; use the default one-subject subset; avoid keeping duplicate full recordings |
| Very high accuracy | Audit split groups, overlapping windows, CSP fitting and cue/artifact confounds |

## Local execution

Install `requirements.txt` and `requirements-dev.txt` in a fresh virtual environment, then use `python scripts/execute_notebooks.py --only 00 06`. For all lessons, omit `--only`; expect large downloads. Use Python 3.11 or later. The execution script installs a temporary kernel specification pointing to the current interpreter, so notebook execution uses the same environment as the command.
