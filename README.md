# Brain–Computer Interfaces — a hands-on course

**ROBT613 · MNE-Python · 16 self-contained notebooks · Google Colab or local CPU**

Start with a P300 speller, learn how EEG becomes a feature vector, and build motor-imagery and SSVEP decoders. Extend the workflow to auditory/visual responses, fNIRS, neural networks and causal replay. Each notebook contains the teaching explanations, equations, visualizations, hands-on exercises and worked answers needed to study the method independently.

The teaching sequence draws on [ROBT613](https://github.com/berdakh/ROBT613), the [Brussels MNE workshop](https://github.com/berdakh/mne-workshops/tree/master/2017_03_Brussels), and the sensor-level sections of the [Brown MNE workshop](https://github.com/berdakh/mne-workshop-brown), with original BCI lessons aligned to the supplied syllabus. This is a modular course, not a four-day schedule. Plan roughly **35–45 guided hours, 15–25 hours of independent practice, and a 6–12 hour capstone**, adjusted for student background and dataset downloads.

## Self-study notebooks

The **notebook is the complete lesson and workbook**. Students do not need an external answer guide. Every lesson follows this sequence:

1. A visual map of the full pipeline and a symbol guide.
2. Equations derived through a numerical calculation that can be done on paper.
3. An annotated concept figure, followed by a question and worked interpretation.
4. Four short guided practices: predict, calculate, run and explain.
5. A real-data analysis broken into small code steps with statement-level explanations.
6. Six practice tasks placed beside the relevant method, each with a hint and a worked answer in the notebook. Computational tasks include executable solutions and numerical checks where appropriate.
7. A worked exit response, troubleshooting advice and a self-assessment checklist.

There are **16 pipeline flowcharts and 16 new concept figures**, in addition to the real-data plots and worked-example figures. Figure-generating code and saved outputs are embedded in each notebook; no external image folder or diagram renderer is needed. Blank student-attempt cells are intentional—the complete worked solutions follow them, so Run all still executes the lesson.

Use the notebooks one cell at a time in class, pausing at each **Try it** question. For independent study, attempt the calculation before continuing to its solution. The [instructor guide](docs/guides/instructors.md) offers pacing suggestions but is not required to solve the exercises.

**Scope:** sensor-level EEG and an optional fNIRS lesson. EEG source imaging, forward/inverse modeling and anatomical source localization are not included. CSP and ICA maps describe sensor patterns only.

## Start here

Open notebook **00** for the environment and MNE foundations, then **01–02** for P300. Every notebook runs independently; no repository clone or hidden helper module is needed in Colab. Select a **Python CPU runtime**, run the setup cell first, then work through the lesson one cell at a time. Run all is useful for a fresh-kernel reproducibility check. Internet is needed for first-time package and dataset downloads.

**Dataset distinction:** MNE provides native EEGBCI, SSVEP, auditory/visual sample and fNIRS fetchers. The competition dataset is **BCI Competition IV 2a (BNCI2014-001)**, loaded through **MOABB into MNE**. BNCI2014-009 is a P300 speller dataset, not a competition speller dataset. Binary P300 target detection is evaluated on real data; row/column character aggregation is a separately labeled simulation because the paradigm loader lacks the necessary character metadata.

## Learning outcomes

- Inspect signal units, sampling, montage, events and experimental provenance.
- Explain and apply filtering, referencing, regression, ICA, epoching and rejection.
- Extract temporal ERP, spectral, time-frequency and spatial CSP features.
- Evaluate LDA, logistic regression, CCA and a small CNN without train/test leakage.
- Distinguish trial, run, session and subject generalization.
- Design a reproducible experiment and explain limits of offline BCI performance.

## Notebooks

| # | Lesson | Guided session (add practice) | Colab |
|---|---|---|---|
| 00 | [Your first EEG in MNE](notebooks/00_start_here.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/00_start_here.ipynb) |
| 01 | [P300 speller: from flashes to ERPs](notebooks/01_p300_signal_to_epochs.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/01_p300_signal_to_epochs.ipynb) |
| 02 | [P300 classification and character selection](notebooks/02_p300_classification_speller.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/02_p300_classification_speller.ipynb) |
| 03 | [Filtering, sampling and spectral inspection](notebooks/03_filtering_sampling.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/03_filtering_sampling.ipynb) |
| 04 | [Artifacts, referencing and noise cancellation](notebooks/04_artifacts_noise_cancellation.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/04_artifacts_noise_cancellation.ipynb) |
| 05 | [Segmentation, baseline and quality control](notebooks/05_segmentation_quality_control.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/05_segmentation_quality_control.ipynb) |
| 06 | [Motor imagery: a first decoder](notebooks/06_motor_imagery_bandpower.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/06_motor_imagery_bandpower.ipynb) |
| 07 | [Time–frequency analysis and ERD/ERS](notebooks/07_time_frequency_erd.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/07_time_frequency_erd.ipynb) |
| 08 | [BCI Competition IV 2a: CSP and LDA](notebooks/08_competition_csp.ipynb) | 120–150 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/08_competition_csp.ipynb) |
| 09 | [Honest validation and model selection](notebooks/09_validation_model_selection.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/09_validation_model_selection.ipynb) |
| 10 | [SSVEP: spectral peaks and canonical correlation](notebooks/10_ssvep_frequency_cca.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/10_ssvep_frequency_cca.ipynb) |
| 11 | [Auditory and visual evoked responses](notebooks/11_auditory_visual_erp.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/11_auditory_visual_erp.ipynb) |
| 12 | [Beyond EEG: fNIRS motor responses](notebooks/12_fnirs_motor_paradigm.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/12_fnirs_motor_paradigm.ipynb) |
| 13 | [Neural networks and autoencoders for EEG](notebooks/13_neural_networks_autoencoders.ipynb) | 120–180 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/13_neural_networks_autoencoders.ipynb) |
| 14 | [From offline analysis to causal replay](notebooks/14_online_replay.ipynb) | 90–120 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/14_online_replay.ipynb) |
| 15 | [Capstone: a reproducible BCI experiment](notebooks/15_capstone_reproducible_bci.ipynb) | 6–12 hours | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613-BCI-Workshop/blob/main/notebooks/15_capstone_reproducible_bci.ipynb) |

**Essential pathway:** 00 → 01 → 02 → 03 → 05 → 06 → 08 → 09 → 15. Add 04 and 07 for a full signal-processing foundation. Lessons 10–14 extend paradigms and deployment concepts.

## Local setup

Use **Python 3.11–3.13** in a fresh environment. The release was exercised with Python 3.13; see the validation record for exact coverage.

```bash
git clone https://github.com/berdakh/ROBT613-BCI-Workshop.git
cd ROBT613-BCI-Workshop
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts/check_env.py
jupyter lab notebooks/00_start_here.ipynb
```

No GPU, API key or paid account is required. Notebook 13 installs PyTorch if absent. Download time is excluded from lesson estimates. Allow several GB of cache space; the optional auditory/visual sample archive is particularly large.

## Handbook and instructor resources

- [Signals, anatomy and measurement](docs/handbook/signals.md)
- [Paradigms and experimental design](docs/handbook/paradigms.md)
- [Signal processing and features](docs/handbook/methods.md)
- [Learning and generalization](docs/handbook/learning.md)
- [From decoder to interface](docs/handbook/interfaces.md)
- [14-week syllabus map and assessments](docs/guides/instructors.md)
- [Optional instructor answer index](docs/guides/instructor-solutions.md)
- [Dataset provenance and download guide](docs/guides/datasets.md)
- [Colab and troubleshooting](docs/guides/colab.md)
- [Evaluation checklist](docs/guides/evaluation.md)
- [Glossary and equation reference](docs/guides/glossary.md)
- [Validation record](docs/guides/validation.md)

## Repository structure

```
notebooks/          student notebooks (standalone, with saved outputs when validated)
notebook_src/       readable percent-format sources
scripts/            rebuild, static validation, execution and environment check
src/bci_workshop/   small reference numerical utilities, not required by notebooks
tests/              numerical and split-integrity checks
docs/handbook/      conceptual background
docs/guides/        dataset, instructor and reproducibility guides
```

Edit `notebook_src/*.py`, run `python scripts/build_notebooks.py`, then validate. Rebuilding intentionally clears outputs; execute notebooks to regenerate them. CI checks source/notebook agreement, valid notebook structure and numerical tests. Network-heavy notebook execution is an explicit local step so a dataset outage does not masquerade as a code regression.

## Reuse and attribution

Original workshop code is MIT licensed; original prose and teaching material are CC BY 4.0. Dataset and third-party software licenses remain separate. Raw EEG is downloaded from its original host and is **not redistributed** here. Cite the dataset publications and MNE/MOABB when reporting results; see the dataset guide. The supplied syllabus informed topic selection; its PDF and personal administrative details are not included.
