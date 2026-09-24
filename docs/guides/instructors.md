# Instructor guide and syllabus alignment

The supplied ROBT613 syllabus emphasizes physiology, ERPs, MNE loading and segmentation, time-frequency analysis, artifact handling, motor imagery/CSP/LDA, supervised learning, neural networks, CNNs, autoencoders and a final project. This course turns those topics into independent labs. It does not reproduce the syllabus’s dated administrative information or change its academic policies.

| Teaching week | Suggested material | Evidence of learning |
|---|---|---|
| 1 | Handbook signals/paradigms; notebook 00 | Explain the measurement-to-command loop and units |
| 2 | Notebook 01, anatomy/EEG genesis discussion | Inspect real target/non-target ERPs |
| 3 | Notebook 02 | Explain LDA, imbalance and character metadata |
| 4 | Notebook 03 | Plot filter response and quantify delay |
| 5 | Notebook 05 | Build and audit epochs |
| 6 | Notebook 07 | Explain ERD/ERS and time-frequency resolution |
| 7 | Notebook 04 | Present an artifact audit with justified exclusions |
| 8 | Notebooks 06 and 08 | Midterm: bandpower versus CSP on a locked split |
| 9 | Notebook 09 | Nested validation and a reproducibility manifest |
| 10 | Notebook 10 | Compare SSVEP spectral and CCA methods |
| 11 | Notebook 13, CNN section | Learning curves and held-out-run comparison |
| 12 | Notebook 13 autoencoder; choose 11 or 12 | Explain reconstruction versus decoding objectives |
| 13 | Notebook 14 and capstone development | Verify causal filtering and define latency |
| 14 | Notebook 15 | Reproducible final presentation and peer critique |

## Prerequisites and pacing

Students need Python arrays, plotting, basic probability, matrix multiplication and train/test concepts. Teach the equations beside the code; do not require prior familiarity with MNE. Pair each lab with a short paper discussion. Use 15 minutes of framing, 25–40 minutes of worked experiments, 30–45 minutes of real-data inspection, then 30–60 minutes of independent practice. Split a lesson across meetings when students need more time. The notebook is the lecture material: pause before each worked example for a prediction, then ask students to explain the output before proceeding. Dataset downloads should happen before class.

## Assessments

Homework 1: units, reference, event mapping and ERP interpretation. Homework 2: filter tradeoffs, quality audit and feature derivation. Midterm: two motor-imagery baselines under identical session splits. Final: one predeclared paradigm extension with a reproducibility package. The supplied syllabus’s broad weighting (homework 20%, quizzes 20%, midterm 30%, final 30%) can be retained if appropriate; instructors set actual deadlines and policy.

## Discussion prompts and expected reasoning

- Why does majority prediction look good on P300? Non-target prevalence inflates accuracy; balanced accuracy and ranking metrics expose it.
- Why must CSP be fitted in training folds? Its covariance contrast uses class labels.
- Does removing a blink-like ICA component prove neural preservation? No; spatial and temporal evidence plus before/after inspection are needed.
- Does SSVEP frequency decoding prove a usable speller? No; intention, rejection, timing, ergonomics and end-to-end character selection are additional requirements.
- Can an autoencoder learn artifacts? Yes; reconstruction preserves whatever helps minimize its loss.
- Can a zero-phase offline model be called real-time? No; preprocessing and temporal support must be causal and latency measured.

The exercises are formative prompts, not answer keys to an existing graded assignment. Students should explain their own choices and report limitations rather than chase a target score.

## Teach from the notebook itself

The self-study revision places every exercise, hint and worked answer inside its notebook. The sequence is **try → hint → worked solution → interpretation**. Ask students to stop before the solution heading; in a live class, discuss their predictions before running the reference implementation. In independent study, students can complete the same cycle without an instructor or an external answer file.

Each notebook includes a pipeline flowchart, a notation/hand-calculation section, an additional concept figure, four guided numerical practices and six applied/conceptual practices. Applied tasks are interleaved with the real-data stages. Code cells have been split at complete Python statements and logical boundaries; loops stay intact so they remain executable. Comments and adjacent prose explain their scientific role.

The embedded figures are reproducible Matplotlib outputs, not remote images. The flowchart drawing code is tagged as a diagram and may be collapsed while teaching. Students should still be able to explain each box and the quantities carried along its arrows.

Use the [optional answer index](instructor-solutions.md) only for navigation. Assess reasoning and reproducible evidence, rather than a predetermined target accuracy. Suggested rubric: 25% correct method and units, 25% reproducible evidence, 25% interpretation, and 25% awareness of assumptions and limits.

The course excludes EEG source imaging, anatomical localization and forward/inverse models. Sensor topographies for ICA/CSP are retained to interpret preprocessing and decoding, with that distinction stated explicitly.
