# Glossary and equation reference

| Term | Meaning |
|---|---|
| ERP | Event-related potential, obtained by averaging event-locked voltage |
| P300 | A task-related positive ERP deflection, often several hundred milliseconds after an event |
| ERD/ERS | Power decrease/increase relative to a declared baseline |
| PSD | Power spectral density, for EEG in V²/Hz |
| CSP | Supervised spatial filters emphasizing class-dependent variance |
| LDA | Linear discriminant analysis using class means and shared covariance |
| CCA | Linear projections maximizing correlation between two variable sets |
| ICA | A statistical source-separation model based on independence |
| Montage | Mapping from channel names to sensor positions |
| Reference | Electrical reference defining measured voltage differences |
| Epoch | Event-aligned segment of continuous data |
| Leakage | Information from evaluation data influencing fitted choices |
| Group | Unit kept intact across splits: trial, character, run, session or person |
| HbO/HbR | Oxygenated/deoxygenated hemoglobin concentration changes |

Core formulas are derived and used in their notebooks:

- Sampling: $t=n/f_s$ (00).
- ERP: $\bar X=N^{-1}\sum_iX_i$ (01).
- LDA: $w=\Sigma^{-1}(\mu_1-\mu_0)$ (02).
- FIR: $y[n]=\sum_kh[k]x[n-k]$ (03).
- Regression removal: $X_{clean}=X-E(E^TE)^{-1}E^TX$ (04; use a stable least-squares solver in code).
- Baseline: $X'=X-\mathrm{mean}_{t\in B}X(t)$ (05).
- Bandpower: $P=\int_a^b PSD(f)df$ (06).
- ERD/ERS in dB: $10\log_{10}(P/P_B)$ (07).
- Binary CSP: $C_1w=\lambda(C_1+C_2)w$ (08).
- Balanced accuracy: mean class recall (09).
- CCA: maximize correlation of projected EEG and sinusoidal references (10).
- Optical density: $-\ln(I/I_0)$ (12).
- Cross-entropy: $-N^{-1}\sum_i\log p(y_i|X_i)$ (13).

Always state dimensions, units, fitted data and assumptions when using an equation.
