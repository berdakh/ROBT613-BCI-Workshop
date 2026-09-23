# Signal processing and feature design

Start from the task’s time scale. P300 preserves slow transient structure; motor imagery emphasizes oscillatory power; SSVEP preserves narrowband frequency structure; fNIRS evolves over seconds. A universal preprocessing recipe can erase the signal of interest.

## Filtering

Specify passband, transition bands, attenuation, phase and temporal support. A cutoff alone is incomplete. Inspect both frequency and impulse response. Offline zero-phase processing can smear post-event activity backward in time. Causal filters preserve online feasibility but add phase distortion or delay. Filter continuous runs separately and avoid sharing samples across evaluation boundaries.

## Artifacts

First detect: channel variance, flatness, saturation, spectra, spatial discontinuity and association with EOG/ECG. Then choose a response: mark bad spans, reject epochs, interpolate a small number of bad sensors, regress a reference, or remove reviewed ICA components. Every correction trades information against contamination. Compare before and after with the same axes and report how many channels, components and trials changed.

## Segmentation

An epoch interval encodes a hypothesis about when the relevant information occurs. Baseline subtraction is a per-trial offset correction, not a substitute for drift handling. Preserve original event sample indices and the selection mapping after rejected trials. For overlapping windows, trial identity is the minimum grouping unit.

## Features

Temporal means summarize an ERP without requiring a fragile peak estimate. Welch bandpower averages spectral estimates to reduce variance. Morlet wavelets trade time resolution for frequency resolution. CSP learns class-discriminative spatial variance directions. CCA compares multichannel activity with known periodic references. Each representation discards information; explain why that discarded information is not needed for the task.

A feature vector is not inherently interpretable. Log power has a clear measurement definition, but a classifier coefficient also depends on scaling and correlation. CSP patterns are more suitable than filters for sensor-space interpretation, yet neither uniquely localizes neural sources.

Read alongside notebooks 03–08 and 10. [MNE preprocessing](https://mne.tools/stable/auto_tutorials/preprocessing/index.html).
