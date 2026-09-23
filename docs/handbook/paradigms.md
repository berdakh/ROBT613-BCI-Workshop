# Paradigms and experimental design

## P300 and selective attention

A P300 interface asks a user to attend a rare relevant event among frequent alternatives. In a row/column speller, every character belongs to one row and one column. Flash-level target detection supplies evidence; a separate decoder maps that evidence to a symbol. Repetitions trade speed for reliability. Character identity and repetition grouping must remain intact during evaluation.

## Motor imagery

Motor imagery uses internally generated sensorimotor changes and can reduce dependence on external flashes. The task requires clear instructions, calibration and meaningful rest conditions. Executed movement, imagined movement and observed movement are distinct paradigms. A classifier trained only on left/right imagery has no evidence about what to do when the user is idle.

## SSVEP

Periodic visual stimulation creates responses near a known frequency and harmonics. Sinusoidal reference methods can decode without labeled calibration, but channel, harmonic and threshold choices still require honest evaluation. Window duration controls frequency resolution and latency. Display refresh rate constrains stimulus timing. Offline analysis of an existing recording does not require generating flicker; any live experiment needs an appropriate participant protocol.

## Other paradigms

Auditory selective-attention paradigms can support users who cannot rely on vision. Somatosensory/tactile paradigms shift stimulation modality. Error-related potentials are aligned to perceived errors or feedback and can support correction rather than primary selection. Passive BCIs estimate a state such as workload; they need valid external labels and must avoid confusing task condition with the desired psychological construct. fNIRS introduces slower hemodynamic dynamics and different confounds.

The executable extensions in this release are auditory/visual evoked responses and fNIRS finger tapping. ErrP, tactile and passive BCI are design extensions, not claimed implemented benchmarks. The course deliberately distinguishes a measured sensory response from intentional online control.

## Design before analysis

Specify the intended command, participant population, stimuli, timing, response window, rest state, feedback, electrode layout and ground truth. Counterbalance conditions and avoid assigning one class to an entire recording day unless day transfer is the explicit question. Store subject/session/run/trial IDs, stimulus onset, intended target and feedback onset. Test synchronization using an external timing measurement when conducting acquisition.

Read alongside notebooks 01–02, 10–12 and 14. [MNE dataset catalog](https://mne.tools/stable/documentation/datasets.html).
