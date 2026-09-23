# Signals, anatomy and measurement

EEG, MEG and fNIRS describe different aspects of brain activity. EEG measures electrical potential differences at electrodes; MEG measures magnetic fields produced by current flow; fNIRS estimates relative hemoglobin changes from optical absorption. None is a direct readout of a person’s thoughts.

## From synapse to sensor

Excitatory and inhibitory postsynaptic currents create spatially organized current sources and sinks. Large populations of aligned cortical pyramidal cells can generate fields detectable at the scalp. Individual action potentials are usually not resolved by conventional scalp EEG. Conductivity and source geometry influence what reaches each sensor: a broad scalp pattern can arise from a compact source, and one electrode receives mixtures from many sources.

A linear forward description is $x(t)=Ls(t)+\epsilon(t)$, where $s$ is source activity, $L$ the lead-field matrix and $\epsilon$ noise. Recovering $s$ from a limited number of sensors is ill-posed; classification can work without uniquely solving source localization. A discriminative sensor weight is therefore not a brain map.

## Reference and montage

An EEG value is a voltage difference. Changing reference changes every waveform and covariance. Common average reference multiplies the sensor vector by $I-\mathbf{1}\mathbf{1}^T/C$, whose rank is at most $C-1$. Linked mastoids, Cz and average reference are different measurements, not interchangeable labels. Keep reference consistent between calibration and evaluation.

A montage assigns sensor coordinates. A cap layout is an approximation to an individual head; assigning standard positions does not create individualized anatomy. Channel names and types must be correct before plotting topographies or interpolating channels.

## Sampling and units

MNE uses volts for EEG. Plot microvolts by multiplying by $10^6$ only at presentation time. Sampling frequency links sample index and time; anti-alias filtering is needed before reducing the rate. Resampling annotations/events requires care so event timing remains aligned. Check a known event against the raw trace before building hundreds of epochs.

## Physiological interpretation

Mu/beta changes, P300-like deflections and frequency-tagged responses are useful task-dependent signals. Their presence, absence and latency vary across people and sessions. Artifacts can carry information about a task without being the neural signal of interest. A useful decoder must be evaluated for the intended user population and experimental setting, not only on a convenient recording.

Read alongside notebooks 00, 01 and 06. [MNE overview](https://mne.tools/stable/auto_tutorials/intro/10_overview.html).
