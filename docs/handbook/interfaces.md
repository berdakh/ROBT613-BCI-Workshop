# From decoder to interface

An offline classifier is one component of a BCI. A complete interface acquires synchronized data, estimates signal quality, computes features causally, produces decisions, handles uncertainty and delivers feedback that a person can use.

## Timing

Measure latency from the physical event or user action to the command becoming available. Buffering, filtering and feature accumulation usually dominate a small linear model’s inference time. Updating a two-second window four times per second does not make the underlying evidence only a quarter-second old.

## Rejection and idle state

A forced-choice classifier always chooses something. Real users spend time resting, adjusting posture, looking away or disengaging. Record such states and evaluate false activations per minute. A rejection threshold needs calibration and may trade coverage for error rate. A stop control should not depend exclusively on the same uncertain neural decoder it is stopping.

## Adaptation

Cap placement, impedance, fatigue and attention change across sessions. Monitor feature distributions and performance when labels become available. Adaptation can help but can also reinforce incorrect pseudo-labels. Separate calibration from evaluation and preserve a log of model updates.

## Responsible interpretation

Public research datasets are useful for learning but do not establish clinical effectiveness. Report the acquisition population and protocol. Avoid interpreting a label as an unrestricted mental state. Treat participant information and raw recordings according to their consent and license. For student projects, prerecorded replay is a useful first deployment milestone before any live hardware integration.

Read alongside notebook 14 and the capstone. The release includes causal replay, not a live robot controller or a clinical device.
