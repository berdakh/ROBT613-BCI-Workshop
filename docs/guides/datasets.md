# Dataset provenance and download planning

| Dataset | Access route | Role | Interpretation boundary |
|---|---|---|---|
| BNCI2014-009 | `moabb.datasets.BNCI2014_009`, `P300` | P300 target/non-target detection, subject 1, 3 sessions | Not a BCI Competition dataset; binary epochs lack character identities |
| EEG Motor Movement/Imagery | `mne.datasets.eegbci.load_data` | PhysioNet subject 1, runs 4/8/12 | Imagined left/right fist; not competition data |
| BCI Competition IV 2a | `moabb.datasets.BNCI2014_001`, `MotorImagery` | Four-class imagery, subject 1, 2 sessions | Course cross-session reanalysis, not an official leaderboard result |
| SSVEP frequency tagging | `mne.datasets.ssvep.data_path` | Participant 02, 12/15 Hz | Frequency-tagging experiment, not a full online speller |
| MNE sample | `mne.datasets.sample.data_path` | Auditory/visual EEG | Sensory responses, not intentional command selection |
| fNIRS motor | `mne.datasets.fnirs_motor.data_path` | Finger tapping/control | Executed movement and hemodynamics, not EEG imagery |

## Before class

Run the relevant notebook once on the classroom network. Fetchers cache files in `mne_data/` (or the `BCI_DATA` environment directory). Colab runtimes are ephemeral; persist the cache in your own Drive only if desired. Do not commit the cache. Small subject selections limit processing but some upstream fetchers download a whole archive. Plan multiple GB for the full course and extra disk for archive extraction; no fixed download size is promised because hosts and packaging change.

If a host is unavailable, retain the error and retry later, or use an authorized local copy through the same documented loader. Do not substitute synthetic data and label the result real. The native EEGBCI lessons remain an independent pathway when a BNCI server is unavailable.

## Sources and citation starting points

- [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/): Schalk et al., BCI2000, IEEE TBME (2004). Check the dataset’s current license and cite PhysioNet as requested there.
- [Competition IV](https://www.bbci.de/competition/iv/): Tangermann et al., Review of the BCI Competition IV, Frontiers in Neuroscience (2012), DOI 10.3389/fnins.2012.00055. Check dataset-specific terms.
- [BNCI2014-009](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_009.html): follow the linked original dataset and publication citation; this loader includes grid-speller data. The catalog identifies a noncommercial/no-derivatives dataset license; it does not inherit this repository’s MIT/CC BY licenses.
- [MNE dataset catalog](https://mne.tools/stable/documentation/datasets.html): source and license information for native example datasets.
- [MNE citation guidance](https://mne.tools/stable/documentation/cite.html).
- [MOABB](https://moabb.neurotechx.com/docs/index.html): cite the framework and the original dataset, not only a downloader.

The notebooks intentionally do not publish individual clinical inferences. They demonstrate processing of already-public research data. Preserve original channel units and acquisition metadata in derivative research.
