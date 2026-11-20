# Josh Talks: Speaker Diarization Evaluation

FDE take-home assignment for Josh Talks / Voice Arena, November 2026. This repository contains the submitted reports, a small AMI evaluation, and a dashboard for the saved measurements.

Speaker diarization answers **who spoke when**. The evaluation explores how microphone conditions, speaker-count information, buffering and scoring policies affect reported quality.

## Completed work

- Three AMI test meetings: ES2004a, IS1009a and TS3003a, with four reference speakers each.
- Two capture conditions per meeting: close-talk headset mix and far-field array channel 1, giving six recordings.
- Seven configurations: pyannote community-1 with automatic/known speaker count, two NVIDIA Sortformer v2 buffering configurations, and three simple baselines.
- Four scoring protocols: with/without boundary tolerance and with/without overlapping speech.
- Per-recording CSVs, environment records, a paired-recording bootstrap analysis and a static results dashboard.

The proposed production benchmark covers approximately **60 hours and 19 systems**. That corpus collection and full system rollout were **not completed** in this pilot. The seven configurations are not seven independently trained model families.

## Assignment files

- [Submitted bundle](reports/Submission.pdf)
- [Main methodology report](reports/Main.pdf)
- [One-page report](reports/OnePager.pdf)
- [Leaderboard sketch](reports/LeaderboardSketch.pdf)
- [Walkthrough script](docs/walkthrough-script.txt)
- [Recorded walkthrough](https://www.loom.com/share/963301fb66b74751a168a7d0a4e1f9a4)

The bundle is the saved submitted version; the individual Main report is the original standalone report. The bundle includes the later one-pager and sketch.

## Results

Mean recording-level DER, strict protocol: zero collar, overlapping speech included. Lower is better.

| Configuration | Mean DER |
|---|---:|
| pyannote community-1, automatic count | 21.4% |
| pyannote community-1, known count | 24.9% |
| Sortformer v2, longer buffer | 31.4% |
| Sortformer v2, shorter buffer | 38.1% |
| S3: embedding/clustering baseline, known count | 46.7% |
| S2: embedding/clustering baseline, automatic count | 49.9% |
| S1: Silero VAD, one speaker | 55.9% |

The same automatic-count pyannote predictions score **11.4%** when a ±250 ms boundary tolerance is introduced and overlap is excluded. This is a scoring-protocol change, not a model improvement. In these scripts, pyannote's `collar=0.5` specifies the total width corresponding to ±250 ms.

## Browse the dashboard

Open [dashboard/public/index.html](dashboard/public/index.html) locally, or use the [deployed pilot](https://diarization-leaderboard.cf-hub.workers.dev/).

The dashboard filters saved results and performs exploratory comparisons in the browser. It has no inference backend. The separate [prototype](dashboard/public/prototype.html) contains **illustrative** production-leaderboard values; the main pilot page uses actual AMI results.

## Repository layout

```text
reports/                  Submitted assignment PDFs
scripts/run_baselines.py  Original Kaggle baseline runner, with development tuning
scripts/run_pyannote.py   Original pretrained pyannote runner
scripts/run_sortformer.py Original pretrained NVIDIA runner
scripts/bootstrap.py     Portable wrapper around the original bootstrap analysis
scripts/original_bootstrap.py  Original analysis, retained for provenance
scripts/verify_results.py Lightweight checks of saved results and copied artifacts
results/                  Saved CSVs and derived reference-duration totals
environment/              Original recorded model/runtime versions
dashboard/                Static interface, source data/template and hosting config
PROVENANCE.json           SHA-256 hashes and original relative paths
```

## Verify saved measurements

Python 3.12+ can run the lightweight verification without installing speech models:

```bash
python scripts/verify_results.py
```

For the exploratory bootstrap:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/bootstrap.py
```

The portable script reads the saved CSVs and `results/reference_totals.json`, then writes `results/bootstrap_recomputed.csv`. Those totals were derived by summing the original AMI reference segment durations. The original script remains unchanged in `scripts/original_bootstrap.py` and expects CSVs plus `ami/*.rttm` in its working directory.

## Rerun model inference

The three inference scripts are the original Kaggle scripts. Use separate fresh working directories or isolated Kaggle notebooks. They install dependencies, download public AMI audio/reference annotations and save results to their current working directory. The scripts remove downloaded audio directories at the end, so run them in dedicated scratch directories.

```bash
mkdir baseline-work
cd baseline-work
python ../scripts/run_baselines.py
```

pyannote needs the model's gated-access terms accepted and an `HF_TOKEN` **Kaggle secret**. Its original runner uses `kaggle_secrets.UserSecretsClient`; running that script unchanged outside Kaggle will not provide the token. Do not hard-code a token.

The pyannote and Sortformer runs used GPU; the saved baseline environment used CPU. Original versions are recorded in `environment/`. Installation commands are not fully locked, so a future reinstall is not guaranteed to reproduce identical software versions or predictions. See [reproduction notes](docs/REPRODUCTION.md).

## Interpretation limits

- Six recordings represent only **three independent meetings**. The original bootstrap resamples recording rows rather than meeting clusters, so its intervals are exploratory.
- Tables use arithmetic means of recording percentages; bootstrap comparisons pool reconstructed error durations. These are different aggregations.
- The original analysis reconstructs errors using raw RTTM duration rather than storing exact scorer denominators for every protocol.
- JER is computed once at zero collar and repeated across the four DER-protocol rows.
- Dashboard groups use unadjusted anchor comparisons. Holm correction is proposed in the methodology, not implemented in the pilot.
- Sortformer buffering configurations were run through whole-file inference. They do not establish measured production streaming latency.
- No universal best model is established by this small sample.

## Data and attribution

AMI audio and reference RTTMs, model weights, access tokens and the separate Growth-task data pack are not bundled. The runners identify the upstream download locations. Models and data retain their respective upstream terms.

References: [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/), [AMI diarization setup](https://github.com/pyannote/AMI-diarization-setup), [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [NVIDIA Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), [pyannote metrics](https://pyannote.github.io/pyannote-metrics/), [dscore](https://github.com/nryant/dscore).

Author: Nitesh Kumar Yadav.
