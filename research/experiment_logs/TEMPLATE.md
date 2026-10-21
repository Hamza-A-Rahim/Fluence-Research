# EXP-NNN — <short title>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Status | planned / running / complete / abandoned |
| Git commit | `<hash>` (clean working tree? yes/no) |
| Author | |

## Objective
What question does this experiment answer?

## Hypothesis
What do we expect, and what result would show we were wrong? Write this **before** running.

## Data
- Dataset(s) and version (provider repo commit, download date):
- Audio retrieved: N of M episodes (x %); class balance vs published:
- Partition: (e.g. SEP-28k-E) — achieved train/dev/test clips, speakers, positive rates:
- Label threshold (min votes) and interjection handling (D-005):
- Clips excluded and why:

## Preprocessing and features
Sample rate, clip length, feature config (link to `FeatureConfig` values), normalisation.

## Model
Architecture, parameters, pretrained weights (name + version) if any.

## Training configuration
Seeds, optimiser, epochs/early stopping, class weighting, hyper-parameter search (on dev / CV only).

## Evaluation
Metrics per `research/methodology/evaluation_protocol.md`. Include majority and random baselines.

## Results
Tables and figures (store generated files in `outputs/`, commit only small summary tables/figures).
Report confidence intervals. Report negative results honestly.

## Environment
Python and library versions (`pip freeze` excerpt), hardware, run time.

## Limitations

## Conclusions and next steps
