# M1 — Listening check

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Listener | The author (not a clinician) |
| Clips | `scripts/listen_examples.py --per-type 4 --seed 0`: 4 per group (clear-cut examples of each event, fluent, disputed, short-episode check) |

## Purpose
Check by ear that the downloaded clips and their labels make sense before relying on model results, with particular focus on the surprising EXP-002 finding that prolongations are the easiest event to detect (AUC 0.74) despite the lowest annotator agreement (κ 0.11).

## Notes (listener's judgement against the published labels)

| Group | Clip | Published votes | Listener heard | Agrees? |
|---|---|---|---|---|
| Prolongation | WomenWhoStutter_62_130 | Prolongation 3/3 | No prolongation | **No** |
| Prolongation | other 3 (WWS_0_275, MSL_32_54, StutterTalk_37_72) | Prolongation 3/3 | Not flagged | Yes (implicit) |
| Block | all 4 (WWS_8_9, WWS_81_22, WWS_81_75, MSL_7_182) | Block 3/3 | Sound right | Yes |
| Disputed | MyStutteringLife_35_10 | Prolongation 1/3, NoStutteredWords 2/3 | **Prolongation** | Sides with the minority |
| Short-episode check | HeStutters_2_38 (latest kept clip of an episode that downloaded 0.30 s short) | Prolongation 1, Block 1, NaturalPause 1 | Abrupt start; can't tell if it continues a sentence | **Inconclusive** |

Other groups were not flagged.

## Observations
1. **Prolongation labels are noisy in both directions**: a unanimous (3/3) prolongation that the listener does not hear, and a 1/3 clip (so labelled *not* prolongation at ≥2 votes) that the listener does hear. With 5 prolongation-related clips this is anecdotal, but it is consistent with κ = 0.11. Implication: EXP-002's prolongation AUC is measured against noisy targets, so it could be under- or over-stated. The finding should be reported cautiously.
2. **Block labels sounded correct** (4/4). The low block AUC (0.60) therefore looks like a limitation of the features, not of the labels.
3. **Alignment check inconclusive.** Abrupt starts are expected in 3-second windows. Given the maximum shortfall of 0.46 s, misalignment cannot be large. An objective check would need the original clip audio for comparison, which isn't available.

## Limitations
One listener, a small sample (28 clips), not blind to labels (votes were shown on the page). Not a clinical assessment.

## Possible follow-up
A blind re-labelling of a larger random sample of prolongation clips (votes hidden), to estimate label noise for that class. Ideally done later with an SLP-supervised protocol.
