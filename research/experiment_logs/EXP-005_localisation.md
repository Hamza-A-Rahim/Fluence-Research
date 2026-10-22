# EXP-005: Offset robustness and sliding-window detection

| Field | Value |
|---|---|
| Date | 2026-10-22 (extraction) / 2026-10-23 (analysis) |
| Status | complete |
| Git commit | `668038c` (clean working tree: yes) |
| Design | D-012 |
| Run | `fetch_context.py` → `extract_windows.py` → `localize.py`, on the desktop PC. Full output: `outputs/EXP-005/report.md`, `metrics.json`, figures |

## Objective
SEP-28k labels 3 s clips at known positions but gives no timing inside a clip. So this experiment asks a question the data can answer: **does the EXP-004 detector still work when its 3 s window is not aligned to a labelled clip**, as it would be when run over unsegmented audio?

## Hypothesis
No hypothesis was written down before the run; D-012 only fixed the design. Results are therefore descriptive, not confirmatory.

## Data
- SEP28k-E **dev** clips from the 108 episodes that contain dev clips and **no test clips**. Context of ±1.5 s around each clip was available for 2,784 of 2,786 clips (116 speakers). 7 offsets give 19,488 windows. **Test set not used.**
- 3 demo episodes as whole-episode timelines: MyStutteringLife 20 (18.4 min), StutterTalk 71 (22.3 min), WomenWhoStutter 85 (17.0 min).
- Labels and interjection handling as in EXP-002/004.

## Model
This is the frozen EXP-004 primary system: facebook/wav2vec2-large-xlsr-53 (revision `c3f9d88`), layer 14, mean+std pooling, logistic regression trained on SEP28k-E train only, with the same C. Nothing was refit or tuned in this experiment.

## Method
- **Offset windows:** a 3 s window starting at −1.5, −1.0, −0.5, 0, +0.5, +1.0 and +1.5 s from each clip start, each scored against that clip's label.
- **Sliding aggregation:** max and mean of the 7 window scores per clip. 95 % CIs come from a speaker bootstrap.
- **Timelines:** 3 s windows with a 1 s hop over the 3 whole episodes.
- **Sanity check:** offset-0 window scores vs the original clip scores.

## Results

**Sanity check passed.** Offset-0 scores correlate 1.0000 with the clip scores for every target, so the window pipeline reproduces the clip pipeline exactly.

**ROC-AUC by offset (dev clips):**

| Target | Clip | −1.5 s | −1.0 s | −0.5 s | 0 | +0.5 s | +1.0 s | +1.5 s | Sliding max [95 % CI] | Sliding mean [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Any event (incl. interj.) | 0.846 | 0.682 | 0.760 | 0.816 | 0.846 | 0.800 | 0.760 | 0.681 | 0.791 [0.774, 0.809] | 0.835 [0.820, 0.850] |
| Any event (excl. interj.) | 0.835 | 0.684 | 0.754 | 0.810 | 0.835 | 0.775 | 0.724 | 0.654 | 0.790 [0.769, 0.811] | 0.820 [0.801, 0.841] |
| Prolongation | 0.870 | 0.739 | 0.810 | 0.853 | 0.870 | 0.826 | 0.779 | 0.692 | 0.836 [0.804, 0.866] | 0.859 [0.832, 0.885] |
| Block | 0.755 | 0.676 | 0.702 | 0.742 | 0.755 | 0.706 | 0.675 | 0.624 | 0.721 [0.682, 0.759] | 0.744 [0.710, 0.778] |
| Sound repetition | 0.888 | 0.766 | 0.814 | 0.860 | 0.888 | 0.851 | 0.787 | 0.712 | 0.869 [0.832, 0.902] | 0.887 [0.856, 0.912] |
| Word repetition | 0.862 | 0.627 | 0.698 | 0.787 | 0.862 | 0.802 | 0.732 | 0.695 | 0.829 [0.800, 0.857] | 0.843 [0.813, 0.874] |
| Interjection | 0.914 | 0.732 | 0.818 | 0.858 | 0.914 | 0.876 | 0.846 | 0.722 | 0.875 [0.856, 0.894] | 0.907 [0.890, 0.923] |

**Share of positive clips by best-scoring offset.** For 7 offsets, a uniform spread would be 0.14 each.

| Target | −1.5 | −1.0 | −0.5 | 0 | +0.5 | +1.0 | +1.5 |
|---|---|---|---|---|---|---|---|
| Any event (incl. interj.) | 0.21 | 0.15 | 0.11 | 0.09 | 0.08 | 0.15 | 0.21 |
| Any event (excl. interj.) | 0.23 | 0.13 | 0.12 | 0.12 | 0.08 | 0.13 | 0.19 |
| Prolongation | 0.14 | 0.11 | 0.15 | 0.22 | 0.12 | 0.11 | 0.14 |
| Block | 0.16 | 0.13 | 0.15 | 0.17 | 0.12 | 0.11 | 0.16 |
| Sound repetition | 0.18 | 0.18 | 0.11 | 0.12 | 0.07 | 0.17 | 0.17 |
| Word repetition | 0.14 | 0.10 | 0.12 | 0.15 | 0.16 | 0.19 | 0.14 |
| Interjection | 0.13 | 0.17 | 0.12 | 0.10 | 0.14 | 0.18 | 0.15 |

**Figures** (in `outputs/EXP-005/`): `offset_auc.png`, `score_profile.png`, and `timeline_*.png` for each demo episode.

## Observations
1. **Misalignment costs accuracy roughly symmetrically, growing with the shift.** For "any event (incl.)", AUC drops by 0.03–0.05 at ±0.5 s, about 0.09 at ±1.0 s and about 0.16 at ±1.5 s. Word repetition drops the most at −1.5 s (0.627). Block is the flattest because it starts lowest.
2. **Averaging over windows recovers almost all of the clip-level performance; taking the max does not.** Sliding mean is within 0.001–0.015 of clip AUC for every target. Sliding max is lower for every target (e.g. 0.791 vs 0.835 for "any event incl."). One caveat: the 7 windows include the aligned one, so this is a best case for a 0.5 s-hop system, not a fully unaligned one.
3. **Part of the drop comes from no-event clips scoring higher when shifted, not only from event clips scoring lower** (`score_profile.png`). This fits shifted windows picking up neighbouring speech that may itself contain events. Those windows are scored against the clip's label, which no longer strictly applies to them, so the offset AUCs are a **lower bound** on real robustness, not a clean measurement of it.
4. **For the "any event" targets, positive clips most often score highest at the outermost offsets** (0.21 at each of ±1.5 s vs 0.09 at 0). This points the same way as observation 3: stuttering tends to cluster, and the context around a labelled clip often holds more events. Prolongation is the exception and peaks at offset 0 (0.22). This is a pattern in the data, not something tested.
5. **The timelines are very spiky.** In StutterTalk 71, both scores cross 0.8 many times between the sparse labelled clips. Only about 40 clips per episode are labelled, so most of these peaks can't be checked. In the last ~30 s of StutterTalk 71, "any event" stays near 1.0 while "interjection" is near 0. That stretch is unlabelled and should be listened to; it may be non-speech audio.

## Environment
Python 3.12.10, torch 2.14.1+cpu, transformers 5.20.2, scikit-learn 1.9.1, numpy 2.5.4; Intel i9-14900K, CPU only. Offset-window extraction took about 1.5 h (2026-10-22, 20:00–21:30 local) and the timelines about 20 min.

## Limitations
- There is no ground truth inside clips or for unlabelled audio, so **event timestamps are not evaluated** (D-012). Offset AUCs mix model degradation with label mismatch (observation 3).
- Dev clips only, restricted to the 108 test-free episodes. These clips were used for model selection in EXP-003, so the numbers carry the same mild optimism as EXP-003's dev results (EXP-004 test was 0.004–0.030 lower).
- Timeline episodes may include training speakers, as the figure notes say. The timelines are illustrations, not results.
- All earlier limitations still apply (3 training speakers, non-clinician labels, frozen encoder, English podcast speech).

## Conclusions and next steps
- The detector works on windows that aren't aligned to a labelled clip, with graceful degradation. With overlapping windows, **mean aggregation** keeps clip-level AUC; max aggregation should not be the default.
- Precise localisation can't be validated on SEP-28k. It needs data with word- or event-level timing (e.g. Boli, licence permitting), which ties into Stage 6.
- Before using the timelines for anything: listen to the high-scoring unlabelled stretches, including the end of StutterTalk 71, to see what produces the false-alarm-like peaks.
