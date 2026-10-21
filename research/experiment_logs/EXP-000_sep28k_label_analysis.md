# EXP-000 — SEP-28k label and partition analysis

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Status | complete |
| Type | Descriptive data analysis (no model, no audio) |
| Script | `python scripts/analyze_sep28k_labels.py` |

## Objective
Describe the real SEP-28k labels before any modelling: class balance, multi-label structure, speaker concentration, and whether the published speaker-aware partitions are actually leak-free.

## Data
- `SEP-28k-Extended_clips.csv` from th-nuernberg/ml-stuttering-events-dataset-extended @ `5f599c5`; labels are identical to apple-aiml-research/ml-stuttering-events-dataset @ `fe954e1` (byte-identical `SEP-28k_labels.csv`).
- 28,177 clips, 385 episodes, 8 shows, 497 clustered speakers.
- Licence: CC BY-NC 4.0. Only aggregate statistics are reproduced here.

## Results

**Label prevalence (share of all 28,177 clips) by annotator-vote threshold**

| Label | ≥1 vote | ≥2 votes (used) | 3 votes |
|---|---|---|---|
| Prolongation | 30.3 % | 10.0 % | 2.8 % |
| Block | 42.5 % | 12.0 % | 1.9 % |
| Sound repetition | 19.9 % | 8.3 % | 3.1 % |
| Word repetition | 16.4 % | 9.8 % | 5.7 % |
| Interjection | 34.3 % | 21.2 % | 12.0 % |
| No stuttered words | 75.9 % | 57.0 % | 30.5 % |

The ≥2-vote figures match those in Lea et al. (2021), confirming the parser and threshold reproduce the paper.

**Exclusions.** 1,962 clips (7.0 %) had ≥2 votes for Unsure, PoorAudioQuality, DifficultToUnderstand, Music or NoSpeech → **26,215 usable clips**.

**Multi-label structure (usable clips, ≥2 votes).** 0 events: 12,858 · 1 event: 10,472 · 2 events: 2,674 · 3: 203 · 4: 8. So **11 % of clips carry two or more event types** — a single-label classifier would be wrong by construction on these. The most frequent pairs are interjection + word repetition (660) and interjection + prolongation (647).

**Binary target ("any event").** Positive rate 51.0 % with interjections, 36.1 % without — the D-005 choice changes the problem substantially.

**Speaker concentration.** The top 4 speakers hold 53.8 % of usable clips; the largest single speaker 31.3 %.

**Partitions.** Partitions SEP28k-E, -T, -D all pass `check_no_leakage` on speaker IDs.

| SEP28k-E | Clips | Speakers | "Any event" rate |
|---|---|---|---|
| train | 14,092 | **4** | 51.8 % |
| dev | 5,932 | 245 | 50.8 % |
| test | 6,191 | 245 | 49.1 % |

-T and -D reuse the same three groups with roles swapped (train on many speakers, test on the 4 dominant ones).

## Interpretation
- **SEP28k-E trains on only four voices** and tests on 245 unseen speakers. It is a hard generalisation test: expect lower scores than papers using random splits, and that gap is the point. We keep it as the primary split for comparability (D-008), and add SEP-12k 5-fold CV (many speakers in training) so results don't rest on four voices alone.
- Blocks are labelled by at least one annotator in 42 % of clips but by all three in under 2 % — the clearest sign of how uncertain block labels are (κ = 0.25 in the paper).
- Positive rates are balanced across splits, so differences between dev and test scores will reflect speakers, not class balance.

## Limitations
Speaker IDs come from clustering, not ground truth. Statistics are for the full label set; once audio is retrieved, the analysis must be repeated on the retrieved subset (see fetch log).

## Next steps
Retrieve audio (`scripts/fetch_sep28k.py`), re-run this analysis restricted to retrieved clips, then Stage 2 baseline (EXP-001).

## Environment
Python 3.13, pandas 3.0.5, numpy 2.5.3. Runtime < 5 s.

---

## Addendum (2026-10-21): what was actually retrieved

Audio retrieved locally with `scripts/fetch_sep28k.py` (log: `data/metadata/sep28k_fetch_log.csv`, not committed).

- **Episodes:** 258 retrieved (248 ok + 10 length-mismatch) of 385; 127 unavailable — StrongVoices (35, HTTP 404), StutteringIsCool (85, 404), IStutterSoWhat (5, HTTP 522 server not responding), MyStutteringLife (2, 404).
- **Clips:** 20,896 of 28,177 (74.2 %); **19,516 usable** after quality exclusions (vs 26,215).
- **Length mismatches are benign.** In all 10 cases the downloaded file is only 0.14–0.46 s shorter than the end of the last labelled clip, so exactly one final clip per episode was dropped. A shortfall this small is consistent with MP3 decoder padding differences, not re-edited episodes; re-edits would shift by seconds or minutes. Earlier clips in these episodes are kept. *Still to do: the M1 listening check on a few clips from these episodes.*

**Label balance — published vs retrieved (usable clips, ≥2 votes)**

| | Published | Retrieved |
|---|---|---|
| Prolongation | 10.2 % | 9.9 % |
| Block | 11.9 % | 12.0 % |
| Sound repetition | 8.2 % | 8.7 % |
| Word repetition | 10.3 % | 11.7 % |
| Interjection | 22.2 % | 23.6 % |
| Any event (incl. interjections) | 51.0 % | 53.3 % |
| Any event (excl. interjections) | 36.1 % | 37.6 % |

Losing three podcasts shifts prevalence by at most ~1.4 points; no class is depleted.

**Partitions after retrieval**

| SEP28k-E | Clips | Speakers | "Any event" rate |
|---|---|---|---|
| train | 11,696 / 14,092 | **3 / 4** | 53.3 % |
| dev | 3,855 / 5,932 | 136 / 245 | 53.5 % |
| test | 3,965 / 6,191 | 142 / 244 | 53.3 % |

SEP-12k (for speaker-grouped CV): 7,820 of 11,972 usable clips, 278 speakers.

**Consequence.** The lost training speaker is the StutteringIsCool host, so SEP28k-E now trains on three voices. **Our SEP28k-E numbers are therefore not directly comparable with papers that had the full dataset**, and that must be stated wherever they are reported. SEP-12k CV (278 training speakers) becomes the more informative estimate of generalisation.
