# Evaluation protocol

*Applies to every experiment from Stage 2 onwards. Changing any rule here requires a decision-log entry.*

## 1. Splits — no speaker in two partitions

- **Primary:** SEP-28k-E partition from SEP-28k-Extended (fixed train/dev/test, speaker-aware). Chosen so results are comparable with published work.
- **Secondary (robustness):** SEP-12k speaker-exclusive 5-fold cross-validation, reported as mean ± standard deviation across folds.
- If we ever split ourselves, use `fluence.data.splits.group_split` with **speaker** as the group (falling back to episode, then show), and run `check_no_leakage` — it raises an error if any group crosses partitions.
- The **test set is touched once per finished model**. All tuning (features, thresholds, hyper-parameters) uses dev or cross-validation.
- Achieved partition sizes, speaker counts and positive rates are reported, not the requested fractions (`split_report`).

## 2. Labels

- A label is present when **≥ 2 of 3 annotators** chose it (Lea et al., 2021). Sensitivity check: also report results at ≥ 1 and = 3 votes for the final model.
- Drop clips where ≥ 2 annotators marked Music, NoSpeech, PoorAudioQuality, Unsure or DifficultToUnderstand; log how many were dropped per partition.
- The binary target's treatment of interjections is fixed per experiment (decision D-005) and stated in the log.

## 3. Metrics

For each event type and for "any event":

- **Precision, recall, F1 for the positive class** (the event), not accuracy — with ~57 % fluent clips, a model that always says "fluent" has 57 % accuracy and zero usefulness.
- **Confusion matrix** (binary) per label.
- **Threshold-free:** ROC-AUC and average precision (PR-AUC), so models can be compared without a chosen threshold.
- **Reference baselines in every table:** majority class, and random guessing at the training positive rate. A model is only interesting relative to these.
- **Why this matters here (from EXP-000):** with interjections counted, 51 % of usable clips contain an event. A "model" that always answers *event* then has precision 0.51, recall 1.0 and **F1 ≈ 0.675** without listening to anything. Lea et al. (2021) report "any event" F1 of 66.8 on SEP-28k; their positive-class definition and split differ from ours, so this is not a like-for-like comparison, but it shows F1 alone cannot tell a useful detector from a trivial one on this target. **ROC-AUC and PR-AUC are therefore the primary comparison metrics for the binary task**, with F1 reported alongside the always-event baseline.
- Multi-label (Stage 3): per-label metrics, macro-F1 across the five events, and the number of clips where predicted label *sets* match exactly.
- **Uncertainty:** 95 % bootstrap confidence intervals over **speakers** (resample speakers, not clips — clips from one speaker are not independent).
- **Context:** report per-label results next to annotator agreement (Fleiss κ). Low-κ labels (prolongation 0.11, block 0.25) get explicitly cautious interpretation.

## 4. Thresholds

Default decision threshold 0.5 on calibrated probabilities. If a threshold is tuned (e.g. to maximise dev F1), it is tuned on dev only and then frozen before testing.

## 5. Reproducibility

Every experiment log (template in `research/experiment_logs/`) records: git commit, dataset versions and download date, fraction of audio retrieved, random seeds, full configuration, library versions, and hardware. Results that cannot be regenerated from these are not reported.

## 6. Known limitations of this protocol

- Speaker IDs in SEP-28k-Extended come from clustering and can be wrong; leakage is reduced, not eliminated.
- Clips were sampled near pauses, so performance on continuous speech may differ.
- Annotators were not clinicians; labels are "what trained listeners heard", not clinical ground truth.
- Missing podcast audio may change the class balance relative to the published dataset.
