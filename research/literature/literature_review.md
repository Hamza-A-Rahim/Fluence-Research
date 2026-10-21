# Literature review — automatic stuttering event detection

*Version 0.1, 21 October 2026. A starting map, not a finished review. Each claim is tied to a source we have read (abstract or full text, as noted). Items marked **to read** have been identified but not yet read in full.*

## 1. The task as the field defines it

Most recent work treats stuttering detection as **clip classification**: given a short (usually 3 s) audio segment, decide which disfluency types it contains. The five event types used across SEP-28k, KSoF, AS-70 and Boli are **blocks, prolongations, sound repetitions, word/phrase repetitions and interjections** (KSoF adds *modified speech* for therapy techniques). Clip-level labels do **not** give event timestamps; localisation is a separate, harder problem (our Stage 5).

## 2. Key findings so far

**The SEP-28k paper (Lea et al., 2021) — full text read.** Introduced the dataset and a baseline. Input: 40 log-mel filterbank energies (25 ms window, 10 ms hop), optionally with pitch, articulatory and phoneme features. Models: a single-layer LSTM baseline and a "ConvLSTM" with per-feature convolutions, multi-task outputs, focal loss and a concordance-correlation loss that uses annotator agreement. Reported F1 on SEP-28k for the improved model ranged from about 56 (blocks) to 71 (interjections), and 67 for "any event". Results were consistently higher on FluencyBank than SEP-28k, which the authors attribute to SEP-28k's wider speaker and style variety. The paper states the SEP-28k split sizes (25k/2k/1k) but **not** whether they were separated by speaker or podcast — one reason later work questioned the comparability of results.

**Dataset partitioning (Bayerl et al., TSD 2022) — full text read.** The single most important methodological paper for us. SEP-28k is dominated by a few podcast hosts, so random splits leak speaker identity between train and test. Using wav2vec 2.0 features with SVM and neural classifiers, they showed that changing only the split changed F1 dramatically — e.g. sound-repetition F1 ranged from 0.19 to 0.70 across leave-one-podcast-out folds. They released speaker-aware partitions and recommend always reporting the partition used and keeping test speakers out of training. *Our evaluation protocol adopts this directly.*

**Pretrained speech encoders (Bayerl et al., Interspeech 2022) — abstract read.** Fine-tuning wav2vec 2.0 on English stuttered speech and training SVMs on its embeddings improved F1 by up to 27 % relative over general-purpose wav2vec 2.0 features, evaluated on English (FluencyBank) and German (KSoF) — i.e. gains carried **across languages**. This is direct prior evidence for our research question.

**Multi-label detection (Bayerl et al., 2022, arXiv:2210.15982) — abstract read.** Argues dysfluencies frequently co-occur and should be modelled as multi-label rather than multi-class; a modified wav2vec 2.0 reached state-of-the-art results on SEP-28k-Extended and transferred across English and German datasets. *Supports our Stage 3 design (independent sigmoid outputs, not softmax).*

**Multilingual detection (Haas & Bayerl, 2026, arXiv:2603.26939) — abstract read; to read in full.** Trained a multi-label detector on pooled English, German and Mandarin data from four corpora; multilingual training matched or beat earlier single-language systems, and the authors argue stuttering is acoustically consistent enough across languages to support language-agnostic detection. **This is the closest existing work to our question and must be read in full before we finalise the research question** — in particular whether it reports train-on-one/test-on-another results.

**AS-70 (Gong et al., 2024) — abstract and summary read.** Mandarin dataset; wav2vec 2.0 beat StutterNet, ConvLSTM and Conformer on every label; blocks were hardest for every model — consistent with SEP-28k's low block agreement.

**Boli (Batra et al., 2025) — summary read.** Hindi + English (and other Indian languages) from the same speakers. Trained classical models (random forest, SVM, LSTM) on SEP-28k MFCCs and tested on Boli English audio; random forest did best and class balancing helped. Small and read speech, but the only South Asian resource found.

**Bilingual stuttering (clinical) — Schäfer & Robb, 2012, abstract read.** In 15 German-English bilinguals, stuttering was more frequent in the less dominant language (L2) and was distributed differently across content and function words. Implication for us: for Urdu-English speakers, **event frequency may differ by language for reasons that have nothing to do with the model.** A fair comparison has to separate "the model is worse on Urdu" from "the speaker stutters differently in Urdu". *To read:* broader reviews of bilingual stuttering.

**Review (Sheikh et al., 2022, arXiv:2107.04057) — to read.** Survey of ML methods for stuttering identification; useful for the related-work section.

## 3. Methods that are realistic for this project

| Approach | Compute | Evidence | Role here |
|---|---|---|---|
| Logistic regression / random forest on clip-summary MFCC + energy statistics | Laptop CPU, minutes | Boli used it; a standard floor | **Stage 2 baseline** |
| LSTM / ConvLSTM on log-mel frames (Lea et al.) | CPU feasible, GPU nicer | Original SEP-28k benchmark | Stage 4 comparison |
| Frozen wav2vec 2.0 / HuBERT / Whisper-encoder embeddings + linear or SVM head | One forward pass per clip; CPU slow but feasible, free GPU (Colab/Kaggle) better | Strongest results in Bayerl et al., AS-70 | **Stage 4 main candidate** |
| Fine-tuning a pretrained encoder | Needs a GPU | Best reported results | Only if Stage 4 shows clear headroom |

## 4. Open questions to resolve by reading

1. Haas & Bayerl (2026): which corpora, which model, and is there zero-shot cross-language evaluation?
2. What multilingual pretrained encoders (e.g. XLS-R, Whisper) have seen Urdu or Hindi, and does that matter for disfluency features?
3. How do clinical frameworks treat interjections — as stuttering-like disfluencies or "other" disfluencies? (Affects our binary target; see decision D-005.)
4. How have prior works handled clips with low annotator agreement — excluded, down-weighted, or modelled as soft labels?

## 5. Provisional refinement of the research question

The original question asks how well English-trained models identify events in Urdu-English bilinguals. The literature suggests a sharper version:

> *Does a stuttering-event detector trained on English speech perform differently on the same bilingual speakers' Urdu (or Hindi) speech than on their English speech, after accounting for differences in how often each event type occurs in each language?*

The within-speaker design controls for voice and severity; the class-frequency adjustment addresses the Schäfer & Robb finding. This is still provisional and will be revisited after the Haas & Bayerl paper is read in full.

## Sources

See `dataset_assessment.md` for dataset papers. Additional:

- Bayerl, S. P., Wagner, D., Nöth, E., Riedhammer, K. (2022). Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0. *Proc. Interspeech 2022*, 2868–2872. doi:10.21437/Interspeech.2022-10908
- Bayerl, S. P., Wagner, D., Hönig, F., Bocklet, T., Nöth, E., Riedhammer, K. (2022). Dysfluencies Seldom Come Alone — Detection as a Multi-Label Problem. arXiv:2210.15982
- Haas, F., Bayerl, S. P. (2026). Multilingual Stutter Event Detection for English, German, and Mandarin Speech. arXiv:2603.26939
- Schäfer, M., Robb, M. P. (2012). Stuttering characteristics of German-English bilingual speakers. *Clinical Linguistics & Phonetics*, 26(7), 597–612.
- Sheikh et al. (2022). Machine learning for stuttering identification: Review, challenges and future directions. arXiv:2107.04057 — *full author list and venue to be verified when read.*
