# Dataset assessment

*Compiled 21 October 2026 from the dataset repositories, websites and papers listed under Sources. Facts are as stated by those sources on that date; anything we have not verified ourselves is marked **unverified**.*

## Summary

| | SEP-28k | SEP-28k-Extended | FluencyBank (TalkBank) | KSoF | AS-70 | Boli |
|---|---|---|---|---|---|---|
| Language | English | English (same clips) | Mainly English; also bilingual children, L2 learners | German | Mandarin | English + Hindi, Telugu, Bengali, Marathi, Assamese |
| Size | 28,177 clips × 3 s (+ 4,144 FluencyBank clips re-annotated) | adds metadata to SEP-28k | Many corpora; 32 adults who stutter used by SEP-28k | 5,597 clips × 3 s, 37 speakers | 48.8 h, 70 speakers | ~2.5–2.8 h, 28 speakers |
| Labels | 5 events + fluent + quality labels, 3 annotators each, multi-label | same + speaker IDs, gender, 4 partitionings | CHAT transcripts | 5 events + *modified speech* | 5 events, character level | 5 events, word level with timestamps |
| Audio included? | **No** — download from podcast URLs | **No** | Yes, for members | On request | Download link (licence **unverified**) | Stated open access (licence **unverified**) |
| Licence | Annotations CC BY-NC 4.0; audio © podcast owners | CC BY-NC 4.0 | Members only; CC BY-NC-SA badge on site | Research use on request | **unverified** | **unverified** |
| Usable by us now? | Labels yes; audio pending decision D-006 | Yes (metadata) | **No** — needs a faculty sponsor | Needs a request | Check terms | Check terms |

## 1. SEP-28k (Apple, 2021)

- **Content.** 28,177 three-second clips drawn from 385 episodes of 8 podcasts on stuttering, plus 4,144 clips from FluencyBank interviews with 32 adults who stutter. Clips were chosen near pauses found by voice-activity detection, because disfluencies cluster around pauses — so the clip distribution is **not** a random sample of speech.
- **Annotations.** Each clip was labelled by three trained, non-clinician annotators. Labels: Prolongation, Block, SoundRep, WordRep, Interjection, NoStutteredWords, plus NaturalPause, Unsure, PoorAudioQuality, DifficultToUnderstand, Music, NoSpeech. The CSV stores the **number of annotators** (0–3) choosing each label, so clips can carry several labels.
- **Format check (done).** `SEP-28k_labels.csv` columns: `Show, EpId, ClipId, Start, Stop` + the label counts. Start/Stop are sample indices; every row we inspected spans exactly 48,000 samples (3 s at 16 kHz). Values have a space after each comma — our parser handles this and is tested against a synthetic copy of the format. `SEP-28k_episodes.csv` has **no header**; columns are show name, episode label, audio URL, show tag, episode index; 385 rows across 8 shows.
- **Class balance (≥2 of 3 annotators, from the paper).** Block 12.0 %, Prolongation 10.0 %, SoundRep 8.3 %, WordRep 9.8 %, Interjection 21.2 %, No dysfluency 56.9 %.
- **Annotation reliability (Fleiss κ).** Word repetition 0.62, interjection 0.57, sound repetition 0.40, fluent 0.39, block 0.25, prolongation 0.11. **This sets a ceiling:** a model cannot be reliably "right" about prolongations when humans agree so little. Any per-class result must be read against these values.
- **Audio access.** Not distributed. Apple's `download_audio.py` fetches episodes from the original hosts. Podcast URLs from 2011–2020 may have died; the number still available is **unknown** until we run `scripts/check_sep28k_availability.py` (HEAD requests only, no download).
- **Speakers.** No speaker IDs in the original release — the gap SEP-28k-Extended fills.

## 2. SEP-28k-Extended (TH Nürnberg, 2022)

- Adds per-episode speaker and gender information, host names, a per-clip speaker assignment from embedding clustering, cluster-quality scores (silhouette etc.), an `is_probably_host` flag, and **four ready-made partitionings** (SEP-12k for 5-fold CV; SEP-28k-E, -T and -D with train/dev/test).
- **Why it matters for us.** Bayerl et al. found one host appears in up to ~46 % of SEP-28k clips and four hosts account for ~59 %. Random clip-level splits therefore put the same voices in training and testing. Their published partitions are the community's answer, and using them makes our numbers comparable with later work.
- Speaker IDs are **inferred by clustering**, not ground truth, so speaker separation is approximate; the quality scores let us apply stricter filters.

## 3. FluencyBank (TalkBank)

- A shared database covering typically developing monolingual **and bilingual** children, children and adults who stutter or clutter, and second-language learners. Transcripts are in CHAT format (CLAN tools).
- **Access.** Research data is password-protected and limited to members. **Students cannot apply on their own**: a faculty advisor must request membership and request the student's access, and takes responsibility for its use. The "Teaching" section is open access and needs no membership.
- The SEP-28k authors discarded FluencyBank's own disfluency alignments as inaccurate and re-annotated 4,144 clips; those **labels** are in Apple's repo, but the **audio** still comes from FluencyBank.
- The site notes it is under review for possible modification in response to US administration directives — access could change.
- **Implication.** FluencyBank (including any bilingual corpora in it) is realistically available to us only with an academic sponsor. That sponsor is also needed for the Urdu-English study, so finding one is a single task that unlocks both.

## 4. Other datasets relevant to the cross-language question

- **KSoF (German, therapy recordings).** 5,597 clips, 37 speakers, adds a *modified speech* label for therapy techniques; κ ranges from 0.37 (blocks) to 0.78 (interjections) with naive annotators. Available for research on request to the authors.
- **AS-70 (Mandarin).** 48.8 h from 70 adults who stutter (conversation + read voice commands), character-level labels for the same five event types; redacted for personal information. wav2vec 2.0 was the best detector on every label; blocks were hardest for all models.
- **Boli (Indian languages).** ~2.5–2.8 h from 28 people who stutter reading the Rainbow Passage in English and their mother tongue, plus picture description. **22 speakers' mother tongue is Hindi.** Spoken Hindi and Urdu are largely mutually intelligible, so Boli is the closest existing public resource to our target population — but it is small, read speech, and contains no Urdu or code-switched speech. Its consent/ethics process is not described in the paper; we must check the release terms before use.

## 5. What this means for Fluence

1. **Development data:** SEP-28k labels + SEP-28k-Extended partitions, with whatever audio is still retrievable. Report the retrieved fraction and check that missing episodes don't skew the class balance.
2. **First evaluation split:** SEP-28k-E (fixed, documented, comparable). Group by speaker, never by clip.
3. **Exclude** clips that ≥2 annotators marked Music, NoSpeech, PoorAudioQuality, Unsure or DifficultToUnderstand; report how many were removed.
4. **Cross-language pilot without new recordings:** Boli's Hindi vs English recordings from the *same speakers* allow a within-speaker comparison — a cleaner test of language effects than comparing different people. Feasibility depends on its licence.
5. **No Urdu-English stuttered-speech dataset was found** in this search. Creating one is a genuine contribution but requires supervision, ethics review and clinical annotation (see `research/ethics/`).

## Sources

- Lea, C., Mitra, V., Joshi, A., Kajarekar, S., Bigham, J. P. (2021). *SEP-28k: A Dataset for Stuttering Event Detection from Podcasts with People Who Stutter.* ICASSP 2021. <https://arxiv.org/abs/2102.12394>; data: <https://github.com/apple-aiml-research/ml-stuttering-events-dataset>
- Bayerl, S. P., Wagner, D., Bocklet, T., Riedhammer, K. (2022). *The Influence of Dataset Partitioning on Dysfluency Detection Systems.* Text, Speech, and Dialogue (TSD 2022). <https://arxiv.org/abs/2206.03400>; data: <https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended>
- Bernstein Ratner, N., MacWhinney, B. (2018). Fluency Bank: A new resource for fluency research and practice. *Journal of Fluency Disorders*, 56, 69–80. Site: <https://talkbank.org/fluency/>
- Bayerl, S. P., Wolff von Gudenberg, A., Hönig, F., Nöth, E., Riedhammer, K. (2022). *KSoF: The Kassel State of Fluency Dataset — A Therapy Centered Dataset of Stuttering.* arXiv:2203.05383 (venue not yet verified). <https://arxiv.org/abs/2203.05383>
- Gong, R., Xue, H., Wang, L., et al. (2024). *AS-70: A Mandarin stuttered speech dataset for automatic speech recognition and stuttering event detection.* arXiv:2406.07256 (venue not yet verified). <https://arxiv.org/abs/2406.07256>
- Batra, A., Narang, M., Sharma, N. K., Das, P. K. (2025). *Boli: A dataset for understanding stuttering experience and analyzing stuttered speech.* arXiv preprint. <https://arxiv.org/abs/2501.15877>
