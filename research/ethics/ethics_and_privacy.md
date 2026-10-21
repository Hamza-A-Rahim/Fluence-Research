# Ethics and privacy

*Living document. Nothing in the "Before…" sections below has been done yet — they are preconditions.*

## Principles

1. **Not a clinical tool.** Fluence never claims to diagnose, measure severity, or recommend treatment. Every interface and report says so.
2. **Respect for people who stutter.** Stuttering is a difference in speech, not a defect to be "caught". Language in the project avoids deficit framing (e.g. "possible stuttering event", never "error"). The stuttering community has published guidance on annotation and terminology (e.g. arXiv:2602.10403, *Disability-First AI Dataset Annotation*) — **to read** before any annotation work.
3. **Data minimisation.** Collect and keep only what an experiment needs. Audio stays local and out of version control.
4. **Licences are honoured.** SEP-28k annotations are CC BY-NC 4.0 and its audio belongs to the podcast owners; we do not redistribute audio, and outputs trained on it are non-commercial.

## Current stage: public research datasets only

- Allowed: published annotations; audio whose provider's scripts make it available for research, as agreed in decision D-006 (local, non-commercial, never redistributed); synthetic audio for testing.
- Not allowed: recording anyone (including friends or classmates), collecting audio through a website, or using the project lead's own recordings in a dataset that will be shared or published.

## Before any recording of human participants

All of the following must be in place and documented here:

- [ ] An academic supervisor (university faculty) who agrees to oversee the study.
- [ ] A qualified speech-language pathologist involved in protocol design and annotation.
- [ ] Ethics review by an appropriate body (a university ethics committee via the supervisor, or equivalent). The project lead is under 18, which normally means an institution must take formal responsibility.
- [ ] Written informed consent in Urdu and English, explaining purpose, recording, storage, who sees the data, withdrawal at any time, and that the tool gives no clinical advice.
- [ ] Parental/guardian consent **and** the participant's own assent for anyone under 18 — or, preferably, adults only.
- [ ] A data management plan: encrypted storage, access list, retention period, deletion procedure, de-identification (muting names, as KSoF and AS-70 did).
- [ ] A rule for what happens if a participant asks for feedback about their speech (refer to a clinician; never give model output as an assessment).

## Before any public upload feature on the website

- [ ] Explicit consent screen before upload; processing purpose stated.
- [ ] Default: process in memory and delete immediately; no retention without separate opt-in.
- [ ] File validation (type, size, duration) — partly implemented in `fluence.preprocessing.audio`.
- [ ] No third-party processing (no external speech APIs) unless disclosed and consented to.
- [ ] Rate limiting and abuse protection; no authentication system until genuinely needed.
- [ ] Privacy notice reviewed by the supervisor.
- [ ] Prominent "experimental research software — not a diagnostic tool" notice next to every result.
