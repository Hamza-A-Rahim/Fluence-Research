# Data directory (local only)

Everything in this folder except this README and the `.gitkeep` placeholders is **git-ignored**. Never commit audio, transcripts, annotation files from licensed datasets, or anything derived from a real person's recording.

| Folder | Contents |
|---|---|
| `raw/` | Audio exactly as obtained (e.g. downloaded podcast episodes). Never modified. |
| `interim/` | Intermediate artefacts (e.g. 16 kHz mono conversions, extracted 3 s clips). Rebuildable from `raw/` + scripts. |
| `processed/` | Model-ready features and split assignments. Rebuildable. |
| `metadata/` | Label/episode CSVs from dataset providers, `sep28k_fetch_log.csv` (what was retrieved). |
| `external/` | Third-party resources (e.g. pretrained-model downloads) if needed. |

## SEP-28k layout (created by `scripts/fetch_sep28k.py`)

```text
data/metadata/SEP-28k_labels.csv
data/metadata/SEP-28k_episodes.csv
data/metadata/SEP-28k-Extended_clips.csv      # speaker IDs + suggested partitions
data/metadata/SEP-28k-Extended_episodes.csv
data/metadata/sep28k_fetch_log.csv            # status of every episode
data/interim/sep28k_clips/<Show>/<EpId>/<Show>_<EpId>_<ClipId>.wav
data/raw/sep28k_episodes/<Show>/<EpId>.wav    # only with --keep-episodes
```

Windows instructions: [`docs/setup_windows.md`](../docs/setup_windows.md).

Labels: <https://github.com/apple-aiml-research/ml-stuttering-events-dataset> (CC BY-NC 4.0).
Speaker metadata and partitions: <https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended> (CC BY-NC 4.0).

## Rules

1. Record the dataset version (commit hash of the provider's repo, download date) in every experiment log.
2. Do not redistribute audio. Copyright for SEP-28k audio remains with the podcast owners.
3. Recordings of real people are not added here until the ethics plan in `research/ethics/` is in place.
