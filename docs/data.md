# Data

SpeechEditBench data is organized by task:

```text
data/<task_id>/samples.jsonl
data/<task_id>/audio/**
```

The GitHub repository includes `samples.jsonl` files for schema inspection.
Audio files are distributed through Hugging Face and can be downloaded with:

```bash
python scripts/download_hf_dataset.py \
  --repo-id DiscreteSpeech/SpeechEditBench \
  --revision v1.1
```

The download script uses the local `samples.jsonl` files to fetch the benchmark
metadata and referenced audio assets by default. This keeps the local checkout
aligned with the release manifest and avoids downloading extra unused audio
files that may exist in the Hugging Face repository. Pass `--all-data` to mirror
every file under `data/**`, or `--all-files` if you explicitly want the complete
Hugging Face snapshot.

## Release Scale

The v1.1 release contains:

- 8 tasks
- 4,700 samples
- 5,400 audio files

The release manifest is stored at:

```text
release_manifests/v1.1/dataset_manifest.json
```

## Sample Format

Every row in `samples.jsonl` is a JSON object with common fields such as:

- `sample_id`: unique sample identifier
- `task`: task id
- `audio_path`: source audio path
- `instruction`: natural-language edit instruction
- `transcript`: source transcript when available
- `anchor`: task-specific target metadata for atomic tasks
- `language`: `en` or `zh`
- `source_dataset`: source corpus tag
- `benchmark_version`: release version

Some tasks include additional paths:

- `speaker_editing` includes `reference_audio_path`
- `acoustic_editing` may include `anchor.target_reference_path`
- `compositional_editing` includes a `components` list; each component reuses an
  atomic-task anchor under `components[].anchor`

All relative audio paths are resolved from the repository root after the Hugging
Face snapshot is downloaded.

## Task Input Protocol

For the standard benchmark setting, each system should generate one edited
audio file from the released model-visible inputs:

- `audio_path`: source speech to edit
- `instruction`: natural-language edit instruction
- `reference_audio_path`: target-speaker reference audio, only for
  `speaker_editing`
- `components[].reference_audio_path` or `components[].anchor.reference_wav`:
  target-speaker reference audio, only for speaker components inside
  `compositional_editing`

All other fields are metadata or evaluator labels. In particular, atomic-task
`anchor`, `components[].anchor`, `transcript`, `source_dataset`,
`benchmark_version`, and `anchor.target_reference_path` should not be used as
hidden model inputs in the standard setting. If a system uses transcripts,
target references, or anchor labels as additional conditioning, report it as a
separate transcript- or reference-conditioned setting.

For `compositional_editing`, use the top-level `audio_path` and the top-level
`instruction` to produce a single final output audio. Do not generate separate
outputs for individual components. Speaker reference audio may appear inside a
component; component anchors and acoustic `target_reference_path` values remain
evaluator-only.

The evaluator expects the output file to be named by `sample_id`:

```text
<output_dir>/<sample_id>.wav
<output_dir>/audio/<sample_id>.wav
```

`.flac` and `.mp3` outputs are also accepted. When running `--task all`, place
outputs under:

```text
<output_root>/<task_id>/<sample_id>.wav
<output_root>/<task_id>/audio/<sample_id>.wav
```

## Task-Specific Anchors

Anchors are ground-truth verification targets used by the evaluator.

| Task | Evaluator anchor fields |
|---|---|
| `content_editing` | `edit_type`, `transcript_target`, `edit_original`, `edit_target`, `insert_after` |
| `speaker_editing` | `reference_wav`, `target_speaker`, `reference_speaker_id` |
| `emotion_editing` | `source_emotion`, `target_emotion`, `emotion_taxonomy` |
| `style_editing` | `source_style`, `target_style` |
| `prosody_editing` | `prosody_type`, `direction`, `stress_words` |
| `paralinguistic_editing` | `operation`, `event` |
| `acoustic_editing` | `subtask`, `degradation_type`, `env_type`, `env_subtype`, `rt60_target_range`, `target_reference_path` |
| `compositional_editing` | `components[]`, where each component reuses an atomic-task anchor |

## Task Counts

| Task | Samples |
|---|---:|
| `content_editing` | 600 |
| `speaker_editing` | 200 |
| `emotion_editing` | 1,400 |
| `style_editing` | 600 |
| `prosody_editing` | 600 |
| `paralinguistic_editing` | 400 |
| `acoustic_editing` | 500 |
| `compositional_editing` | 400 |
