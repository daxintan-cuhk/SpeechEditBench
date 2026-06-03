# SpeechEditBench

SpeechEditBench is a bilingual benchmark for instruction-guided speech editing.
It evaluates whether a system can apply a requested edit to source speech while
preserving the expected lexical content. The benchmark covers seven atomic
editing tasks and compositional editing:

- `content_editing`
- `speaker_editing`
- `emotion_editing`
- `style_editing`
- `prosody_editing`
- `paralinguistic_editing`
- `acoustic_editing`
- `compositional_editing`

This repository contains the benchmark metadata, evaluation code, release
manifest, and user-facing documentation. The full audio assets are distributed
through Hugging Face.

## Data

The GitHub repository includes `data/*/samples.jsonl` for inspecting task
schemas. Audio files are not stored in git. Download the released audio assets
from Hugging Face:

```bash
python scripts/download_hf_dataset.py \
  --repo-id DiscreteSpeech/SpeechEditBench \
  --revision v1.1
```

The download helper fetches `data/**` by default, so it will not overwrite this
repository's README or documentation.

After download, the repository should contain:

```text
data/<task_id>/samples.jsonl
data/<task_id>/audio/**
```

See [docs/data.md](docs/data.md) for schema notes, the task input protocol,
and release manifest details.

## Installation

Create an environment with Python 3.11, which is the tested baseline for the
v1.1 release, then install the Python dependencies. `requirements.txt` installs
both the lightweight dataset helpers and the full evaluator stack:

```bash
conda create -n speecheditbench python=3.11
conda activate speecheditbench
pip install -r requirements.txt
```

For data download or metadata inspection only, use:

```bash
pip install -r requirements-core.txt
```

Several metrics require external evaluator models such as Whisper, Paraformer,
WavLM, UTMOS, DNSMOS, and PANNs. These large files are not included in this
repository. See [docs/eval_models.md](docs/eval_models.md) for the expected
paths and upstream sources.

To check local evaluator readiness:

```bash
python scripts/check_eval_setup.py
```

The evaluator model download commands are listed in
[docs/eval_models.md](docs/eval_models.md).

## Model Outputs

For each sample, save the edited audio as:

```text
<output_dir>/<sample_id>.wav
```

The evaluator also accepts `.flac` and `.mp3`, and it can read outputs placed in
an `audio/` subdirectory:

```text
<output_dir>/audio/<sample_id>.wav
```

For all-task evaluation, use:

```text
<output_root>/<task_id>/<sample_id>.wav
```

or:

```text
<output_root>/<task_id>/audio/<sample_id>.wav
```

In the standard benchmark setting, the model-visible inputs are the source
`audio_path`, the natural-language `instruction`, and speaker reference audio
only for speaker-editing tasks/components. Ground-truth `anchor` fields and
acoustic `target_reference_path` entries are evaluator labels, not hidden model
inputs.

## Evaluation

Run one task:

```bash
python scripts/run_eval.py \
  --task content_editing \
  --output-dir outputs/my_model/content_editing \
  --model-name my_model
```

Run all tasks:

```bash
python scripts/run_eval.py \
  --task all \
  --output-root outputs/my_model \
  --model-name my_model
```

Runner messages default to Chinese. Use English runner messages with:

```bash
python scripts/run_eval.py \
  --task content_editing \
  --output-dir outputs/my_model/content_editing \
  --model-name my_model \
  --cli-lang en
```

Results are written under:

```text
eval_results/<model_name>/<task_id>/<eval_set>/
```

See [docs/evaluation.md](docs/evaluation.md) for metric definitions, output
layout, and task-specific notes.

Use `--strict` when you want CI-style failure semantics: missing outputs and
per-sample evaluation errors will make the runner exit non-zero. Task-level
preflight errors, task exceptions, and zero evaluated outputs always exit
non-zero.

## Version

The current release manifest is:

```text
release_manifests/v1.1/dataset_manifest.json
```

It records 4,700 samples across 8 tasks and 5,400 audio files in the full data
release.

## License

The SpeechEditBench code, documentation, metadata, and benchmark assets authored
by the SpeechEditBench contributors are released under the Apache License 2.0.
See [LICENSE](LICENSE).

The benchmark is derived from multiple upstream speech corpora. Users are
responsible for complying with the applicable upstream dataset licenses and
terms when using the released audio assets.

## Citation

If you use SpeechEditBench, please cite:

```bibtex
@misc{zhang2026speecheditbenchbilingualmultiattributebenchmark,
      title={SpeechEditBench: A Bilingual Multi-Attribute Benchmark for Instruction-Guided Speech Editing},
      author={Hanlin Zhang and Daxin Tan and Dehua Tao and Xiao Chen and Haochen Tan and Linqi Song},
      year={2026},
      eprint={2606.01804},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2606.01804},
}
```
