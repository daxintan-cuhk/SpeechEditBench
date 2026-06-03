# Evaluation

The main entry point is:

```bash
python scripts/run_eval.py
```

## Output Naming

Each model output must be named by `sample_id`:

```text
<output_dir>/<sample_id>.wav
```

The evaluator also accepts `.flac` and `.mp3`, and it checks both the specified
directory and an optional `audio/` subdirectory.

## Single-Task Evaluation

```bash
python scripts/run_eval.py \
  --task style_editing \
  --output-dir outputs/my_model/style_editing \
  --model-name my_model
```

Use `--samples-file` to evaluate a custom subset:

```bash
python scripts/run_eval.py \
  --task style_editing \
  --output-dir outputs/my_model/style_editing \
  --samples-file data/style_editing/samples.jsonl \
  --model-name my_model \
  --eval-set full
```

## All-Task Evaluation

```bash
python scripts/run_eval.py \
  --task all \
  --output-root outputs/my_model \
  --model-name my_model
```

This expects task-specific output directories:

```text
outputs/my_model/<task_id>/<sample_id>.wav
```

For CI or official submission checks, add `--strict`. In strict mode, missing
outputs and per-sample evaluation errors make the runner exit non-zero. Even
without `--strict`, task-level preflight errors, task exceptions, and zero
evaluated outputs exit non-zero.

## Preflight And Language

Before each task starts, `run_eval.py` checks the evaluator assets that are
required by that task and the selected `samples.jsonl`. For example, an English
content subset requires Whisper, a Chinese subset requires Paraformer, a
speaker task requires WavLM speaker verification, and noise-transfer acoustic
samples require PANNs. Diagnostic-only models such as UTMOS are not hard gates.

Use the setup checker for a full local readiness report:

```bash
python scripts/check_eval_setup.py
```

Runner-level messages are Chinese by default. Use English with:

```bash
python scripts/run_eval.py \
  --task style_editing \
  --output-dir outputs/my_model/style_editing \
  --model-name my_model \
  --cli-lang en
```

## Result Layout

Results are written to:

```text
eval_results/<model_name>/<task_id>/<eval_set>/
```

Each task directory contains:

- `latest.json`: latest per-sample task results
- `aggregate_summary.json`: runner-level summary for the latest run
- `runs/<run_id>.json`: historical per-sample results
- `runs/<run_id>_aggregate_summary.json`: historical aggregate result

## Metrics

SpeechEditBench reports three high-level quantities:

- `target_success_rate`: whether the requested edit target is achieved
- `content_preservation_pass_rate`: whether the expected transcript is preserved
- `joint_success_rate`: whether both target and preservation criteria pass

For content editing, the target is lexical content itself, so target and joint
success are reported together through edit success / exact match style metrics.

Task-specific target checks include:

- content editing: ASR-based target transcript and edit-span checks
- speaker editing: output-reference speaker similarity
- emotion/style/paralinguistic editing: Gemini-compatible multimodal judge
- prosody editing: duration ratio, F0 shift, or stress prominence
- acoustic editing: DNSMOS gain, RT60 target range, or PANNs scene match
- compositional editing: component-wise reuse of atomic task metrics

For non-content tasks, reported preservation is operationalized as lexical
content preservation through ASR-based WER/CER.

## Multimodal Judge

Emotion, style, paralinguistic, and some compositional components require a
Gemini-compatible audio judge. Configure:

```bash
export GEMINI_API_KEY="..."
```

By default the evaluator uses Google Gemini's public API endpoint. To use a
compatible proxy, set:

```bash
export GEMINI_BASE_URL="https://your-compatible-endpoint"
```

Parallel judge workers can be controlled with:

```bash
export SPEECHEDITBENCH_JUDGE_WORKERS=4
```
