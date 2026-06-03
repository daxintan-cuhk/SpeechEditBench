"""
Benchmark-level evaluation runner.

Design goals:
- unified CLI for all tasks
- task-level independent evaluators
- safe defaults for benchmark users
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.registry import TaskSpec, get_task_registry
from scripts.check_eval_setup import check_model_groups, required_model_groups_for_samples

EVALUATOR_VERSION = "v2"


def _cli_lang() -> str:
    return os.getenv("SPEECHEDITBENCH_CLI_LANG", "zh")


def _t(zh: str, en: str) -> str:
    return en if _cli_lang() == "en" else zh


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _call_task(spec: TaskSpec, kwargs: dict) -> dict:
    """Call task evaluator with signature-aware kwargs filtering."""
    sig = inspect.signature(spec.run_evaluation)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = sorted(k for k in kwargs if k not in sig.parameters)
    if dropped:
        print(f"[WARN] {spec.task_id} {_t('未消费参数', 'unused arguments')}: {dropped}")
    return spec.run_evaluation(**accepted)


def _resolve_output_dir(args: argparse.Namespace, task_id: str) -> Path:
    if args.task == "all":
        if not args.output_root:
            raise ValueError(_t("--task all 需要提供 --output-root", "--task all requires --output-root"))
        return _normalize_output_dir(args.output_root / task_id)
    if not args.output_dir:
        raise ValueError(_t("单任务模式需要提供 --output-dir", "single-task mode requires --output-dir"))
    return _normalize_output_dir(args.output_dir)


def _normalize_output_dir(path: Path) -> Path:
    """Accept both <task>/<sample_id>.* and <task>/audio/<sample_id>.* layouts."""
    audio_dir = path / "audio"
    root_audio = any(path.glob(f"*{ext}") for ext in (".wav", ".flac", ".mp3"))
    if audio_dir.is_dir() and not root_audio:
        return audio_dir
    return path


def _resolve_results_file(results_root: Path, task_id: str, disabled: bool = False) -> Path:
    suffix = "_skipped.json" if disabled else ".json"
    return results_root / f"{task_id}{suffix}"


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "default"


def _is_default_samples_file(task_id: str, samples_file: Path | None) -> bool:
    if samples_file is None:
        return False
    default_samples = REPO_ROOT / "data" / task_id / "samples.jsonl"
    try:
        return samples_file.resolve() == default_samples.resolve()
    except FileNotFoundError:
        return False


def _infer_eval_set(task_id: str, output_dir: Path, samples_file: Path | None) -> str:
    """Infer a human-readable eval set name from common smoke/full layouts."""
    path_parts = {part.lower() for part in output_dir.parts}
    sample_stem = samples_file.stem.lower() if samples_file else ""
    is_smoke = sample_stem.startswith("smoke") or "test_model" in path_parts

    if is_smoke and task_id == "speaker_editing":
        if "source_copy" in path_parts:
            return "smoke_source_copy"
        if "reference_copy" in path_parts:
            return "smoke_reference_copy"
        return "smoke_copy"

    if is_smoke and task_id == "content_editing":
        return "smoke_source_copy"

    if is_smoke and task_id == "prosody_editing":
        if "source_copy" in path_parts:
            return "smoke_source_copy"
        return "smoke_copy"

    if is_smoke and task_id == "acoustic_editing":
        if "source_copy" in path_parts:
            return "smoke_source_copy"
        return "smoke_copy"

    if is_smoke and task_id == "compositional_editing":
        if "source_copy" in path_parts:
            return "smoke_source_copy"
        return "smoke_copy"

    if is_smoke and task_id in {"emotion_editing", "style_editing", "paralinguistic_editing"}:
        return "smoke_copy"

    if _is_default_samples_file(task_id, samples_file):
        return "full"

    if sample_stem:
        return _slug(sample_stem)
    return "default"


def _infer_dataset_version(samples_file: Path | None) -> str:
    """Infer the benchmark data version from common data roots."""
    if samples_file is None:
        return "unknown"
    try:
        rel = samples_file.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return "custom"
    parts = rel.parts
    if parts[:1] == ("data",):
        return "v1.1"
    if parts and re.fullmatch(r"data_v\d+_\d+", parts[0]):
        return parts[0].replace("data_", "v").replace("_", ".")
    return "custom"


def _infer_model_name(output_path: Path) -> str:
    """Infer model name from model_output/<model_name>/... layouts."""
    parts = output_path.parts
    for idx, part in enumerate(parts):
        if part == "model_output" and idx + 1 < len(parts):
            return _slug(parts[idx + 1])
    return _slug(output_path.name)


def _resolve_task_result_dir(
    results_root: Path,
    model_name: str,
    task_id: str,
    eval_set: str,
) -> Path:
    return results_root / model_name / task_id / eval_set


def _load_samples_for_preflight(samples_file: Path | None) -> tuple[list[dict], list[str]]:
    if samples_file is None:
        return [], [_t("缺少 samples_file", "missing samples_file")]
    if not samples_file.exists():
        return [], [_t(f"samples_file 不存在: {samples_file}", f"samples_file does not exist: {samples_file}")]

    samples: list[dict] = []
    errors: list[str] = []
    with samples_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    _t(
                        f"samples_file 第 {line_no} 行不是合法 JSON: {exc}",
                        f"samples_file line {line_no} is not valid JSON: {exc}",
                    )
                )
                continue
            if not isinstance(row, dict):
                errors.append(
                    _t(
                        f"samples_file 第 {line_no} 行不是 JSON object",
                        f"samples_file line {line_no} is not a JSON object",
                    )
                )
                continue
            samples.append(row)

    if not samples and not errors:
        errors.append(_t("samples_file 没有可评测样本", "samples_file contains no evaluable samples"))
    return samples, errors


def _format_missing(paths: list[str], *, limit: int = 4) -> str:
    if not paths:
        return "-"
    shown = paths[:limit]
    suffix = f", ... (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return ", ".join(shown) + suffix


def _preflight_errors_for_task(task_id: str, samples_file: Path | None) -> list[str]:
    """Return blocking preflight errors for a task."""
    samples, errors = _load_samples_for_preflight(samples_file)
    if errors:
        return errors

    model_groups = check_model_groups()
    required_groups = required_model_groups_for_samples(task_id, samples)
    for group_name in sorted(required_groups):
        row = model_groups.get(group_name)
        if not row:
            errors.append(_t(f"未知 evaluator model group: {group_name}", f"unknown evaluator model group: {group_name}"))
            continue
        if row.get("ok"):
            continue
        missing = _format_missing(row.get("missing") or [group_name])
        errors.append(
            _t(
                f"缺少 evaluator model group {group_name}（{row.get('description', '')}）：{missing}",
                f"missing evaluator model group {group_name} ({row.get('description', '')}): {missing}",
            )
        )
    return errors


def _summary_overall(summary: dict | None) -> dict:
    if not isinstance(summary, dict):
        return {}
    overall = summary.get("overall")
    return overall if isinstance(overall, dict) else {}


def _task_validation_errors(
    summary: dict | None,
    error_stats: dict,
    *,
    strict: bool,
) -> list[str]:
    """Return runner-level validation errors after a task evaluator completes."""
    errors: list[str] = []
    overall = _summary_overall(summary)

    outputs_evaluated = overall.get("outputs_evaluated")
    total = overall.get("total")
    missing_outputs = overall.get("missing_outputs")

    if isinstance(total, int) and total > 0 and outputs_evaluated == 0:
        errors.append("no model outputs were evaluated")

    if strict:
        if isinstance(missing_outputs, int) and missing_outputs > 0:
            errors.append(f"missing model outputs: {missing_outputs}")
        if error_stats.get("error_count", 0) > 0:
            errors.append(f"sample-level evaluation errors: {error_stats['error_count']}")

    return errors


def _summarize_detail_errors(results_file: Path) -> dict:
    """
    Extract per-sample error diagnostics from task result json.
    Returns: {"error_count": int, "top_errors": list[{"message": str, "count": int}]}
    """
    if not results_file.exists():
        return {"error_count": 0, "top_errors": []}
    try:
        with results_file.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {"error_count": 0, "top_errors": []}

    details = payload.get("details")
    if not isinstance(details, list):
        return {"error_count": 0, "top_errors": []}

    errs = [
        str(row.get("error")).strip()
        for row in details
        if isinstance(row, dict) and row.get("error")
    ]
    if not errs:
        return {"error_count": 0, "top_errors": []}

    counter = Counter(errs)
    top = [{"message": msg, "count": cnt} for msg, cnt in counter.most_common(3)]
    return {"error_count": len(errs), "top_errors": top}


def _annotate_result_file(
    results_file: Path,
    *,
    dataset_version: str,
    evaluator_version: str,
    model_name: str,
    task_id: str,
    eval_set: str,
    samples_file: Path | None,
) -> None:
    """Attach benchmark-level metadata to task result JSON files."""
    if not results_file.exists():
        return
    try:
        with results_file.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {_t('无法写入结果元数据', 'unable to write result metadata')} {results_file}: {exc}")
        return
    payload.setdefault("metadata", {})
    payload["metadata"].update({
        "dataset_version": dataset_version,
        "evaluator_version": evaluator_version,
        "model_name": model_name,
        "task_id": task_id,
        "eval_set": eval_set,
        "samples_file": str(samples_file) if samples_file else None,
    })
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeechEditBench benchmark evaluator runner")
    parser.add_argument("--task", required=True, help="task id or all / 任务名或 all")
    parser.add_argument("--output-dir", type=Path, default=None, help="single-task output directory / 单任务输出目录")
    parser.add_argument("--output-root", type=Path, default=None, help="all-task output root with task subdirectories / 全任务输出根目录")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "eval_results",
                        help="evaluation results root / 评测结果根目录")
    parser.add_argument("--samples-file", type=Path, default=None,
                        help="custom samples.jsonl for single-task mode; ignored for all / 单任务自定义 samples.jsonl")
    parser.add_argument("--model-name", default=None,
                        help="model name; inferred from model_output/<model_name>/... by default / 模型名称")
    parser.add_argument("--eval-set", default=None,
                        help="result set name, e.g. smoke_source_copy; inferred by default / 结果集合名")
    parser.add_argument("--dataset-version", default=None,
                        help="dataset version, e.g. v1.1; inferred from samples path by default / 数据版本")
    parser.add_argument("--include-disabled", action="store_true",
                        help="include registry-disabled tasks in all mode / all 模式包含 disabled 任务")
    parser.add_argument("--strict", action="store_true",
                        help="strict mode: missing outputs or sample-level errors exit non-zero / 严格模式")
    parser.add_argument("--cli-lang", choices=["zh", "en"], default=os.getenv("SPEECHEDITBENCH_CLI_LANG", "zh"),
                        help="runner 输出语言 / runner output language")
    args = parser.parse_args()
    os.environ["SPEECHEDITBENCH_CLI_LANG"] = args.cli_lang

    registry = get_task_registry(REPO_ROOT)
    task_ids = list(registry.keys())

    if args.task != "all" and args.task not in registry:
        print(f"[ERROR] {_t('未知任务', 'unknown task')}: {args.task}")
        print(f"{_t('支持任务', 'supported tasks')}: {task_ids + ['all']}")
        sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(args.results_root)

    if args.task == "all":
        selected = task_ids
    else:
        selected = [args.task]

    aggregate: dict[str, dict] = {
        "run_id": run_id,
        "result_layout": "model_task_eval_set_latest_runs",
        "selected_tasks": selected,
        "model_name": None,
        "eval_sets": {},
        "tasks": {},
        "skipped": {},
        "validation_errors": {},
    }
    task_result_dirs: dict[str, Path] = {}

    for task_id in selected:
        spec = registry[task_id]
        if args.task == "all" and (not args.include_disabled) and (not spec.enabled):
            aggregate["skipped"][task_id] = "disabled in registry"
            print(f"[SKIP] {task_id}: {aggregate['skipped'][task_id]}")
            continue

        output_dir = _resolve_output_dir(args, task_id)
        samples_file = spec.default_samples_file if args.task == "all" else (args.samples_file or spec.default_samples_file)
        model_base = args.output_root if args.task == "all" else (args.output_dir or output_dir)
        model_name = _slug(args.model_name) if args.model_name else _infer_model_name(model_base)
        eval_set = _slug(args.eval_set) if args.eval_set else _infer_eval_set(task_id, output_dir, samples_file)
        dataset_version = args.dataset_version or _infer_dataset_version(samples_file)
        task_result_dir = _resolve_task_result_dir(args.results_root, model_name, task_id, eval_set)
        task_runs_dir = task_result_dir / "runs"
        _ensure_dir(task_runs_dir)
        results_file = task_runs_dir / f"{run_id}.json"
        task_result_dirs[task_id] = task_result_dir
        aggregate["model_name"] = model_name
        aggregate["eval_sets"][task_id] = eval_set

        print(f"\n=== Running: {task_id} ===")
        print(f"model_name   : {model_name}")
        print(f"output_dir   : {output_dir}")
        print(f"samples_file : {samples_file}")
        print(f"eval_set     : {eval_set}")
        print(f"dataset_ver  : {dataset_version}")
        print(f"results_file : {results_file}")

        preflight_errors = _preflight_errors_for_task(task_id, samples_file)
        if preflight_errors:
            aggregate["tasks"][task_id] = {
                "status": "preflight_error",
                "errors": preflight_errors,
                "model_name": model_name,
                "eval_set": eval_set,
                "dataset_version": dataset_version,
                "evaluator_version": EVALUATOR_VERSION,
            }
            print(f"[ERROR] {task_id} {_t('预检失败', 'preflight failed')}: {'; '.join(preflight_errors)}")
            continue

        try:
            summary = _call_task(
                spec,
                {
                    "output_dir": output_dir,
                    "results_file": results_file,
                    "samples_file": samples_file,
                },
            )
            _annotate_result_file(
                results_file,
                dataset_version=dataset_version,
                evaluator_version=EVALUATOR_VERSION,
                model_name=model_name,
                task_id=task_id,
                eval_set=eval_set,
                samples_file=samples_file,
            )
            error_stats = _summarize_detail_errors(results_file)
            if error_stats["error_count"] > 0:
                top_msg = ", ".join(
                    f"{x['count']}x {x['message']}" for x in error_stats["top_errors"]
                )
                print(
                    f"[WARN] {task_id} "
                    f"{_t('结果含样本级错误', 'result contains sample-level errors')} "
                    f"{error_stats['error_count']}: {top_msg}"
                )
            validation_errors = _task_validation_errors(
                summary,
                error_stats,
                strict=args.strict,
            )
            if validation_errors:
                aggregate["validation_errors"][task_id] = validation_errors
                print(f"[ERROR] {task_id} {_t('结果校验失败', 'result validation failed')}: {'; '.join(validation_errors)}")
            latest_file = task_result_dir / "latest.json"
            if results_file.exists():
                shutil.copy2(results_file, latest_file)
                print(f"latest_file  : {latest_file}")
            aggregate["tasks"][task_id] = {
                "status": "ok",
                "summary": summary,
                "error_stats": error_stats,
                "model_name": model_name,
                "eval_set": eval_set,
                "dataset_version": dataset_version,
                "evaluator_version": EVALUATOR_VERSION,
                "result_file": str(results_file),
                "latest_file": str(latest_file),
                "validation_errors": validation_errors,
            }
        except NotImplementedError as e:
            aggregate["tasks"][task_id] = {
                "status": "not_implemented",
                "error": str(e),
                "model_name": model_name,
                "eval_set": eval_set,
                "dataset_version": dataset_version,
                "evaluator_version": EVALUATOR_VERSION,
            }
            print(f"[WARN] {task_id} {_t('未实现', 'not implemented')}: {e}")
        except Exception as e:  # noqa: BLE001
            aggregate["tasks"][task_id] = {
                "status": "error",
                "error": str(e),
                "model_name": model_name,
                "eval_set": eval_set,
                "dataset_version": dataset_version,
                "evaluator_version": EVALUATOR_VERSION,
            }
            print(f"[ERROR] {task_id} {_t('评测失败', 'evaluation failed')}: {e}")

    if len(selected) == 1 and selected[0] in task_result_dirs:
        aggregate_dir = task_result_dirs[selected[0]]
    else:
        aggregate_model = _slug(args.model_name) if args.model_name else (
            _infer_model_name(args.output_root) if args.output_root else "unknown_model"
        )
        aggregate_set = _slug(args.eval_set) if args.eval_set else "mixed"
        aggregate_dir = args.results_root / aggregate_model / "aggregate" / aggregate_set

    aggregate_runs_dir = aggregate_dir / "runs"
    _ensure_dir(aggregate_runs_dir)
    aggregate_file = aggregate_runs_dir / f"{run_id}_aggregate_summary.json"
    task_statuses = {
        task_id: info.get("status")
        for task_id, info in aggregate["tasks"].items()
        if isinstance(info, dict)
    }
    fatal_statuses = {"preflight_error", "not_implemented", "error"}
    failed_status_tasks = {
        task_id: status
        for task_id, status in task_statuses.items()
        if status in fatal_statuses
    }
    exit_code = 1 if failed_status_tasks or aggregate["validation_errors"] else 0
    with aggregate_file.open("w", encoding="utf-8") as f:
        aggregate["evaluator_version"] = EVALUATOR_VERSION
        aggregate["status"] = "failed" if exit_code else "ok"
        aggregate["failed_status_tasks"] = failed_status_tasks
        aggregate["dataset_versions"] = {
            task_id: info.get("dataset_version")
            for task_id, info in aggregate["tasks"].items()
            if isinstance(info, dict) and info.get("dataset_version")
        }
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    latest_aggregate = aggregate_dir / "aggregate_summary.json"
    shutil.copy2(aggregate_file, latest_aggregate)
    print(f"\n{_t('已写入聚合结果', 'wrote aggregate result')}: {aggregate_file}")
    print(f"{_t('latest 聚合结果', 'latest aggregate result')}: {latest_aggregate}")
    if exit_code:
        if failed_status_tasks:
            print(f"[ERROR] {_t('任务级失败', 'task-level failures')}: {failed_status_tasks}")
        if aggregate["validation_errors"]:
            print(f"[ERROR] {_t('结果校验失败', 'result validation failed')}: {aggregate['validation_errors']}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
