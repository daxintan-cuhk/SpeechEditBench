"""
韵律编辑（Prosody Editing）任务评估编排。

输入：
  - data/prosody_editing/samples.jsonl
  - <output_dir>/<sample_id>.wav 或 <output_dir>/audio/<sample_id>.wav

评估指标：
  - speed: duration ratio direction accuracy
  - pitch: median F0 semitone-shift direction accuracy
  - stress: timestamp-ASR target-window prominence gain (RPG)
  - UTMOS: 官方 UTMOS22 demo strong learner 自然度评分
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "data" / "prosody_editing" / "samples.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.editing_common import (  # noqa: E402
    batch_source_output_speaker_similarity,
    batch_utmos_scores,
    content_preservation_metrics,
    find_output_audio,
    mean_or_none,
    progress_log,
    rate_or_none,
    resolve_sample_path,
    should_log_progress,
)


def load_samples(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_output_wav(output_dir: Path, sample_id: str) -> Path | None:
    """在输出目录中查找与 sample_id 匹配的音频文件。"""
    return find_output_audio(output_dir, sample_id)


def _mean(values: list[float | None]) -> float | None:
    return mean_or_none(values)


def _rate(values: list[bool | None]) -> float | None:
    return rate_or_none(values)


def _stats(items: list[dict]) -> dict:
    evaluated = [row for row in items if row.get("target_success") is not None]
    return {
        "total": len(items),
        "evaluated": len(evaluated),
        "pass_rate": _rate([row.get("joint_success") for row in evaluated]),
        "target_success_rate": _rate([row.get("target_success") for row in evaluated]),
        "content_preservation_pass_rate": _rate([
            row.get("content_preservation_pass") for row in items
        ]),
        "joint_success_rate": _rate([row.get("joint_success") for row in items]),
        "direction_accuracy": _rate([
            row.get("passed")
            for row in evaluated
            if row.get("prosody_type") in {"speed", "pitch"}
        ]),
        "avg_duration_ratio": _mean([row.get("duration_ratio") for row in evaluated]),
        "avg_f0_semitone_shift": _mean([row.get("shift_semitone") for row in evaluated]),
        "avg_rpg": _mean([row.get("rpg") for row in evaluated]),
        "avg_utmos": _mean([row.get("utmos") for row in items]),
        "avg_content_preservation_error": _mean([
            row.get("content_preservation_error") for row in items
        ]),
        "avg_source_output_speaker_similarity": _mean([
            row.get("source_output_speaker_similarity") for row in items
        ]),
        "error_count": sum(1 for row in items if row.get("error")),
    }


def _build_summary(total_samples: int, missing: list[str], details: list[dict]) -> dict:
    by_prosody_type: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by_prosody_type[row.get("prosody_type", "unknown")].append(row)
        by_language[row.get("language", "unknown")].append(row)
        by_dataset[row.get("source_dataset", "unknown")].append(row)

    overall_stats = _stats(details)
    return {
        "overall": {
            "total": total_samples,
            "outputs_evaluated": len(details),
            "missing_outputs": len(missing),
            "coverage": round(len(details) / total_samples, 4) if total_samples else 0.0,
            **{k: v for k, v in overall_stats.items() if k != "total"},
        },
        "by_prosody_type": {k: _stats(v) for k, v in sorted(by_prosody_type.items())},
        "by_language": {k: _stats(v) for k, v in sorted(by_language.items())},
        "by_dataset": {k: _stats(v) for k, v in sorted(by_dataset.items())},
    }


def _write_snapshot(
    results_file: Path,
    *,
    total_samples: int,
    missing: list[str],
    details: list[dict],
    processed: int,
    available_total: int,
) -> None:
    payload = {
        "summary": _build_summary(total_samples, missing, details),
        "details": details,
        "missing_outputs": missing,
        "progress": {
            "processed_outputs": processed,
            "available_outputs": available_total,
            "done": processed >= available_total,
        },
    }
    tmp_file = results_file.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_file.replace(results_file)


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*68}")
    print("  韵律编辑评估报告（Prosody Editing）")
    print(f"{'='*68}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  Target：{overall['target_success_rate']}  |"
        f"  Joint：{overall['joint_success_rate']}  |"
        f"  Preservation：{overall['content_preservation_pass_rate']}  |"
        f"  UTMOS：{overall['avg_utmos']}"
    )

    if summary.get("by_prosody_type"):
        print("\n  按子任务：")
        for ptype, stats in summary["by_prosody_type"].items():
            print(
                f"    {ptype:8s}  n={stats['total']:3d}"
                f"  evaluated={stats['evaluated']:3d}"
                f"  target={stats['target_success_rate']}"
                f"  joint={stats['joint_success_rate']}"
                f"  UTMOS={stats['avg_utmos']}"
            )
            if ptype == "speed":
                print(f"      avg_duration_ratio={stats['avg_duration_ratio']}")
            elif ptype == "pitch":
                print(f"      avg_f0_semitone_shift={stats['avg_f0_semitone_shift']}")
            elif ptype == "stress":
                print(f"      avg_rpg={stats['avg_rpg']}")

    print(f"{'='*68}\n")


def evaluate(
    samples: list[dict],
    samples_jsonl_path: Path,
    output_dir: Path,
    *,
    results_file: Path | None = None,
    device: str = "auto",
    verbose: bool = False,
    utmos_batch_size: int | None = None,
    metric_batch_size: int = 16,
    progress_every: int = 10,
) -> dict:
    from eval.metrics.prosody_accuracy import predict as prosody_predict

    available: list[tuple[dict, Path]] = []
    missing: list[str] = []

    for sample in samples:
        output_wav = find_output_wav(output_dir, sample["sample_id"])
        if output_wav is None:
            missing.append(sample["sample_id"])
        else:
            available.append((sample, output_wav))
    progress_log("prosody_editing", f"available outputs {len(available)}/{len(samples)}")

    if missing and verbose:
        for sample_id in missing:
            print(f"[缺失] {sample_id}")

    details: list[dict] = []
    total_available = len(available)
    try:
        progress_log("prosody_editing", f"UTMOS start {len(available)}")
        utmos_by_sample_id = batch_utmos_scores(
            [(sample["sample_id"], output_wav) for sample, output_wav in available],
            device=device,
            batch_size=utmos_batch_size,
        )
        progress_log("prosody_editing", "UTMOS done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] UTMOS 批量计算失败，将逐条置空：{exc}")
        utmos_by_sample_id = {sample["sample_id"]: None for sample, _ in available}

    speaker_pairs = [
        {
            "sample_id": sample["sample_id"],
            "output_wav": str(output_wav),
            "reference_wav": str(resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)),
        }
        for sample, output_wav in available
    ]
    try:
        progress_log("prosody_editing", f"source-output speaker similarity start {len(speaker_pairs)}")
        speaker_by_sample_id = batch_source_output_speaker_similarity(speaker_pairs, device=device)
        progress_log("prosody_editing", "source-output speaker similarity done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] source-output speaker similarity 批量计算失败，将逐条置空：{exc}")
        speaker_by_sample_id = {sample["sample_id"]: None for sample, _ in available}

    processed = 0
    progress_log("prosody_editing", "prosody/content preservation start")
    for batch_start in range(0, total_available, metric_batch_size):
        batch = available[batch_start:batch_start + metric_batch_size]

        for sample, output_wav in batch:
            sample_id = sample["sample_id"]
            anchor = sample.get("anchor", {})
            language = sample.get("language", "en")
            source_wav = resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
            prosody_type = anchor.get("prosody_type", "unknown")

            row = {
                "sample_id": sample_id,
                "prosody_type": prosody_type,
                "language": language,
                "source_dataset": sample.get("source_dataset", "unknown"),
                "direction": anchor.get("direction"),
                "stress_words": anchor.get("stress_words", []),
                "source_wav": str(source_wav),
                "output_wav": str(output_wav),
                "utmos": utmos_by_sample_id.get(sample_id),
                "source_output_speaker_similarity": speaker_by_sample_id.get(sample_id),
                "target_success": None,
                "content_preservation_metric": None,
                "content_preservation_error": None,
                "content_preservation_threshold": None,
                "content_preservation_pass": None,
                "joint_success": None,
                "passed": None,
                "error": None,
            }

            try:
                result = prosody_predict(
                    source_wav,
                    output_wav,
                    anchor,
                    input_transcript=sample.get("transcript"),
                    language=language,
                )
                row.update(result)
                preservation = content_preservation_metrics(
                    output_wav,
                    sample.get("transcript", ""),
                    language,
                )
                row.update(preservation)
                row["target_success"] = row.get("passed")
                row["joint_success"] = (
                    bool(row.get("target_success") and row.get("content_preservation_pass"))
                    if row.get("content_preservation_pass") is not None
                    else None
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                print(f"[错误] {sample_id}: {exc}")

            details.append(row)
            processed += 1
            if processed == 1 or processed % progress_every == 0 or processed == total_available:
                progress_log(
                    "prosody_editing",
                    f"prosody/content preservation {processed}/{total_available} "
                    f"sample={sample_id} prosody_type={prosody_type}",
                )
            if results_file and (processed % progress_every == 0 or processed == total_available):
                _write_snapshot(
                    results_file,
                    total_samples=len(samples),
                    missing=missing,
                    details=details,
                    processed=processed,
                    available_total=total_available,
                )

    summary = _build_summary(len(samples), missing, details)
    return {"summary": summary, "details": details, "missing_outputs": missing}


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")
    print("开始评估 Prosody DA/RPG + UTMOS...")

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        samples=samples,
        samples_jsonl_path=samples_file,
        output_dir=output_dir,
        results_file=results_file,
        device="auto",
        verbose=False,
        utmos_batch_size=(
            max(1, int(os.getenv("PROSODY_UTMOS_BATCH_SIZE")))
            if os.getenv("PROSODY_UTMOS_BATCH_SIZE")
            else None
        ),
        metric_batch_size=max(1, int(os.getenv("PROSODY_METRIC_BATCH_SIZE", "16"))),
        progress_every=max(1, int(os.getenv("PROSODY_PROGRESS_EVERY", "10"))),
    )
    print_report(result["summary"])

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{results_file}")

    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prosody editing outputs / 评估韵律编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/prosody_editing/samples.jsonl) / 测试用例文件")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.samples_file)
    result = evaluate(
        samples=samples,
        samples_jsonl_path=args.samples_file,
        output_dir=args.output_dir,
        device="auto",
        verbose=args.verbose,
        results_file=args.results_file,
        utmos_batch_size=(
            max(1, int(os.getenv("PROSODY_UTMOS_BATCH_SIZE")))
            if os.getenv("PROSODY_UTMOS_BATCH_SIZE")
            else None
        ),
        metric_batch_size=max(1, int(os.getenv("PROSODY_METRIC_BATCH_SIZE", "16"))),
        progress_every=max(1, int(os.getenv("PROSODY_PROGRESS_EVERY", "10"))),
    )
    print_report(result["summary"])

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.results_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{args.results_file}")


if __name__ == "__main__":
    main()
