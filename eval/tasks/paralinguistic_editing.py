"""
副语言编辑任务评估编排（Paralinguistic Editing Task Evaluator）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.resources.paralinguistic_taxonomy import ALL_EVENTS  # noqa: E402

SAMPLES_FILE = REPO_ROOT / "data" / "paralinguistic_editing" / "samples.jsonl"

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


def _judge_workers() -> int:
    try:
        return max(1, int(os.getenv("SPEECHEDITBENCH_JUDGE_WORKERS", "1")))
    except ValueError:
        return 1


def load_samples(samples_file: Path) -> list[dict]:
    with samples_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stats(items: list[dict]) -> dict:
    total = len(items)
    correct = sum(1 for row in items if row.get("correct"))
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "correct": correct,
        "target_success_rate": rate_or_none([row.get("target_success") for row in items]),
        "content_preservation_pass_rate": rate_or_none(
            [row.get("content_preservation_pass") for row in items]
        ),
        "joint_success_rate": rate_or_none([row.get("joint_success") for row in items]),
        "avg_content_preservation_error": mean_or_none(
            [row.get("content_preservation_error") for row in items]
        ),
        "avg_utmos": mean_or_none([row.get("utmos") for row in items]),
        "avg_source_output_speaker_similarity": mean_or_none(
            [row.get("source_output_speaker_similarity") for row in items]
        ),
    }


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*76}")
    print("  副语言编辑评估报告（Paralinguistic Editing）")
    print(f"{'='*76}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  Target：{(overall['target_success_rate'] or 0.0):.1%}  |"
        f"  Preservation：{(overall['content_preservation_pass_rate'] or 0.0):.1%}  |"
        f"  Joint：{(overall['joint_success_rate'] or 0.0):.1%}  |"
        f"  UTMOS：{(overall['avg_utmos'] or 0.0):.3f}"
    )

    if summary.get("by_operation"):
        print("\n  按操作方向：")
        for op, acc in summary["by_operation"].items():
            print(
                f"    {op:8s}"
                f"  target={acc['target_success_rate']:.1%}"
                f"  joint={(acc['joint_success_rate'] or 0.0):.1%}"
            )

    if summary.get("by_event"):
        print("\n  按目标事件：")
        for event in ALL_EVENTS:
            acc = summary["by_event"].get(event)
            if acc is None:
                continue
            bar = "█" * int((acc["joint_success_rate"] or 0.0) * 20)
            print(
                f"    {event:8s}  {bar:<20s}"
                f"  target={acc['target_success_rate']:.1%}"
                f"  joint={(acc['joint_success_rate'] or 0.0):.1%}"
            )

    print(f"{'='*76}\n")


def batch_evaluate(
    samples: list[dict],
    output_audio_dir: Path,
    samples_file: Path,
) -> list[dict]:
    from eval.metrics.paralinguistic_accuracy import compute_pa, predict

    available = [
        (sample, audio_path)
        for sample in samples
        if (audio_path := find_output_audio(output_audio_dir, sample["sample_id"])) is not None
    ]
    progress_log("paralinguistic_editing", f"available outputs {len(available)}/{len(samples)}")

    try:
        progress_log("paralinguistic_editing", f"UTMOS start {len(available)}")
        utmos_by_sample_id = batch_utmos_scores(
            [(sample["sample_id"], audio_path) for sample, audio_path in available],
            device="auto",
        )
        progress_log("paralinguistic_editing", "UTMOS done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] UTMOS 批量计算失败，将逐条置空：{exc}")
        utmos_by_sample_id = {sample["sample_id"]: None for sample in samples}

    speaker_pairs = [
        {
            "sample_id": sample["sample_id"],
            "output_wav": str(audio_path),
            "reference_wav": str(resolve_sample_path(sample.get("audio_path", ""), samples_file)),
        }
        for sample, audio_path in available
    ]
    try:
        progress_log("paralinguistic_editing", f"source-output speaker similarity start {len(speaker_pairs)}")
        speaker_by_sample_id = batch_source_output_speaker_similarity(speaker_pairs, device="auto")
        progress_log("paralinguistic_editing", "source-output speaker similarity done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] source-output speaker similarity 批量计算失败，将逐条置空：{exc}")
        speaker_by_sample_id = {sample["sample_id"]: None for sample in samples}

    judge_workers = _judge_workers()
    predicted_by_sample_id: dict[str, dict[str, int]] = {}
    prediction_errors: dict[str, str] = {}
    if judge_workers > 1 and available:
        progress_log(
            "paralinguistic_editing",
            f"target judge start {len(available)} workers={judge_workers}",
        )

        def _predict_one(item: tuple[dict, Path]) -> tuple[str, dict[str, int]]:
            sample, audio_path = item
            return sample["sample_id"], predict(audio_path, sample.get("language", "en"))

        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            future_to_sid = {
                executor.submit(_predict_one, item): item[0]["sample_id"]
                for item in available
            }
            for done, future in enumerate(as_completed(future_to_sid), start=1):
                sid = future_to_sid[future]
                try:
                    sample_id, predicted = future.result()
                    predicted_by_sample_id[sample_id] = predicted
                except Exception as exc:  # noqa: BLE001
                    prediction_errors[sid] = str(exc)
                if should_log_progress(done, len(available)):
                    progress_log(
                        "paralinguistic_editing",
                        f"target judge {done}/{len(available)} sample={sid}",
                    )

    results = []
    progress_log("paralinguistic_editing", "judge/content preservation start")
    for index, (sample, audio_path) in enumerate(available, start=1):
        sid = sample["sample_id"]
        language = sample.get("language", "en")

        try:
            if judge_workers > 1:
                if sid in prediction_errors:
                    raise RuntimeError(prediction_errors[sid])
                predicted_scores = predicted_by_sample_id.get(sid, {})
            else:
                predicted_scores = predict(audio_path, language)
            result = compute_pa(sample, predicted_scores)
            preservation = content_preservation_metrics(audio_path, sample.get("transcript", ""), language)
            result.update(preservation)
            result.update({
                "target_success": bool(result.get("correct")),
                "joint_success": bool(
                    result.get("correct") and preservation.get("content_preservation_pass")
                ),
                "utmos": utmos_by_sample_id.get(sid),
                "source_output_speaker_similarity": speaker_by_sample_id.get(sid),
            })
        except NotImplementedError:
            raise
        except Exception as exc:
            anchor = sample.get("anchor", {})
            result = {
                "sample_id": sid,
                "operation": anchor.get("operation", ""),
                "target_event": anchor.get("event", ""),
                "predicted_score": None,
                "correct": False,
                "target_success": False,
                "content_preservation_metric": None,
                "content_preservation_error": None,
                "content_preservation_threshold": None,
                "content_preservation_pass": None,
                "joint_success": False,
                "transcript_target": sample.get("transcript", ""),
                "transcript_predicted": None,
                "norm_target": None,
                "norm_predicted": None,
                "wer": None,
                "cer": None,
                "utmos": utmos_by_sample_id.get(sid),
                "source_output_speaker_similarity": speaker_by_sample_id.get(sid),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "language": sample.get("language", "unknown"),
                "error": str(exc),
            }

        results.append(result)
        if should_log_progress(index, len(available)):
            progress_log("paralinguistic_editing", f"judge/content preservation {index}/{len(available)} sample={sid}")

    return results


def summarize(
    results: list[dict],
    *,
    total_samples: int | None = None,
    missing_outputs: int = 0,
) -> dict:
    by_op: dict[str, list] = defaultdict(list)
    by_ev: dict[str, list] = defaultdict(list)
    by_op_ev: dict[str, list] = defaultdict(list)
    by_ds: dict[str, list] = defaultdict(list)
    by_lang: dict[str, list] = defaultdict(list)

    for row in results:
        by_op[row["operation"]].append(row)
        by_ev[row["target_event"]].append(row)
        by_op_ev[f"{row['operation']}/{row['target_event']}"].append(row)
        by_ds[row.get("source_dataset", "unknown")].append(row)
        by_lang[row.get("language", "unknown")].append(row)

    summary = {
        "overall": _stats(results),
        "by_operation": {k: _stats(v) for k, v in sorted(by_op.items())},
        "by_event": {k: _stats(v) for k, v in sorted(by_ev.items())},
        "by_operation_event": {k: _stats(v) for k, v in sorted(by_op_ev.items())},
        "by_dataset": {k: _stats(v) for k, v in sorted(by_ds.items())},
        "by_language": {k: _stats(v) for k, v in sorted(by_lang.items())},
    }
    total = len(results) if total_samples is None else total_samples
    outputs_evaluated = len(results)
    summary["overall"].update({
        "total": total,
        "outputs_evaluated": outputs_evaluated,
        "missing_outputs": missing_outputs,
        "coverage": round(outputs_evaluated / total, 4) if total else 0.0,
        "error_count": sum(1 for row in results if row.get("error")),
    })
    return summary


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")

    available, missing = [], []
    for sample in samples:
        if find_output_audio(output_dir, sample["sample_id"]) is not None:
            available.append(sample)
        else:
            missing.append(sample["sample_id"])

    if missing:
        print(f"[WARN] {len(missing)} 条样本未找到模型输出，已跳过。")

    print(f"开始评估 {len(available)} 条样本...")
    results = batch_evaluate(available, output_dir, samples_file)
    summary = summarize(results, total_samples=len(samples), missing_outputs=len(missing))
    print_report(summary)

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{results_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paralinguistic editing outputs / 评估副语言编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/paralinguistic_editing/samples.jsonl) / 测试用例文件")
    args = parser.parse_args()

    run_evaluation(
        output_dir=args.output_dir,
        results_file=args.results_file,
        samples_file=args.samples_file,
    )


if __name__ == "__main__":
    main()
