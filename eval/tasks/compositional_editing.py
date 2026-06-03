"""
组合编辑任务评估编排（Compositional Editing Evaluator）。

输入：
  - data/compositional_editing/samples.jsonl 或 cache/smoke_samples.jsonl
  - <output_dir>/<sample_id>.wav 或 <output_dir>/audio/<sample_id>.wav

核心指标：
  - component_success: 每个子编辑目标是否成功
  - joint_success: 所有 component 是否同时成功
  - preservation_success: 非目标 content 是否保持

该 evaluator 复用已有原子任务 metric，不引入新的底层模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "data" / "compositional_editing" / "samples.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.editing_common import (  # noqa: E402
    SPEAKER_PRESERVATION_THRESHOLD,
    SPEAKER_TARGET_THRESHOLD,
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
    return find_output_audio(output_dir, sample_id)


def _mean(values: list[float | None]) -> float | None:
    return mean_or_none(values)


def _rate(values: list[bool | None]) -> float | None:
    return rate_or_none(values)


def _expected_transcript(sample: dict) -> str:
    for component in sample.get("components", []):
        if component.get("component_type") == "content":
            return component.get("anchor", {}).get("transcript_target") or sample.get("transcript", "")
    return sample.get("transcript", "")


def _component_sample(sample: dict, component: dict) -> dict:
    """Materialize the minimum atomic-task-like sample needed by metric helpers."""
    return {
        "sample_id": f"{sample['sample_id']}::{component['component_type']}",
        "task": component.get("edit_type"),
        "audio_path": sample.get("audio_path"),
        "instruction": sample.get("instruction"),
        "transcript": sample.get("transcript"),
        "anchor": component.get("anchor", {}),
        "duration_tag": sample.get("duration_tag"),
        "language": sample.get("language", "en"),
        "source_dataset": component.get("source_dataset", sample.get("source_dataset", "unknown")),
        "subset": sample.get("subset", "compositional"),
        "emotion_taxonomy": component.get("emotion_taxonomy")
                            or component.get("anchor", {}).get("emotion_taxonomy"),
        "benchmark_version": sample.get("benchmark_version"),
    }


def _content_component(
    sample: dict,
    component: dict,
    output_wav: Path,
) -> dict:
    from eval.metrics.content_accuracy import asr_predict, compute_content_accuracy

    language = sample.get("language", "en")
    pred_transcript = asr_predict(output_wav, language)
    atomic_sample = _component_sample(sample, component)
    result = compute_content_accuracy(atomic_sample, pred_transcript)
    result.update({
        "component_type": "content",
        "success": bool(result.get("edit_success")),
        "target_success": bool(result.get("edit_success")),
        "success_metric": "edit_success",
    })
    return result


def _speaker_component(
    sample: dict,
    component: dict,
    output_wav: Path,
    samples_jsonl_path: Path,
) -> dict:
    from eval.metrics.speaker_similarity import predict as ss_predict

    reference_value = component.get("reference_audio_path") or component.get("anchor", {}).get("reference_wav")
    if not reference_value:
        raise ValueError("speaker component missing reference_audio_path/reference_wav")
    reference_wav = resolve_sample_path(reference_value, samples_jsonl_path)
    score = round(ss_predict(output_wav, reference_wav, device="auto"), 4)
    anchor = component.get("anchor", {})
    return {
        "component_type": "speaker",
        "target_speaker": anchor.get("target_speaker") or anchor.get("reference_speaker_id"),
        "reference_wav": str(reference_wav),
        "speaker_similarity": score,
        "threshold": SPEAKER_TARGET_THRESHOLD,
        "success": score >= SPEAKER_TARGET_THRESHOLD,
        "target_success": score >= SPEAKER_TARGET_THRESHOLD,
        "success_metric": "speaker_similarity",
    }


def _speaker_preservation(
    sample: dict,
    output_wav: Path,
    samples_jsonl_path: Path,
    speaker_similarity: float | None = None,
) -> dict:
    source_wav = resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
    score = speaker_similarity
    if score is None:
        from eval.metrics.speaker_similarity import predict as ss_predict

        score = round(ss_predict(output_wav, source_wav, device="auto"), 4)
    return {
        "preservation_type": "speaker",
        "source_wav": str(source_wav),
        "speaker_similarity": score,
        "threshold": SPEAKER_PRESERVATION_THRESHOLD,
        "success": score >= SPEAKER_PRESERVATION_THRESHOLD if score is not None else None,
    }


def _emotion_component(
    sample: dict,
    component: dict,
    output_wav: Path,
) -> dict:
    from eval.metrics.emotion_accuracy import compute_emotion_accuracy, predict

    anchor = component.get("anchor", {})
    target = anchor.get("target_emotion")
    atomic_sample = _component_sample(sample, component)
    atomic_sample["anchor"] = {
        "type": "emotion",
        "source_emotion": anchor.get("source_emotion", "unknown"),
        "target_emotion": target,
        "emotion_taxonomy": anchor.get("emotion_taxonomy") or component.get("emotion_taxonomy"),
    }
    predicted = predict(
        output_wav,
        sample.get("language", "en"),
        sample.get("instruction", ""),
        _expected_transcript(sample),
        atomic_sample,
    )
    result = compute_emotion_accuracy(atomic_sample, predicted)
    result.update({
        "component_type": "emotion",
        "success": bool(result.get("correct")),
        "target_success": bool(result.get("correct")),
        "success_metric": "emotion_accuracy",
    })
    return result


def _prosody_component(
    sample: dict,
    component: dict,
    output_wav: Path,
    samples_jsonl_path: Path,
) -> dict:
    from eval.metrics.prosody_accuracy import predict as prosody_predict

    source_wav = resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
    anchor = component.get("anchor", {})
    result = prosody_predict(
        source_wav,
        output_wav,
        anchor,
        input_transcript=sample.get("transcript"),
        language=sample.get("language", "en"),
    )
    result.update({
        "component_type": "prosody",
        "prosody_type": anchor.get("prosody_type"),
        "direction": anchor.get("direction"),
        "source_wav": str(source_wav),
        "success": result.get("passed"),
        "target_success": result.get("passed"),
        "success_metric": "prosody_passed",
    })
    return result


def _acoustic_component(
    sample: dict,
    component: dict,
    output_wav: Path,
    samples_jsonl_path: Path,
) -> dict:
    from eval.metrics.acoustic_accuracy import (
        acoustic_scene_predict,
        compute_acoustic_accuracy,
        pesq_predict,
        rt60_measure,
        stoi_predict,
    )
    from eval.metrics.content_accuracy import asr_predict, compute_cer, compute_wer, normalize
    from eval.metrics.dnsmos import predict as dnsmos_predict

    anchor = component.get("anchor", {})
    source_wav = resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
    atomic_sample = _component_sample(sample, component)
    kwargs: dict = {}

    if anchor.get("subtask") == "enhancement":
        output_dnsmos = dnsmos_predict(output_wav)
        source_dnsmos = dnsmos_predict(source_wav)
        language = sample.get("language", "en")
        transcript_target = sample.get("transcript", "")
        transcript_predicted = asr_predict(output_wav, language)
        norm_target = normalize(transcript_target, language)
        norm_predicted = normalize(transcript_predicted, language)
        wer = compute_wer(norm_target, norm_predicted)
        cer = compute_cer(norm_target, norm_predicted)

        kwargs.update({
            "dnsmos_sig": output_dnsmos.get("SIG"),
            "dnsmos_bak": output_dnsmos.get("BAK"),
            "dnsmos_ovrl": output_dnsmos.get("OVRL"),
            "source_dnsmos_sig": source_dnsmos.get("SIG"),
            "source_dnsmos_bak": source_dnsmos.get("BAK"),
            "source_dnsmos_ovrl": source_dnsmos.get("OVRL"),
            "transcript_target": transcript_target,
            "transcript_predicted": transcript_predicted,
            "norm_target": norm_target,
            "norm_predicted": norm_predicted,
            "wer": wer,
            "cer": cer,
            "content_preservation_error": cer if language == "zh" else wer,
        })

        reference_value = anchor.get("target_reference_path")
        if reference_value:
            reference_wav = resolve_sample_path(reference_value, samples_jsonl_path)
            try:
                kwargs["pesq"] = pesq_predict(output_wav, reference_wav)
                kwargs["stoi"] = stoi_predict(output_wav, reference_wav)
                kwargs["source_pesq"] = pesq_predict(source_wav, reference_wav)
                kwargs["source_stoi"] = stoi_predict(source_wav, reference_wav)
            except Exception:
                pass
    elif anchor.get("subtask") == "env_transfer" and anchor.get("env_type") == "reverb":
        kwargs["rt60_measured"] = rt60_measure(output_wav, source_wav)
    elif anchor.get("subtask") == "env_transfer" and anchor.get("env_type") == "noise":
        kwargs["scene_scores"] = acoustic_scene_predict(output_wav, device="auto")
        kwargs["source_scene_scores"] = acoustic_scene_predict(source_wav, device="auto")

    result = compute_acoustic_accuracy(atomic_sample, **kwargs)
    result.update({
        "component_type": "acoustic",
        "source_wav": str(source_wav),
        "success": result.get("passed"),
        "target_success": result.get("passed"),
        "success_metric": "acoustic_passed",
    })
    return result


def _evaluate_component(
    sample: dict,
    component: dict,
    output_wav: Path,
    samples_jsonl_path: Path,
) -> dict:
    ctype = component.get("component_type")
    try:
        if ctype == "content":
            return _content_component(sample, component, output_wav)
        if ctype == "speaker":
            return _speaker_component(sample, component, output_wav, samples_jsonl_path)
        if ctype == "emotion":
            return _emotion_component(sample, component, output_wav)
        if ctype == "prosody":
            return _prosody_component(sample, component, output_wav, samples_jsonl_path)
        if ctype == "acoustic":
            return _acoustic_component(sample, component, output_wav, samples_jsonl_path)
        raise ValueError(f"unknown component_type: {ctype}")
    except Exception as exc:  # noqa: BLE001
        return {
            "component_type": ctype,
            "success": None,
            "target_success": None,
            "error": str(exc),
        }


def _evaluate_preservation(
    sample: dict,
    component_types: set[str],
    output_wav: Path,
    samples_jsonl_path: Path,
    source_output_speaker_similarity: float | None = None,
) -> list[dict]:
    checks: list[dict] = []
    if "content" not in component_types and sample.get("transcript"):
        try:
            preservation = content_preservation_metrics(
                output_wav,
                sample.get("transcript", ""),
                sample.get("language", "en"),
            )
            checks.append({
                "preservation_type": "content",
                "language": sample.get("language", "en"),
                "transcript_target": preservation.get("transcript_target"),
                "transcript_predicted": preservation.get("transcript_predicted"),
                "norm_target": preservation.get("norm_target"),
                "norm_predicted": preservation.get("norm_predicted"),
                "wer": preservation.get("wer"),
                "cer": preservation.get("cer"),
                "content_preservation_metric": preservation.get("content_preservation_metric"),
                "primary_error": preservation.get("content_preservation_error"),
                "threshold": preservation.get("content_preservation_threshold"),
                "success": preservation.get("content_preservation_pass"),
            })
        except Exception as exc:  # noqa: BLE001
            checks.append({"preservation_type": "content", "success": None, "error": str(exc)})

    return checks


def evaluate(
    samples: list[dict],
    samples_jsonl_path: Path,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> dict:

    available: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for sample in samples:
        output_wav = find_output_wav(output_dir, sample["sample_id"])
        if output_wav is None:
            missing.append(sample["sample_id"])
        else:
            available.append((sample, output_wav))
    progress_log("compositional_editing", f"available outputs {len(available)}/{len(samples)}")

    if missing and verbose:
        for sample_id in missing:
            print(f"[缺失] {sample_id}")

    try:
        progress_log("compositional_editing", f"UTMOS start {len(available)}")
        utmos_by_sample_id = batch_utmos_scores(
            [(sample["sample_id"], output_wav) for sample, output_wav in available],
            device="auto",
        )
        progress_log("compositional_editing", "UTMOS done")
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
        progress_log("compositional_editing", f"source-output speaker similarity start {len(speaker_pairs)}")
        speaker_by_sample_id = batch_source_output_speaker_similarity(speaker_pairs, device="auto")
        progress_log("compositional_editing", "source-output speaker similarity done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] source-output speaker similarity 批量计算失败，将逐条置空：{exc}")
        speaker_by_sample_id = {sample["sample_id"]: None for sample, _ in available}

    details: list[dict] = []
    progress_log("compositional_editing", "component + preservation evaluation start")
    for index, (sample, output_wav) in enumerate(available, start=1):
        components = sample.get("components", [])
        component_types = {component.get("component_type") for component in components}
        component_results = [
            _evaluate_component(sample, component, output_wav, samples_jsonl_path)
            for component in components
        ]
        preservation_results = _evaluate_preservation(
            sample,
            component_types,
            output_wav,
            samples_jsonl_path,
            speaker_by_sample_id.get(sample["sample_id"]),
        )

        component_success_values = [row.get("success") for row in component_results]
        preservation_success_values = [row.get("success") for row in preservation_results]
        preservation_success = (
            all(value is True for value in preservation_success_values)
            if preservation_success_values and all(value is not None for value in preservation_success_values)
            else None
        )
        preservation_ok_for_joint = True if not preservation_success_values else preservation_success
        joint_success = (
            all(value is True for value in component_success_values) and preservation_ok_for_joint is True
            if all(value is not None for value in component_success_values)
            and preservation_ok_for_joint is not None
            else None
        )

        details.append({
            "sample_id": sample["sample_id"],
            "language": sample.get("language", "unknown"),
            "source_dataset": sample.get("source_dataset", "unknown"),
            "difficulty": sample.get("difficulty", "unknown"),
            "combo_type": sample.get("combo_type", "unknown"),
            "base_component": sample.get("base_component"),
            "base_sample_id": sample.get("base_sample_id"),
            "output_wav": str(output_wav),
            "utmos": utmos_by_sample_id.get(sample["sample_id"]),
            "source_output_speaker_similarity": speaker_by_sample_id.get(sample["sample_id"]),
            "component_results": component_results,
            "preservation_results": preservation_results,
            "component_success_rate": _rate(component_success_values),
            "joint_success": joint_success,
            "preservation_success": preservation_success,
            "component_error_count": sum(1 for row in component_results if row.get("error")),
            "preservation_error_count": sum(1 for row in preservation_results if row.get("error")),
            "error": "; ".join(
                [row["error"] for row in [*component_results, *preservation_results] if row.get("error")]
            ) or None,
        })
        if should_log_progress(index, len(available)):
            progress_log(
                "compositional_editing",
                f"component + preservation evaluation {index}/{len(available)} "
                f"sample={sample['sample_id']}",
            )

    return {
        "summary": summarize(details, total_samples=len(samples), missing_outputs=len(missing)),
        "details": details,
        "missing_outputs": missing,
    }


def _stats(items: list[dict]) -> dict:
    component_success_values: list[bool | None] = []
    for row in items:
        for component in row.get("component_results", []):
            component_success_values.append(component.get("success"))

    preservation_success_values = [row.get("preservation_success") for row in items]
    joint_values = [row.get("joint_success") for row in items]
    return {
        "total": len(items),
        "evaluated": len(items),
        "joint_success_rate": _rate(joint_values),
        "component_success_rate": _rate(component_success_values),
        "preservation_success_rate": _rate(preservation_success_values),
        "avg_sample_component_success_rate": _mean([row.get("component_success_rate") for row in items]),
        "avg_utmos": _mean([row.get("utmos") for row in items]),
        "avg_source_output_speaker_similarity": _mean([
            row.get("source_output_speaker_similarity") for row in items
        ]),
        "error_count": sum(1 for row in items if row.get("error")),
    }


def summarize(
    details: list[dict],
    *,
    total_samples: int,
    missing_outputs: int,
) -> dict:
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    by_combo_type: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    by_combo_language: dict[str, list[dict]] = defaultdict(list)

    for row in details:
        by_difficulty[row.get("difficulty", "unknown")].append(row)
        by_combo_type[row.get("combo_type", "unknown")].append(row)
        by_language[row.get("language", "unknown")].append(row)
        by_combo_language[f"{row.get('combo_type', 'unknown')}::{row.get('language', 'unknown')}"].append(row)

    overall = _stats(details)
    overall.update({
        "total": total_samples,
        "outputs_evaluated": len(details),
        "missing_outputs": missing_outputs,
        "coverage": round(len(details) / total_samples, 4) if total_samples else 0.0,
    })
    return {
        "overall": overall,
        "by_difficulty": {k: _stats(v) for k, v in sorted(by_difficulty.items())},
        "by_combo_type": {k: _stats(v) for k, v in sorted(by_combo_type.items())},
        "by_language": {k: _stats(v) for k, v in sorted(by_language.items())},
        "by_combo_language": {k: _stats(v) for k, v in sorted(by_combo_language.items())},
    }


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*76}")
    print("  组合编辑评估报告（Compositional Editing）")
    print(f"{'='*76}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  Joint：{overall['joint_success_rate']}  |"
        f"  Component：{overall['component_success_rate']}  |"
        f"  Preservation：{overall['preservation_success_rate']}  |"
        f"  UTMOS：{overall['avg_utmos']}"
    )

    if summary.get("by_difficulty"):
        print("\n  按难度：")
        for key, stats in summary["by_difficulty"].items():
            print(
                f"    {key:10s}  n={stats['total']:3d}"
                f"  joint={stats['joint_success_rate']}"
                f"  component={stats['component_success_rate']}"
                f"  preservation={stats['preservation_success_rate']}"
            )

    if summary.get("by_combo_type"):
        print("\n  按组合类型：")
        for key, stats in summary["by_combo_type"].items():
            print(
                f"    {key:28s}  n={stats['total']:3d}"
                f"  joint={stats['joint_success_rate']}"
                f"  component={stats['component_success_rate']}"
                f"  preservation={stats['preservation_success_rate']}"
            )

    print(f"{'='*76}\n")


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")
    print("开始评估 compositional component/joint/preservation...")

    result = evaluate(
        samples=samples,
        samples_jsonl_path=samples_file,
        output_dir=output_dir,
        verbose=False,
    )
    print_report(result["summary"])

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{results_file}")

    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compositional editing outputs / 评估组合编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/compositional_editing/samples.jsonl) / 测试用例文件")
    args = parser.parse_args()

    run_evaluation(
        output_dir=args.output_dir,
        results_file=args.results_file,
        samples_file=args.samples_file,
    )


if __name__ == "__main__":
    main()
