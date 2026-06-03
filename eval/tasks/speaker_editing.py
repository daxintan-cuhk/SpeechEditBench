"""
说话人编辑（Voice Conversion）任务评估编排。

输入：
  - data/speaker_editing/samples.jsonl
  - <output_dir>/<sample_id>.wav 或 <output_dir>/audio/<sample_id>.wav

评估指标：
  - Speaker Similarity: WavLM large speaker verification embedding cosine similarity
  - WER/CER: 英文 Whisper large-v3，中文 Paraformer zh，检查原文内容是否保留
  - UTMOS: 官方 UTMOS22 demo strong learner 自然度评分
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "data" / "speaker_editing" / "samples.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.editing_common import (  # noqa: E402
    SPEAKER_TARGET_THRESHOLD,
    content_preservation_threshold,
    find_output_audio,
    progress_log,
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
    valid = [value for value in values if value is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _content_preservation_error(row: dict) -> float | None:
    """Use English WER and Chinese CER as the language-appropriate preservation error."""
    if row.get("language") == "zh":
        return row.get("cer")
    return row.get("wer")


def _stats(items: list[dict]) -> dict:
    return {
        "total": len(items),
        "target_success_rate": _mean([1.0 if row.get("target_success") else 0.0 for row in items if row.get("target_success") is not None]),
        "content_preservation_pass_rate": _mean([
            1.0 if row.get("content_preservation_pass") else 0.0
            for row in items
            if row.get("content_preservation_pass") is not None
        ]),
        "joint_success_rate": _mean([1.0 if row.get("joint_success") else 0.0 for row in items if row.get("joint_success") is not None]),
        "avg_speaker_similarity": _mean([row.get("speaker_similarity") for row in items]),
        "avg_en_wer": _mean([row.get("wer") for row in items if row.get("language") == "en"]),
        "avg_zh_cer": _mean([row.get("cer") for row in items if row.get("language") == "zh"]),
        "avg_content_preservation_error": _mean([_content_preservation_error(row) for row in items]),
        "avg_utmos": _mean([row.get("utmos") for row in items]),
    }


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*68}")
    print("  说话人编辑评估报告（Speaker Editing）")
    print(f"{'='*68}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  Target：{overall['target_success_rate']}  |"
        f"  Joint：{overall['joint_success_rate']}  |"
        f"  SS：{overall['avg_speaker_similarity']}  |"
        f"  内容保留错误率：{overall['avg_content_preservation_error']}  |"
        f"  UTMOS：{overall['avg_utmos']}"
    )
    print(
        "  内容保留错误率说明：英文样本用 WER，中文样本用 CER；"
        "不汇总跨语言 WER。"
    )

    if summary.get("by_language"):
        print("\n  按语言：")
        for lang, stats in summary["by_language"].items():
            primary = "WER" if lang == "en" else "CER"
            primary_val = stats["avg_en_wer"] if lang == "en" else stats["avg_zh_cer"]
            print(
                f"    {lang:6s}  n={stats['total']:3d}"
                f"  target={stats['target_success_rate']}"
                f"  joint={stats['joint_success_rate']}"
                f"  SS={stats['avg_speaker_similarity']}"
                f"  {primary}={primary_val}"
                f"  UTMOS={stats['avg_utmos']}"
            )

    if summary.get("by_dataset"):
        print("\n  按数据集：")
        for dataset, stats in summary["by_dataset"].items():
            print(
                f"    {dataset:16s}  n={stats['total']:3d}"
                f"  target={stats['target_success_rate']}"
                f"  joint={stats['joint_success_rate']}"
                f"  SS={stats['avg_speaker_similarity']}"
                f"  content_preservation_error={stats['avg_content_preservation_error']}"
                f"  UTMOS={stats['avg_utmos']}"
            )
    print(f"{'='*68}\n")


def evaluate(
    samples: list[dict],
    samples_jsonl_path: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    verbose: bool = False,
) -> dict:
    from eval.metrics.content_accuracy import asr_predict, compute_cer, compute_wer, normalize
    from eval.metrics.speaker_similarity import batch_predict as ss_batch_predict
    from eval.metrics.utmos import predict_many as utmos_predict_many

    available: list[tuple[dict, Path]] = []
    missing: list[str] = []

    for sample in samples:
        output_wav = find_output_wav(output_dir, sample["sample_id"])
        if output_wav is None:
            missing.append(sample["sample_id"])
        else:
            available.append((sample, output_wav))
    progress_log("speaker_editing", f"available outputs {len(available)}/{len(samples)}")

    if missing and verbose:
        for sample_id in missing:
            print(f"[缺失] {sample_id}")

    utmos_by_sample_id: dict[str, float | None] = {}
    if available:
        try:
            progress_log("speaker_editing", f"UTMOS start {len(available)}")
            utmos_scores = utmos_predict_many([output_wav for _, output_wav in available], device=device)
            utmos_by_sample_id = {
                sample["sample_id"]: round(score, 4)
                for (sample, _), score in zip(available, utmos_scores, strict=True)
            }
            progress_log("speaker_editing", "UTMOS done")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] UTMOS 批量计算失败，将逐条记录错误：{exc}")
            utmos_by_sample_id = {sample["sample_id"]: None for sample, _ in available}

    speaker_pairs: list[dict] = []
    reference_by_sample_id: dict[str, str | None] = {}
    speaker_errors: dict[str, str] = {}
    for sample, output_wav in available:
        sample_id = sample["sample_id"]
        anchor = sample.get("anchor", {})
        reference_value = sample.get("reference_audio_path") or anchor.get("reference_wav")
        if not reference_value:
            reference_by_sample_id[sample_id] = None
            speaker_errors[sample_id] = "missing reference_audio_path/reference_wav"
            continue
        reference_wav = resolve_sample_path(reference_value, samples_jsonl_path)
        reference_by_sample_id[sample_id] = str(reference_wav)
        speaker_pairs.append({
            "sample_id": sample_id,
            "output_wav": str(output_wav),
            "reference_wav": str(reference_wav),
        })

    speaker_by_sample_id: dict[str, float | None] = {
        sample["sample_id"]: None for sample, _ in available
    }
    if speaker_pairs:
        try:
            progress_log("speaker_editing", f"speaker similarity start {len(speaker_pairs)}")
            speaker_scores = ss_batch_predict(speaker_pairs, device=device)
            speaker_by_sample_id.update({
                pair["sample_id"]: (round(score, 4) if score is not None else None)
                for pair, score in zip(speaker_pairs, speaker_scores, strict=True)
            })
            progress_log("speaker_editing", "speaker similarity done")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] speaker similarity 批量计算失败，将逐条置空：{exc}")

    details: list[dict] = []
    progress_log("speaker_editing", "ASR/content preservation start")
    for index, (sample, output_wav) in enumerate(available, start=1):
        sample_id = sample["sample_id"]
        language = sample.get("language", "en")
        anchor = sample.get("anchor", {})
        reference_value = sample.get("reference_audio_path") or anchor.get("reference_wav")
        target_transcript = sample.get("transcript", "")

        row = {
            "sample_id": sample_id,
            "language": language,
            "source_dataset": sample.get("source_dataset", "unknown"),
            "source_speaker": anchor.get("source_speaker", ""),
            "target_speaker": anchor.get("target_speaker", ""),
            "reference_speaker_id": anchor.get("reference_speaker_id", ""),
            "output_wav": str(output_wav),
            "reference_wav": None,
            "speaker_similarity": None,
            "transcript_target": target_transcript,
            "transcript_predicted": None,
            "norm_target": None,
            "norm_predicted": None,
            "wer": None,
            "cer": None,
            "utmos": utmos_by_sample_id.get(sample_id),
            "target_success": None,
            "content_preservation_metric": None,
            "content_preservation_error": None,
            "content_preservation_threshold": None,
            "content_preservation_pass": None,
            "joint_success": None,
            "error": None,
        }

        try:
            if not reference_value:
                raise ValueError("missing reference_audio_path/reference_wav")
            row["reference_wav"] = reference_by_sample_id.get(sample_id)
            if speaker_errors.get(sample_id):
                raise ValueError(speaker_errors[sample_id])
            row["speaker_similarity"] = speaker_by_sample_id.get(sample_id)

            transcript_predicted = asr_predict(output_wav, language)
            norm_target = normalize(target_transcript, language)
            norm_predicted = normalize(transcript_predicted, language)
            row.update(
                {
                    "transcript_predicted": transcript_predicted,
                    "norm_target": norm_target,
                    "norm_predicted": norm_predicted,
                    "wer": round(compute_wer(norm_target, norm_predicted), 4),
                    "cer": round(compute_cer(norm_target, norm_predicted), 4),
                }
            )
            row["target_success"] = (
                row["speaker_similarity"] >= SPEAKER_TARGET_THRESHOLD
                if row["speaker_similarity"] is not None
                else None
            )
            row["content_preservation_metric"] = "cer" if language == "zh" else "wer"
            row["content_preservation_error"] = _content_preservation_error(row)
            row["content_preservation_threshold"] = content_preservation_threshold(language)
            row["content_preservation_pass"] = (
                row["content_preservation_error"] is not None
                and row["content_preservation_error"] <= row["content_preservation_threshold"]
            )
            row["joint_success"] = (
                bool(row["target_success"] and row["content_preservation_pass"])
                if row["target_success"] is not None
                and row["content_preservation_pass"] is not None
                else None
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            print(f"[错误] {sample_id}: {exc}")

        details.append(row)
        if should_log_progress(index, len(available)):
            progress_log("speaker_editing", f"ASR/content preservation {index}/{len(available)} sample={sample_id}")

    by_language: dict[str, list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_target_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by_language[row.get("language", "unknown")].append(row)
        by_dataset[row.get("source_dataset", "unknown")].append(row)
        by_target_speaker[row.get("target_speaker", "unknown")].append(row)

    evaluated = len(details)
    overall_stats = _stats(details)
    summary = {
        "overall": {
            "total": len(samples),
            "outputs_evaluated": evaluated,
            "missing_outputs": len(missing),
            "coverage": round(evaluated / len(samples), 4) if samples else 0.0,
            "error_count": sum(1 for row in details if row.get("error")),
            "evaluated": evaluated,
            **{k: v for k, v in overall_stats.items() if k != "total"},
        },
        "by_language": {k: _stats(v) for k, v in sorted(by_language.items())},
        "by_dataset": {k: _stats(v) for k, v in sorted(by_dataset.items())},
        "by_target_speaker": {k: _stats(v) for k, v in sorted(by_target_speaker.items())},
    }
    return {"summary": summary, "details": details, "missing_outputs": missing}


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")
    print("开始评估 Speaker Similarity + WER/CER + UTMOS...")

    result = evaluate(
        samples=samples,
        samples_jsonl_path=samples_file,
        output_dir=output_dir,
        device="auto",
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
    parser = argparse.ArgumentParser(description="Evaluate speaker editing outputs / 评估说话人编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/speaker_editing/samples.jsonl) / 测试用例文件")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.samples_file)
    result = evaluate(
        samples=samples,
        samples_jsonl_path=args.samples_file,
        output_dir=args.output_dir,
        device="auto",
        verbose=args.verbose,
    )
    print_report(result["summary"])

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.results_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{args.results_file}")


if __name__ == "__main__":
    main()
