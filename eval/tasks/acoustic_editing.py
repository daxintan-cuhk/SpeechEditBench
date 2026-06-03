"""
Acoustic editing evaluation runner.

Inputs:
  - data/acoustic_editing/samples.jsonl
  - <output_dir>/<sample_id>.wav or <output_dir>/audio/<sample_id>.wav

Metrics:
  - enhancement: DNSMOS P.835 gain + content preservation, with optional PESQ/STOI diagnostics
  - env_transfer/reverb: RT60 target-range hit
  - env_transfer/noise: PANNs grouped acoustic scene accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "data" / "acoustic_editing" / "samples.jsonl"

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
    """Find output audio matching sample_id."""
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
        "avg_dnsmos_sig": _mean([row.get("dnsmos_sig") for row in items]),
        "avg_dnsmos_bak": _mean([row.get("dnsmos_bak") for row in items]),
        "avg_dnsmos_ovrl": _mean([row.get("dnsmos_ovrl") for row in items]),
        "avg_dnsmos_sig_gain": _mean([row.get("dnsmos_sig_gain") for row in items]),
        "avg_dnsmos_bak_gain": _mean([row.get("dnsmos_bak_gain") for row in items]),
        "avg_dnsmos_ovrl_gain": _mean([row.get("dnsmos_ovrl_gain") for row in items]),
        "avg_content_preservation_error": _mean([
            row.get("content_preservation_error") for row in items
        ]),
        "avg_pesq": _mean([row.get("pesq") for row in items]),
        "avg_stoi": _mean([row.get("stoi") for row in items]),
        "avg_pesq_gain": _mean([row.get("pesq_gain") for row in items]),
        "avg_stoi_gain": _mean([row.get("stoi_gain") for row in items]),
        "avg_utmos": _mean([row.get("utmos") for row in items]),
        "avg_source_output_speaker_similarity": _mean([
            row.get("source_output_speaker_similarity") for row in items
        ]),
        "avg_rt60": _mean([row.get("rt60_measured") for row in items]),
        "avg_target_scene_score": _mean([row.get("target_scene_score") for row in items]),
        "error_count": sum(1 for row in items if row.get("error")),
    }


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*68}")
    print("  声学编辑评估报告（Acoustic Editing）")
    print(f"{'='*68}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  Target：{overall['target_success_rate']}  |"
        f"  Preservation：{overall['content_preservation_pass_rate']}  |"
        f"  Joint：{overall['joint_success_rate']}"
    )

    if summary.get("by_subtask"):
        print("\n  按子任务：")
        for subtask, stats in summary["by_subtask"].items():
            print(
                f"    {subtask:14s}  n={stats['total']:3d}"
                f"  evaluated={stats['evaluated']:3d}"
                f"  target={stats['target_success_rate']}"
                f"  joint={stats['joint_success_rate']}"
                f"  OVRL_gain={stats['avg_dnsmos_ovrl_gain']}"
                f"  BAK_gain={stats['avg_dnsmos_bak_gain']}"
                f"  UTMOS={stats['avg_utmos']}"
                f"  RT60={stats['avg_rt60']}"
            )

    if summary.get("by_degradation_type"):
        print("\n  enhancement 按退化类型：")
        for dtype, stats in summary["by_degradation_type"].items():
            print(
                f"    {dtype:12s}  n={stats['total']:3d}"
                f"  pass_rate={stats['pass_rate']}"
                f"  DNSMOS_OVRL_gain={stats['avg_dnsmos_ovrl_gain']}"
                f"  DNSMOS_BAK_gain={stats['avg_dnsmos_bak_gain']}"
                f"  preservation={stats['avg_content_preservation_error']}"
                f"  PESQ={stats['avg_pesq']}"
                f"  STOI={stats['avg_stoi']}"
                f"  UTMOS={stats['avg_utmos']}"
            )

    if summary.get("by_env_subtype"):
        print("\n  env_transfer 按环境子类型：")
        for env_subtype, stats in summary["by_env_subtype"].items():
            print(
                f"    {env_subtype:12s}  n={stats['total']:3d}"
                f"  pass_rate={stats['pass_rate']}"
                f"  RT60={stats['avg_rt60']}"
                f"  target_scene_score={stats['avg_target_scene_score']}"
            )

    if summary.get("by_language"):
        print("\n  按语言：")
        for lang, stats in summary["by_language"].items():
            print(
                f"    {lang:6s}  n={stats['total']:3d}"
                f"  pass_rate={stats['pass_rate']}"
                f"  OVRL_gain={stats['avg_dnsmos_ovrl_gain']}"
                f"  preservation={stats['avg_content_preservation_error']}"
                f"  PESQ={stats['avg_pesq']}"
                f"  STOI={stats['avg_stoi']}"
                f"  RT60={stats['avg_rt60']}"
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
    from eval.metrics.acoustic_accuracy import (
        acoustic_scene_predict,
        compute_acoustic_accuracy,
        pesq_predict,
        rt60_measure,
        stoi_predict,
    )
    from eval.metrics.dnsmos import predict_many as dnsmos_predict_many

    available: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for sample in samples:
        output_wav = find_output_wav(output_dir, sample["sample_id"])
        if output_wav is None:
            missing.append(sample["sample_id"])
        else:
            available.append((sample, output_wav))
    progress_log("acoustic_editing", f"available outputs {len(available)}/{len(samples)}")

    if missing and verbose:
        for sample_id in missing:
            print(f"[缺失] {sample_id}")

    enhancement_pairs = [
        (sample, output_wav)
        for sample, output_wav in available
        if sample.get("anchor", {}).get("subtask") == "enhancement"
    ]
    try:
        progress_log("acoustic_editing", f"UTMOS start {len(available)}")
        utmos_by_sample_id = batch_utmos_scores(
            [(sample["sample_id"], output_wav) for sample, output_wav in available],
            device=device,
        )
        progress_log("acoustic_editing", "UTMOS done")
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
        progress_log("acoustic_editing", f"source-output speaker similarity start {len(speaker_pairs)}")
        speaker_by_sample_id = batch_source_output_speaker_similarity(speaker_pairs, device=device)
        progress_log("acoustic_editing", "source-output speaker similarity done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] source-output speaker similarity 批量计算失败，将逐条置空：{exc}")
        speaker_by_sample_id = {sample["sample_id"]: None for sample, _ in available}

    dnsmos_by_sample_id: dict[str, dict[str, float] | None] = {}
    source_dnsmos_by_sample_id: dict[str, dict[str, float] | None] = {}
    if enhancement_pairs:
        try:
            progress_log("acoustic_editing", f"DNSMOS output start {len(enhancement_pairs)}")
            output_scores = dnsmos_predict_many([output_wav for _, output_wav in enhancement_pairs])
            dnsmos_by_sample_id = {
                sample["sample_id"]: score
                for (sample, _), score in zip(enhancement_pairs, output_scores, strict=True)
            }
            progress_log("acoustic_editing", "DNSMOS output done")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] DNSMOS 输出评分失败，将逐条置空：{exc}")
            dnsmos_by_sample_id = {sample["sample_id"]: None for sample, _ in enhancement_pairs}

        try:
            progress_log("acoustic_editing", f"DNSMOS source start {len(enhancement_pairs)}")
            source_scores = dnsmos_predict_many([
                resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
                for sample, _ in enhancement_pairs
            ])
            source_dnsmos_by_sample_id = {
                sample["sample_id"]: score
                for (sample, _), score in zip(enhancement_pairs, source_scores, strict=True)
            }
            progress_log("acoustic_editing", "DNSMOS source done")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] DNSMOS 输入评分失败，将逐条置空：{exc}")
            source_dnsmos_by_sample_id = {
                sample["sample_id"]: None for sample, _ in enhancement_pairs
            }

    details: list[dict] = []
    progress_log("acoustic_editing", "target metrics + content preservation start")
    for index, (sample, output_wav) in enumerate(available, start=1):
        sample_id = sample["sample_id"]
        anchor = sample.get("anchor", {})
        subtask = anchor.get("subtask", "unknown")
        env_type = anchor.get("env_type")
        source_wav = resolve_sample_path(sample.get("audio_path", ""), samples_jsonl_path)
        row_context = {
            "source_wav": str(source_wav),
            "output_wav": str(output_wav),
            "target_reference_wav": None,
            "utmos": utmos_by_sample_id.get(sample_id),
            "source_output_speaker_similarity": speaker_by_sample_id.get(sample_id),
            "error": None,
        }

        kwargs: dict = {}
        errors: list[str] = []

        try:
            if subtask == "enhancement":
                ref_value = anchor.get("target_reference_path")
                kwargs["utmos"] = utmos_by_sample_id.get(sample_id)
                output_dnsmos = dnsmos_by_sample_id.get(sample_id) or {}
                source_dnsmos = source_dnsmos_by_sample_id.get(sample_id) or {}
                kwargs.update({
                    "dnsmos_sig": output_dnsmos.get("SIG"),
                    "dnsmos_bak": output_dnsmos.get("BAK"),
                    "dnsmos_ovrl": output_dnsmos.get("OVRL"),
                    "source_dnsmos_sig": source_dnsmos.get("SIG"),
                    "source_dnsmos_bak": source_dnsmos.get("BAK"),
                    "source_dnsmos_ovrl": source_dnsmos.get("OVRL"),
                })

                if ref_value:
                    reference_wav = resolve_sample_path(ref_value, samples_jsonl_path)
                    row_context["target_reference_wav"] = str(reference_wav)
                    try:
                        kwargs["pesq"] = pesq_predict(output_wav, reference_wav)
                        kwargs["stoi"] = stoi_predict(output_wav, reference_wav)
                        kwargs["source_pesq"] = pesq_predict(source_wav, reference_wav)
                        kwargs["source_stoi"] = stoi_predict(source_wav, reference_wav)
                    except Exception as exc:  # noqa: BLE001
                        row_context["reference_diagnostic_error"] = str(exc)

            elif subtask == "env_transfer" and env_type == "reverb":
                ref_value = anchor.get("target_reference_path")
                if ref_value:
                    row_context["target_reference_wav"] = str(
                        resolve_sample_path(ref_value, samples_jsonl_path)
                    )
                kwargs["rt60_measured"] = rt60_measure(output_wav, source_wav)

            elif subtask == "env_transfer" and env_type == "noise":
                ref_value = anchor.get("target_reference_path")
                if ref_value:
                    row_context["target_reference_wav"] = str(
                        resolve_sample_path(ref_value, samples_jsonl_path)
                    )
                kwargs["scene_scores"] = acoustic_scene_predict(output_wav, device=device)
                kwargs["source_scene_scores"] = acoustic_scene_predict(source_wav, device=device)

        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            print(f"[错误] {sample_id}: {exc}")

        row = compute_acoustic_accuracy(sample, **kwargs)
        try:
            preservation = content_preservation_metrics(
                output_wav,
                sample.get("transcript", ""),
                sample.get("language", "en"),
            )
            row.update(preservation)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"content_preservation: {exc}")
            row.update({
                "transcript_target": sample.get("transcript", ""),
                "transcript_predicted": None,
                "norm_target": None,
                "norm_predicted": None,
                "wer": None,
                "cer": None,
                "content_preservation_metric": None,
                "content_preservation_error": None,
                "content_preservation_threshold": None,
                "content_preservation_pass": None,
            })

        row["target_success"] = row.get("passed")
        row["joint_success"] = (
            bool(row["target_success"] and row.get("content_preservation_pass"))
            if row["target_success"] is not None
            and row.get("content_preservation_pass") is not None
            else None
        )
        row.update(row_context)
        if errors:
            row["error"] = "; ".join(errors)
        details.append(row)
        if should_log_progress(index, len(available)):
            progress_log(
                "acoustic_editing",
                f"target metrics + content preservation {index}/{len(available)} "
                f"sample={sample_id} subtask={subtask}",
            )

    by_subtask: dict[str, list[dict]] = defaultdict(list)
    by_degradation_type: dict[str, list[dict]] = defaultdict(list)
    by_env_type: dict[str, list[dict]] = defaultdict(list)
    by_env_subtype: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)

    for row in details:
        by_subtask[row.get("subtask", "unknown")].append(row)
        if row.get("degradation_type"):
            by_degradation_type[row["degradation_type"]].append(row)
        if row.get("env_type"):
            by_env_type[row["env_type"]].append(row)
        if row.get("env_subtype"):
            by_env_subtype[row["env_subtype"]].append(row)
        by_language[row.get("language", "unknown")].append(row)
        by_dataset[row.get("source_dataset", "unknown")].append(row)

    overall_stats = _stats(details)
    summary = {
        "overall": {
            "total": len(samples),
            "outputs_evaluated": len(details),
            "missing_outputs": len(missing),
            "coverage": round(len(details) / len(samples), 4) if samples else 0.0,
            **{k: v for k, v in overall_stats.items() if k != "total"},
        },
        "by_subtask": {k: _stats(v) for k, v in sorted(by_subtask.items())},
        "by_degradation_type": {k: _stats(v) for k, v in sorted(by_degradation_type.items())},
        "by_env_type": {k: _stats(v) for k, v in sorted(by_env_type.items())},
        "by_env_subtype": {k: _stats(v) for k, v in sorted(by_env_subtype.items())},
        "by_language": {k: _stats(v) for k, v in sorted(by_language.items())},
        "by_dataset": {k: _stats(v) for k, v in sorted(by_dataset.items())},
    }
    return {"summary": summary, "details": details, "missing_outputs": missing}


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")
    print("开始评估 Acoustic DNSMOS + preservation + RT60 + PANNs ASA...")

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
    parser = argparse.ArgumentParser(description="Evaluate acoustic editing outputs / 评估声学编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/acoustic_editing/samples.jsonl) / 测试用例文件")
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
