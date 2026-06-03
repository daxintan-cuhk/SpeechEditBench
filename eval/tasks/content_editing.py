"""
内容编辑任务评估编排（Content Editing Task Evaluator）

输入：
  - data/content_editing/samples.jsonl    测试用例
  - <output_dir>/                         模型输出音频目录

  模型输出目录约定：
    <output_dir>/<sample_id>.wav
    例：outputs/libritts__a3f8c0e1__replace.wav

输出：
  - 终端打印评估报告（整体 CA + 按编辑类型 / 按数据集 / 按语言）
  - （可选）保存详细结果到 JSON 文件

用法：
  python eval/tasks/content_editing.py \\
      --output-dir /path/to/model/outputs \\
      [--results-file results.json]

当前依赖 eval/metrics/content_accuracy.py 中的 asr_predict()：
英文使用 eval_models/asr/whisper-large-v3，中文使用 eval_models/asr/paraformer-zh。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "data" / "content_editing" / "samples.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.editing_common import (  # noqa: E402
    batch_source_output_speaker_similarity,
    batch_utmos_scores,
    find_output_audio,
    mean_or_none,
    progress_log,
    resolve_sample_path,
    should_log_progress,
)


def load_samples(samples_file: Path) -> list[dict]:
    """从 samples.jsonl 加载内容编辑测试用例。"""
    with samples_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*60}")
    print(f"  内容编辑评估报告（Content Accuracy, CA）")
    print(f"{'='*60}")
    print(
        f"  总样本：{overall['total']}  |"
        f"  已找到输出：{overall['outputs_evaluated']}  |"
        f"  缺失输出：{overall['missing_outputs']}  |"
        f"  EM：{overall['exact_match_rate']:.1%}  |"
        f"  ES：{overall['edit_success_rate']:.1%}  |"
        f"  Target：{overall['target_success_rate']:.1%}  |"
        f"  Joint：{overall['joint_success_rate']:.1%}  |"
        f"  WER：{overall['avg_wer']:.3f}  |"
        f"  CER：{overall['avg_cer']:.3f}  |"
        f"  UTMOS：{(overall['avg_utmos'] or 0.0):.3f}  |"
        f"  Src->Out SS：{(overall['avg_source_output_speaker_similarity'] or 0.0):.3f}"
    )

    edit_types = ["replace", "insert", "delete"]

    if summary.get("by_edit_type"):
        print("\n  按编辑类型：")
        for et in edit_types:
            acc = summary["by_edit_type"].get(et)
            if acc is None:
                continue
            bar = "█" * int(acc["exact_match_rate"] * 20)
            print(
                f"    {et:10s}  {bar:<20s}"
                f"  EM={acc['exact_match_rate']:.1%}"
                f"  ES={acc['edit_success_rate']:.1%}"
                f"  WER={acc['avg_wer']:.3f}"
                f"  CER={acc['avg_cer']:.3f}"
                f"  UTMOS={(acc['avg_utmos'] or 0.0):.3f}"
                f"  SS={(acc['avg_source_output_speaker_similarity'] or 0.0):.3f}"
                f"  (EM {acc.get('exact_match_count', '?')}/{acc['total']};"
                f" ES {acc.get('edit_success_count', '?')}/{acc['total']})"
            )

    if summary.get("by_dataset"):
        print("\n  按数据集：")
        for ds, acc in summary["by_dataset"].items():
            print(
                f"    {ds:20s}  EM={acc['exact_match_rate']:.1%}"
                f"  ES={acc['edit_success_rate']:.1%}"
                f"  WER={acc['avg_wer']:.3f}"
                f"  UTMOS={(acc['avg_utmos'] or 0.0):.3f}"
                f"  (EM {acc.get('exact_match_count', '?')}/{acc['total']})"
            )

    if summary.get("by_language"):
        print("\n  按语言：")
        for lang, acc in summary["by_language"].items():
            primary = "WER" if lang == "en" else "CER"
            val = acc["avg_wer"] if lang == "en" else acc["avg_cer"]
            print(
                f"    {lang:6s}  EM={acc['exact_match_rate']:.1%}"
                f"  ES={acc['edit_success_rate']:.1%}"
                f"  {primary}={val:.3f}"
                f"  UTMOS={(acc['avg_utmos'] or 0.0):.3f}"
                f"  (EM {acc.get('exact_match_count', '?')}/{acc['total']})"
            )

    print(f"{'='*60}\n")


def batch_evaluate(
    samples: list[dict],
    output_audio_dir: Path,
    samples_file: Path,
) -> list[dict]:
    """
    批量评估内容准确率。

    参数：
        samples:           load_samples() 返回的记录列表
        output_audio_dir:  模型输出音频目录，文件名为 <sample_id>.wav
    返回：每条记录的评估结果（同 compute_content_accuracy 的返回格式）
    """
    from eval.metrics.content_accuracy import asr_predict, compute_content_accuracy
    available = [
        (sample, audio_path)
        for sample in samples
        if (audio_path := find_output_audio(output_audio_dir, sample["sample_id"])) is not None
    ]
    total = len(available)
    progress_log("content_editing", f"available outputs {total}/{len(samples)}")
    try:
        progress_log("content_editing", f"UTMOS start {total}")
        utmos_by_sample_id = batch_utmos_scores(
            [(sample["sample_id"], audio_path) for sample, audio_path in available],
            device="auto",
        )
        progress_log("content_editing", "UTMOS done")
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
        progress_log("content_editing", f"source-output speaker similarity start {len(speaker_pairs)}")
        speaker_by_sample_id = batch_source_output_speaker_similarity(speaker_pairs, device="auto")
        progress_log("content_editing", "source-output speaker similarity done")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] source-output speaker similarity 批量计算失败，将逐条置空：{exc}")
        speaker_by_sample_id = {sample["sample_id"]: None for sample in samples}

    results = []
    progress_log("content_editing", "ASR/content accuracy start")
    for index, (sample, audio_path) in enumerate(available, start=1):
        sid        = sample["sample_id"]
        language   = sample.get("language", "en")

        try:
            pred_transcript = asr_predict(audio_path, language)
            result = compute_content_accuracy(sample, pred_transcript)
            result.update({
                "target_success": bool(result.get("edit_success")),
                "content_preservation_metric": None,
                "content_preservation_error": None,
                "content_preservation_threshold": None,
                "content_preservation_pass": None,
                "joint_success": bool(result.get("edit_success")),
                "utmos": utmos_by_sample_id.get(sid),
                "source_output_speaker_similarity": speaker_by_sample_id.get(sid),
            })
        except NotImplementedError:
            raise
        except Exception as e:
            result = {
                "sample_id":            sid,
                "edit_type":            sample["anchor"].get("edit_type", "unknown"),
                "transcript_target":    sample["anchor"].get("transcript_target", ""),
                "transcript_predicted": None,
                "norm_target":          None,
                "norm_predicted":       None,
                "exact_match":          False,
                "edit_success":         False,
                "edit_success_reason":  None,
                "wer":                  None,
                "cer":                  None,
                "target_success":       False,
                "content_preservation_metric": None,
                "content_preservation_error": None,
                "content_preservation_threshold": None,
                "content_preservation_pass": None,
                "joint_success":        False,
                "utmos":                utmos_by_sample_id.get(sid),
                "source_output_speaker_similarity": speaker_by_sample_id.get(sid),
                "source_dataset":       sample.get("source_dataset", "unknown"),
                "language":             sample.get("language", "en"),
                "error":                str(e),
            }

        results.append(result)
        if should_log_progress(index, len(available)):
            progress_log("content_editing", f"ASR/content accuracy {index}/{len(available)} sample={sid}")

    return results


def summarize(
    results: list[dict],
    *,
    total_samples: int | None = None,
    missing_outputs: int = 0,
) -> dict:
    """
    汇总批量评估结果。

    返回：
        {
            "overall":        {"exact_match_rate": float, "edit_success_rate": float, ...},
            "by_edit_type":   {...},   # 按 replace / insert / delete 分组
            "by_dataset":     {...},   # 按 source_dataset 分组
            "by_language":    {...},   # 按 language 分组
        }
    """
    def _stats(items: list[dict]) -> dict:
        total = len(items)
        if not total:
            return {
                "exact_match_rate": 0.0,
                "edit_success_rate": 0.0,
                "avg_wer": 0.0,
                "avg_cer": 0.0,
                "avg_utmos": None,
                "avg_source_output_speaker_similarity": None,
                "target_success_rate": 0.0,
                "content_preservation_pass_rate": None,
                "avg_content_preservation_error": None,
                "joint_success_rate": 0.0,
                "total": 0,
                "exact_match_count": 0,
                "edit_success_count": 0,
            }
        em    = sum(1 for r in items if r.get("exact_match"))
        es    = sum(1 for r in items if r.get("edit_success"))
        wers  = [r["wer"] for r in items if r.get("wer") is not None]
        cers  = [r["cer"] for r in items if r.get("cer") is not None]
        return {
            "exact_match_rate": round(em / total, 4),
            "edit_success_rate": round(es / total, 4),
            "avg_wer":          round(sum(wers) / len(wers), 4) if wers else 0.0,
            "avg_cer":          round(sum(cers) / len(cers), 4) if cers else 0.0,
            "avg_utmos": mean_or_none([r.get("utmos") for r in items]),
            "avg_source_output_speaker_similarity": mean_or_none([
                r.get("source_output_speaker_similarity") for r in items
            ]),
            "target_success_rate": round(es / total, 4),
            "content_preservation_pass_rate": None,
            "avg_content_preservation_error": None,
            "joint_success_rate": round(es / total, 4),
            "total":            total,
            "exact_match_count": em,
            "edit_success_count": es,
        }

    by_type: dict[str, list] = defaultdict(list)
    by_ds:   dict[str, list] = defaultdict(list)
    by_lang: dict[str, list] = defaultdict(list)

    for r in results:
        by_type[r.get("edit_type", "unknown")].append(r)
        by_ds  [r.get("source_dataset", "unknown")].append(r)
        by_lang[r.get("language", "unknown")].append(r)

    summary = {
        "overall":      _stats(results),
        "by_edit_type": {k: _stats(v) for k, v in sorted(by_type.items())},
        "by_dataset":   {k: _stats(v) for k, v in sorted(by_ds.items())},
        "by_language":  {k: _stats(v) for k, v in sorted(by_lang.items())},
    }
    total = len(results) if total_samples is None else total_samples
    outputs_evaluated = len(results)
    summary["overall"].update({
        "total": total,
        "outputs_evaluated": outputs_evaluated,
        "missing_outputs": missing_outputs,
        "coverage": round(outputs_evaluated / total, 4) if total else 0.0,
        "error_count": sum(1 for r in results if r.get("error")),
    })
    return summary


def run_evaluation(
    output_dir: Path,
    results_file: Path | None = None,
    samples_file: Path = SAMPLES_FILE,
) -> dict:
    samples = load_samples(samples_file)
    print(f"加载测试集：{len(samples)} 条（来自 {samples_file}）")

    # 过滤掉没有对应输出文件的样本
    available, missing = [], []
    for s in samples:
        if find_output_audio(output_dir, s["sample_id"]) is not None:
            available.append(s)
        else:
            missing.append(s["sample_id"])

    if missing:
        print(f"[WARN] {len(missing)} 条样本未找到模型输出，已跳过。")

    print(f"开始评估 {len(available)} 条样本...")
    results = batch_evaluate(available, output_dir, samples_file)
    summary = summarize(
        results,
        total_samples=len(samples),
        missing_outputs=len(missing),
    )
    print_report(summary)

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存：{results_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate content editing outputs / 评估内容编辑任务。")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="model output audio directory; files are named <sample_id>.wav / 模型输出目录")
    parser.add_argument("--results-file", default=None, type=Path,
                        help="optional detailed result JSON path / 详细结果保存路径")
    parser.add_argument("--samples-file", default=SAMPLES_FILE, type=Path,
                        help="samples JSONL path (default: data/content_editing/samples.jsonl) / 测试用例文件")
    args = parser.parse_args()

    run_evaluation(
        output_dir=args.output_dir,
        results_file=args.results_file,
        samples_file=args.samples_file,
    )


if __name__ == "__main__":
    main()
