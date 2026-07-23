from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]

SCRIPT_DIR = (
    ROOT
    / "local_eval/emotion_judge/scripts"
)

sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen25_omni_micro5 as base  # noqa: E402


MANIFEST = (
    ROOT
    / "local_eval/emotion_judge/manifests/"
      "emotion_judge_calibration452.jsonl"
)

STAGE1_PREDICTIONS = (
    ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/"
      "calibration452_prompt_v2/"
      "predictions.jsonl"
)

EXPECTED_MANIFEST_SHA256 = (
    "b6397dbffa8eb87bf49a62428df7ad136"
    "0776bd079d4b415058ffe069d5a7094"
)

EXPECTED_STAGE1_SHA256 = (
    "622719b80557e9a461d4e7a3e7a17cb9"
    "643e0ab9f1e548e5ddb6fd5557f619ba"
)

PROMPT_VERSION = (
    "stage2_en_happy_vs_excited_v1"
)

TRIGGER_LANGUAGE = "en"
TRIGGER_LABEL = "happy"

CANDIDATE_LABELS = [
    "happy",
    "excited",
]

EXPECTED_TRIGGER_COUNT = 35

RESULT_DIR = (
    ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/"
      "calibration452_stage2/"
      "en_happy_vs_excited_v1"
)

STAGE2_PREDICTIONS = (
    RESULT_DIR / "stage2_predictions.jsonl"
)

COMBINED_PREDICTIONS = (
    RESULT_DIR / "combined_predictions.jsonl"
)

METRICS_FILE = RESULT_DIR / "metrics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} 第 {line_number} 行 JSON 非法"
                ) from exc

    return rows


def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    temporary.write_text(
        content,
        encoding="utf-8",
    )

    os.replace(temporary, path)


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    content = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    )

    atomic_write_text(path, content)


def safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float:
    return (
        numerator / denominator
        if denominator
        else 0.0
    )


def build_stage2_prompt(row: dict) -> str:
    transcript = row.get("transcript", "")

    return f"""
You are performing a focused second-stage speech-emotion assessment.

The first-stage system predicted HAPPY. That prediction may be correct or
incorrect. Reassess the audio independently, choosing only between HAPPY and
EXCITED.

The transcript is supplied only to align the spoken content. Do not infer
emotion from positive words, events, gratitude, success, or the semantic
meaning of the sentence. Judge primarily from the acoustic delivery.

Definitions:

HAPPY:
Positive pleasure, warmth, satisfaction, friendliness, or contentment.
The delivery may sound positive and expressive, but it remains relatively
calm, controlled, and moderate in arousal.

EXCITED:
Clearly high-arousal positive enthusiasm, anticipation, or exhilaration.
The voice should sound strongly energized, animated, lively, or activated,
often with faster rhythm, larger pitch movement, stronger intensity, or
sustained vocal brightness.

Decision rule:

Choose EXCITED only when the audio contains clear and sustained high-arousal
positive vocal evidence. Positive words alone are not sufficient.

Choose HAPPY when the emotion is positive but its energy and arousal remain
moderate, calm, warm, or controlled.

Do not classify ordinary friendliness or gratitude as EXCITED unless the
voice itself is strongly animated.

Transcript:
{transcript}

Return only one of the following JSON objects:

{{"predicted_emotion":"happy"}}

or

{{"predicted_emotion":"excited"}}

Do not output confidence, explanation, Markdown, or any additional text.
""".strip()


def classification_metrics(
    prediction_rows: list[dict],
    manifest_rows: list[dict],
) -> dict:
    labels_by_language: dict[str, list[str]] = {}

    for row in manifest_rows:
        language = row["language"]

        if language not in labels_by_language:
            labels_by_language[language] = list(
                row["allowed_labels"]
            )

    language_metrics = {}

    for language in ["en", "zh"]:
        labels = labels_by_language[language]

        subset = [
            row
            for row in prediction_rows
            if row["language"] == language
        ]

        per_class = {}

        for label in labels:
            true_positive = sum(
                1
                for row in subset
                if row["ground_truth_emotion"] == label
                and row["predicted_emotion"] == label
            )

            false_positive = sum(
                1
                for row in subset
                if row["ground_truth_emotion"] != label
                and row["predicted_emotion"] == label
            )

            false_negative = sum(
                1
                for row in subset
                if row["ground_truth_emotion"] == label
                and row["predicted_emotion"] != label
            )

            support = sum(
                1
                for row in subset
                if row["ground_truth_emotion"] == label
            )

            precision = safe_div(
                true_positive,
                true_positive + false_positive,
            )

            recall = safe_div(
                true_positive,
                true_positive + false_negative,
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0.0
            )

            per_class[label] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        correct = sum(
            row["predicted_emotion"]
            == row["ground_truth_emotion"]
            for row in subset
        )

        macro_f1 = sum(
            result["f1"]
            for result in per_class.values()
        ) / len(labels)

        language_metrics[language] = {
            "sample_count": len(subset),
            "correct_count": correct,
            "accuracy": safe_div(
                correct,
                len(subset),
            ),
            "macro_f1": macro_f1,
            "per_class": per_class,
        }

    total_correct = sum(
        row["predicted_emotion"]
        == row["ground_truth_emotion"]
        for row in prediction_rows
    )

    selection_score = (
        0.5
        * language_metrics["en"]["macro_f1"]
        + 0.5
        * language_metrics["zh"]["macro_f1"]
    )

    return {
        "sample_count": len(prediction_rows),
        "correct_count": total_correct,
        "accuracy": safe_div(
            total_correct,
            len(prediction_rows),
        ),
        "language_metrics": language_metrics,
        "selection_score": selection_score,
    }


def load_existing_stage2() -> dict[str, dict]:
    if not STAGE2_PREDICTIONS.exists():
        return {}

    rows = load_jsonl(STAGE2_PREDICTIONS)
    result = {}

    for row in rows:
        if row.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError(
                "已有二阶段结果的 Prompt 版本不一致"
            )

        if (
            row.get("model_revision")
            != base.core.MODEL_REVISION
        ):
            raise RuntimeError(
                "已有二阶段结果的模型版本不一致"
            )

        sample_id = row["sample_id"]

        if sample_id in result:
            raise RuntimeError(
                f"二阶段结果存在重复样本：{sample_id}"
            )

        result[sample_id] = row

    return result


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    manifest_hash = sha256_file(MANIFEST)
    stage1_hash = sha256_file(
        STAGE1_PREDICTIONS
    )

    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "calibration manifest SHA256 不匹配：\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"actual={manifest_hash}"
        )

    if stage1_hash != EXPECTED_STAGE1_SHA256:
        raise RuntimeError(
            "第一阶段预测 SHA256 不匹配：\n"
            f"expected={EXPECTED_STAGE1_SHA256}\n"
            f"actual={stage1_hash}"
        )

    manifest_rows = load_jsonl(MANIFEST)
    stage1_rows = load_jsonl(
        STAGE1_PREDICTIONS
    )

    if len(manifest_rows) != 452:
        raise RuntimeError(
            "calibration manifest 应为 452 条"
        )

    if len(stage1_rows) != 452:
        raise RuntimeError(
            "第一阶段预测应为 452 条"
        )

    manifest_by_id = {
        row["sample_id"]: row
        for row in manifest_rows
    }

    stage1_by_id = {
        row["sample_id"]: row
        for row in stage1_rows
    }

    if set(manifest_by_id) != set(stage1_by_id):
        raise RuntimeError(
            "manifest 与第一阶段预测 sample_id 不一致"
        )

    triggered_rows = [
        {
            **manifest_by_id[sample_id],
            "stage1_prediction": stage1_row[
                "stage1_prediction"
            ],
            "stage1_correct": stage1_row[
                "stage1_correct"
            ],
        }
        for sample_id, stage1_row
        in stage1_by_id.items()
        if (
            stage1_row["language"]
            == TRIGGER_LANGUAGE
            and stage1_row["stage1_prediction"]
            == TRIGGER_LABEL
        )
    ]

    triggered_rows.sort(
        key=lambda row: row["sample_id"]
    )

    if len(triggered_rows) != EXPECTED_TRIGGER_COUNT:
        raise RuntimeError(
            "触发样本数量不一致："
            f"expected={EXPECTED_TRIGGER_COUNT}, "
            f"actual={len(triggered_rows)}"
        )

    print("=" * 100)
    print(
        "二阶段实验：英文 happy 与 excited"
    )
    print("=" * 100)
    print("Prompt version       :", PROMPT_VERSION)
    print("calibration SHA256   :", manifest_hash)
    print("stage1 SHA256        :", stage1_hash)
    print("触发语言             :", TRIGGER_LANGUAGE)
    print("第一阶段触发标签     :", TRIGGER_LABEL)
    print("二阶段候选标签       :", CANDIDATE_LABELS)
    print("触发样本数量         :", len(triggered_rows))
    print(
        "触发集真实标签分布   :",
        dict(
            sorted(
                Counter(
                    row["ground_truth_emotion"]
                    for row in triggered_rows
                ).items()
            )
        ),
    )
    print(
        "reserve accessed     : False"
    )

    completed = load_existing_stage2()

    print(
        "已有结果并跳过       :",
        len(completed),
    )
    print(
        "本次待推理           :",
        len(triggered_rows) - len(completed),
    )

    if len(completed) < len(triggered_rows):
        base.core.build_prompt = (
            build_stage2_prompt
        )

        device_map = {
            "thinker.model": "cuda",
            "thinker.lm_head": "cuda",
            "thinker.visual": "cpu",
            "thinker.audio_tower": "cpu",
            "talker": "cuda",
            "token2wav": "cuda",
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print("\n===== 加载模型 =====")

        load_start = time.time()

        model = base.core.GPTQModel.load(
            str(base.core.MODEL_DIR),
            device_map=device_map,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        model.eval()

        processor = (
            base.core.Qwen2_5OmniProcessor.from_pretrained(
                str(base.core.MODEL_DIR),
                local_files_only=True,
            )
        )

        print(
            "模型加载耗时（秒）   :",
            round(time.time() - load_start, 3),
        )
        print(
            "当前显存占用（GiB） :",
            round(
                torch.cuda.memory_allocated()
                / 1024**3,
                3,
            ),
        )

        print("\n===== 二阶段推理 =====")

        for index, row in enumerate(
            triggered_rows,
            start=1,
        ):
            sample_id = row["sample_id"]

            if sample_id in completed:
                continue

            audio_path = base.resolve_audio_path(
                row["audio_path"]
            )

            actual_hash, audio_metadata = (
                base.normalized_audio_sha256(
                    audio_path
                )
            )

            expected_hash = (
                row["audio_sha256"].lower()
            )

            if actual_hash != expected_hash:
                raise RuntimeError(
                    "规范化音频 SHA256 不一致："
                    f"{sample_id}"
                )

            prompt_row = {
                **row,
                "allowed_labels": CANDIDATE_LABELS,
            }

            inputs = base.prepare_inputs(
                processor,
                prompt_row,
                audio_path,
            )

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            inference_start = time.time()

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    use_audio_in_video=False,
                    return_audio=False,
                    thinker_do_sample=False,
                    thinker_max_new_tokens=48,
                )

            torch.cuda.synchronize()

            inference_seconds = (
                time.time() - inference_start
            )

            raw_output, token_metadata = (
                base.decode_generated_text(
                    processor,
                    inputs,
                    output,
                )
            )

            parsed = (
                base.core.parse_json_object(
                    raw_output
                )
            )

            stage2_prediction = None

            if parsed is not None:
                stage2_prediction = str(
                    parsed.get(
                        "predicted_emotion",
                        "",
                    )
                ).strip().lower()

            legal_stage2 = (
                stage2_prediction
                in CANDIDATE_LABELS
            )

            # 安全回退：
            # 二阶段输出非法时保留第一阶段 happy。
            final_prediction = (
                stage2_prediction
                if legal_stage2
                else row["stage1_prediction"]
            )

            result = {
                "trigger_index": index,
                "sample_id": sample_id,
                "language": row["language"],
                "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"],
                "audio_hash_verified": True,
                "audio_metadata": audio_metadata,
                "transcript": row.get(
                    "transcript",
                    "",
                ),
                "ground_truth_emotion": row[
                    "ground_truth_emotion"
                ],
                "stage1_prediction": row[
                    "stage1_prediction"
                ],
                "stage1_correct": row[
                    "stage1_correct"
                ],
                "stage2_prediction": stage2_prediction,
                "stage2_legal_label": legal_stage2,
                "final_prediction": final_prediction,
                "final_correct": (
                    final_prediction
                    == row["ground_truth_emotion"]
                ),
                "prediction_changed": (
                    final_prediction
                    != row["stage1_prediction"]
                ),
                "raw_output": raw_output,
                "parsed_json": parsed,
                "token_metadata": token_metadata,
                "inference_seconds": round(
                    inference_seconds,
                    6,
                ),
                "peak_gpu_gib": round(
                    torch.cuda.max_memory_allocated()
                    / 1024**3,
                    6,
                ),
                "prompt_version": PROMPT_VERSION,
                "model_revision": (
                    base.core.MODEL_REVISION
                ),
                "source_revision": (
                    base.core.SOURCE_REVISION
                ),
                "manifest_sha256": (
                    EXPECTED_MANIFEST_SHA256
                ),
                "stage1_predictions_sha256": (
                    EXPECTED_STAGE1_SHA256
                ),
            }

            completed[sample_id] = result

            ordered_results = [
                completed[item["sample_id"]]
                for item in triggered_rows
                if item["sample_id"] in completed
            ]

            write_jsonl(
                STAGE2_PREDICTIONS,
                ordered_results,
            )

            print(
                f"[{index:02d}/{len(triggered_rows)}] "
                f"target="
                f"{row['ground_truth_emotion']:<9s} "
                f"stage1=happy     "
                f"stage2="
                f"{str(stage2_prediction):<9s} "
                f"final_correct="
                f"{str(result['final_correct']):<5s} "
                f"changed="
                f"{str(result['prediction_changed']):<5s} "
                f"time="
                f"{inference_seconds:.3f}s",
                flush=True,
            )

            del inputs
            del output
            torch.cuda.empty_cache()

    if len(completed) != EXPECTED_TRIGGER_COUNT:
        raise RuntimeError(
            "二阶段结果不完整："
            f"{len(completed)}/"
            f"{EXPECTED_TRIGGER_COUNT}"
        )

    stage2_rows = [
        completed[row["sample_id"]]
        for row in triggered_rows
    ]

    stage2_by_id = {
        row["sample_id"]: row
        for row in stage2_rows
    }

    combined_rows = []

    baseline_metric_rows = []

    for manifest_row in manifest_rows:
        sample_id = manifest_row["sample_id"]
        stage1_row = stage1_by_id[sample_id]

        stage1_prediction = stage1_row[
            "stage1_prediction"
        ]

        final_prediction = stage1_prediction
        stage2_applied = False

        if sample_id in stage2_by_id:
            stage2_applied = True
            final_prediction = stage2_by_id[
                sample_id
            ]["final_prediction"]

        combined_row = {
            "sample_id": sample_id,
            "language": manifest_row["language"],
            "ground_truth_emotion": (
                manifest_row[
                    "ground_truth_emotion"
                ]
            ),
            "stage1_prediction": stage1_prediction,
            "stage2_applied": stage2_applied,
            "final_prediction": final_prediction,
            "correct": (
                final_prediction
                == manifest_row[
                    "ground_truth_emotion"
                ]
            ),
        }

        combined_rows.append(combined_row)

        baseline_metric_rows.append(
            {
                "sample_id": sample_id,
                "language": manifest_row[
                    "language"
                ],
                "ground_truth_emotion": (
                    manifest_row[
                        "ground_truth_emotion"
                    ]
                ),
                "predicted_emotion": (
                    stage1_prediction
                ),
            }
        )

    hybrid_metric_rows = [
        {
            "sample_id": row["sample_id"],
            "language": row["language"],
            "ground_truth_emotion": row[
                "ground_truth_emotion"
            ],
            "predicted_emotion": row[
                "final_prediction"
            ],
        }
        for row in combined_rows
    ]

    baseline_metrics = classification_metrics(
        baseline_metric_rows,
        manifest_rows,
    )

    hybrid_metrics = classification_metrics(
        hybrid_metric_rows,
        manifest_rows,
    )

    write_jsonl(
        COMBINED_PREDICTIONS,
        combined_rows,
    )

    gained = [
        row
        for row in stage2_rows
        if not row["stage1_correct"]
        and row["final_correct"]
    ]

    lost = [
        row
        for row in stage2_rows
        if row["stage1_correct"]
        and not row["final_correct"]
    ]

    changed = [
        row
        for row in stage2_rows
        if row["prediction_changed"]
    ]

    legal_count = sum(
        row["stage2_legal_label"]
        for row in stage2_rows
    )

    stage1_correct_trigger = sum(
        row["stage1_correct"]
        for row in stage2_rows
    )

    final_correct_trigger = sum(
        row["final_correct"]
        for row in stage2_rows
    )

    metrics = {
        "prompt_version": PROMPT_VERSION,
        "reserve_accessed": False,
        "trigger": {
            "language": TRIGGER_LANGUAGE,
            "stage1_prediction": TRIGGER_LABEL,
            "candidate_labels": CANDIDATE_LABELS,
            "sample_count": len(stage2_rows),
            "stage1_correct_count": (
                stage1_correct_trigger
            ),
            "final_correct_count": (
                final_correct_trigger
            ),
            "stage1_accuracy": safe_div(
                stage1_correct_trigger,
                len(stage2_rows),
            ),
            "final_accuracy": safe_div(
                final_correct_trigger,
                len(stage2_rows),
            ),
            "legal_stage2_count": legal_count,
            "legal_stage2_rate": safe_div(
                legal_count,
                len(stage2_rows),
            ),
            "changed_count": len(changed),
            "gained_count": len(gained),
            "lost_count": len(lost),
            "net_gain_count": (
                len(gained) - len(lost)
            ),
            "stage2_prediction_distribution": dict(
                sorted(
                    Counter(
                        row["final_prediction"]
                        for row in stage2_rows
                    ).items()
                )
            ),
            "gained_sample_ids": [
                row["sample_id"]
                for row in gained
            ],
            "lost_sample_ids": [
                row["sample_id"]
                for row in lost
            ],
        },
        "baseline_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "deltas": {
            "overall_accuracy": (
                hybrid_metrics["accuracy"]
                - baseline_metrics["accuracy"]
            ),
            "en_macro_f1": (
                hybrid_metrics[
                    "language_metrics"
                ]["en"]["macro_f1"]
                - baseline_metrics[
                    "language_metrics"
                ]["en"]["macro_f1"]
            ),
            "selection_score": (
                hybrid_metrics["selection_score"]
                - baseline_metrics[
                    "selection_score"
                ]
            ),
            "happy_f1": (
                hybrid_metrics[
                    "language_metrics"
                ]["en"]["per_class"][
                    "happy"
                ]["f1"]
                - baseline_metrics[
                    "language_metrics"
                ]["en"]["per_class"][
                    "happy"
                ]["f1"]
            ),
            "excited_f1": (
                hybrid_metrics[
                    "language_metrics"
                ]["en"]["per_class"][
                    "excited"
                ]["f1"]
                - baseline_metrics[
                    "language_metrics"
                ]["en"]["per_class"][
                    "excited"
                ]["f1"]
            ),
        },
        "files": {
            "stage2_predictions": str(
                STAGE2_PREDICTIONS
            ),
            "combined_predictions": str(
                COMBINED_PREDICTIONS
            ),
        },
    }

    atomic_write_json(
        METRICS_FILE,
        metrics,
    )

    baseline_en = baseline_metrics[
        "language_metrics"
    ]["en"]

    hybrid_en = hybrid_metrics[
        "language_metrics"
    ]["en"]

    print("\n" + "=" * 100)
    print(
        "英文 happy–excited 二阶段实验结果"
    )
    print("=" * 100)

    print(
        "触发样本数量           :",
        len(stage2_rows),
    )
    print(
        "二阶段合法标签         :",
        f"{legal_count}/{len(stage2_rows)} "
        f"= {safe_div(legal_count, len(stage2_rows)):.6f}",
    )
    print(
        "预测发生改变           :",
        len(changed),
    )
    print(
        "第一阶段触发集正确     :",
        f"{stage1_correct_trigger}/"
        f"{len(stage2_rows)} "
        f"= {safe_div(stage1_correct_trigger, len(stage2_rows)):.6f}",
    )
    print(
        "二阶段后触发集正确     :",
        f"{final_correct_trigger}/"
        f"{len(stage2_rows)} "
        f"= {safe_div(final_correct_trigger, len(stage2_rows)):.6f}",
    )
    print(
        "修复成功数量 gained    :",
        len(gained),
    )
    print(
        "破坏正确数量 lost      :",
        len(lost),
    )
    print(
        "净增益 net gain        :",
        len(gained) - len(lost),
    )

    print()
    print(
        "baseline 总体准确率    :",
        f"{baseline_metrics['accuracy']:.6f}",
    )
    print(
        "hybrid 总体准确率      :",
        f"{hybrid_metrics['accuracy']:.6f}",
    )
    print(
        "总体准确率变化         :",
        f"{metrics['deltas']['overall_accuracy']:+.6f}",
    )

    print()
    print(
        "baseline EN macro-F1   :",
        f"{baseline_en['macro_f1']:.6f}",
    )
    print(
        "hybrid EN macro-F1     :",
        f"{hybrid_en['macro_f1']:.6f}",
    )
    print(
        "EN macro-F1 变化       :",
        f"{metrics['deltas']['en_macro_f1']:+.6f}",
    )

    print()
    print(
        "baseline happy F1      :",
        f"{baseline_en['per_class']['happy']['f1']:.6f}",
    )
    print(
        "hybrid happy F1        :",
        f"{hybrid_en['per_class']['happy']['f1']:.6f}",
    )
    print(
        "baseline excited F1    :",
        f"{baseline_en['per_class']['excited']['f1']:.6f}",
    )
    print(
        "hybrid excited F1      :",
        f"{hybrid_en['per_class']['excited']['f1']:.6f}",
    )

    print()
    print(
        "baseline selection     :",
        f"{baseline_metrics['selection_score']:.6f}",
    )
    print(
        "hybrid selection       :",
        f"{hybrid_metrics['selection_score']:.6f}",
    )
    print(
        "selection 变化         :",
        f"{metrics['deltas']['selection_score']:+.6f}",
    )

    print()
    print(
        "二阶段预测文件         :",
        STAGE2_PREDICTIONS,
    )
    print(
        "组合预测文件           :",
        COMBINED_PREDICTIONS,
    )
    print(
        "指标文件               :",
        METRICS_FILE,
    )


if __name__ == "__main__":
    main()
