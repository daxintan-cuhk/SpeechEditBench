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


ROOT = Path(
    "/home/litianxiang/workplace/datasets/SpeechEditBench"
)

SCRIPT_DIR = (
    ROOT
    / "local_eval/emotion_judge/scripts"
)

sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen25_omni_micro5 as base  # noqa: E402


STAGE1_PREDICTIONS = (
    ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/"
      "calibration452_prompt_v2/"
      "predictions.jsonl"
)

EXPECTED_STAGE1_SHA256 = (
    "622719b80557e9a461d4e7a3e7a17cb9"
    "643e0ab9f1e548e5ddb6fd5557f619ba"
)

PROMPT_VERSION = (
    "probe_en_neutral_pairwise_capability_v1"
)

RESULT_DIR = (
    ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/"
      "calibration452_pairwise_probe/"
      "en_neutral_pairs_v1"
)

PREDICTIONS_FILE = (
    RESULT_DIR / "predictions.jsonl"
)

SUMMARY_FILE = RESULT_DIR / "summary.json"


PAIR_CONFIGS = {
    "neutral_vs_fearful": {
        "labels": ["neutral", "fearful"],
        "definitions": {
            "neutral": (
                "No clearly dominant emotion. The voice is ordinary, "
                "stable, controlled, and relatively unmarked."
            ),
            "fearful": (
                "Fear, anxiety, apprehension, insecurity, or perceived "
                "threat. The voice may be tense, shaky, breathy, "
                "hesitant, unstable, cautious, or constricted."
            ),
        },
        "contrast": (
            "Choose fearful only when anxiety, shakiness, breathiness, "
            "hesitation, instability, or threat-related tension is "
            "clearly audible. Calm or mildly expressive ordinary speech "
            "is neutral."
        ),
    },

    "neutral_vs_sad": {
        "labels": ["neutral", "sad"],
        "definitions": {
            "neutral": (
                "No clearly dominant emotion. The voice is ordinary, "
                "stable, controlled, and relatively unmarked."
            ),
            "sad": (
                "Sorrow, discouragement, resignation, loss, or low mood. "
                "The voice may be subdued, withdrawn, slower, softer, "
                "lower in energy, or emotionally heavy."
            ),
        },
        "contrast": (
            "Choose sad only when low energy, withdrawal, heaviness, "
            "subdued delivery, or resignation is clearly audible. "
            "Ordinary calm speech without those cues is neutral."
        ),
    },

    "neutral_vs_surprise": {
        "labels": ["neutral", "surprise"],
        "definitions": {
            "neutral": (
                "No clearly dominant emotion. The voice is ordinary, "
                "stable, controlled, and relatively unmarked."
            ),
            "surprise": (
                "A sudden astonished or startled reaction, normally "
                "marked by an abrupt and noticeable pitch, loudness, "
                "duration, rhythm, or timing change."
            ),
        },
        "contrast": (
            "Choose surprise only when an abrupt astonished or startled "
            "vocal event is audible. A question, unexpected statement, "
            "or surprising sentence meaning alone is not sufficient."
        ),
    },
}


EXPECTED_PAIR_COUNTS = {
    "neutral_vs_fearful": 25,
    "neutral_vs_sad": 23,
    "neutral_vs_surprise": 23,
}

EXPECTED_CASE_COUNT = 71


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


def counterbalanced_label_order(
    pair_name: str,
    sample_id: str,
    labels: list[str],
) -> list[str]:
    digest = hashlib.sha256(
        f"{pair_name}|{sample_id}".encode(
            "utf-8"
        )
    ).digest()

    if digest[0] % 2 == 0:
        return list(labels)

    return list(reversed(labels))


def build_pairwise_prompt(row: dict) -> str:
    pair_name = row["_pair_name"]
    first_label = row["_displayed_labels"][0]
    second_label = row["_displayed_labels"][1]

    config = PAIR_CONFIGS[pair_name]
    definitions = config["definitions"]
    transcript = row.get("transcript", "")

    return f"""
Perform a strict two-way acoustic speech-emotion comparison.

Choose exactly one of the following two labels:

1. {first_label}
2. {second_label}

The order of the labels was randomized and carries no meaning. Neither label
is the default, preferred, safer, or fallback answer.

The transcript is supplied only to align the spoken content. Do not infer
emotion from word meaning, described events, sentence topic, interpersonal
context, or whether the text sounds positive or negative.

Judge primarily from the voice itself:

- pitch level, range, and movement
- loudness and vocal energy
- speaking rate, rhythm, and pauses
- tension, breathiness, shakiness, and vocal stability
- whether the vocal pattern is ordinary, sustained, or abrupt

Definitions:

{first_label.upper()}:
{definitions[first_label]}

{second_label.upper()}:
{definitions[second_label]}

Critical distinction:

{config["contrast"]}

Assess both alternatives independently. Select the label supported by the
stronger acoustic evidence.

Transcript:
{transcript}

Return only one JSON object using one of the two exact labels:

{{"predicted_emotion":"{first_label}"}}

or

{{"predicted_emotion":"{second_label}"}}

Do not output confidence, explanation, Markdown, or any additional text.
""".strip()


def build_cases(rows: list[dict]) -> list[dict]:
    neutral_triggered = [
        row
        for row in rows
        if (
            row["language"] == "en"
            and row["stage1_prediction"] == "neutral"
        )
    ]

    if len(neutral_triggered) != 39:
        raise RuntimeError(
            "英文 stage1=neutral 样本应为 39 条，"
            f"实际为 {len(neutral_triggered)}"
        )

    cases = []

    for pair_name, config in PAIR_CONFIGS.items():
        labels = config["labels"]

        pair_rows = [
            row
            for row in neutral_triggered
            if row["ground_truth_emotion"] in labels
        ]

        expected_count = EXPECTED_PAIR_COUNTS[
            pair_name
        ]

        if len(pair_rows) != expected_count:
            raise RuntimeError(
                f"{pair_name} 应为 {expected_count} 条，"
                f"实际为 {len(pair_rows)}"
            )

        for row in sorted(
            pair_rows,
            key=lambda item: item["sample_id"],
        ):
            displayed_labels = (
                counterbalanced_label_order(
                    pair_name,
                    row["sample_id"],
                    labels,
                )
            )

            cases.append(
                {
                    **row,
                    "_case_id": (
                        f"{pair_name}::{row['sample_id']}"
                    ),
                    "_pair_name": pair_name,
                    "_pair_labels": list(labels),
                    "_displayed_labels": (
                        displayed_labels
                    ),
                    "allowed_labels": list(labels),
                }
            )

    if len(cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"pairwise case 应为 {EXPECTED_CASE_COUNT} 条，"
            f"实际为 {len(cases)}"
        )

    return cases


def load_completed() -> dict[str, dict]:
    if not PREDICTIONS_FILE.exists():
        return {}

    rows = load_jsonl(PREDICTIONS_FILE)
    completed = {}

    for row in rows:
        if (
            row.get("prompt_version")
            != PROMPT_VERSION
        ):
            raise RuntimeError(
                "已有结果 Prompt 版本不一致"
            )

        if (
            row.get("stage1_predictions_sha256")
            != EXPECTED_STAGE1_SHA256
        ):
            raise RuntimeError(
                "已有结果所绑定的 stage1 SHA256 不一致"
            )

        case_id = row["case_id"]

        if case_id in completed:
            raise RuntimeError(
                f"已有结果存在重复 case_id：{case_id}"
            )

        completed[case_id] = row

    return completed


def calculate_pair_metrics(
    pair_name: str,
    results: list[dict],
) -> dict:
    labels = PAIR_CONFIGS[pair_name]["labels"]

    subset = [
        row
        for row in results
        if row["pair_name"] == pair_name
    ]

    legal_count = sum(
        row["legal_label"]
        for row in subset
    )

    correct_count = sum(
        row["correct"]
        for row in subset
    )

    predicted_distribution = Counter(
        (
            row["predicted_emotion"]
            if row["legal_label"]
            else "__invalid__"
        )
        for row in subset
    )

    true_distribution = Counter(
        row["ground_truth_emotion"]
        for row in subset
    )

    confusion = {
        true_label: {
            predicted_label: 0
            for predicted_label in (
                labels + ["__invalid__"]
            )
        }
        for true_label in labels
    }

    recalls = {}

    for row in subset:
        true_label = row["ground_truth_emotion"]

        predicted_label = (
            row["predicted_emotion"]
            if row["legal_label"]
            else "__invalid__"
        )

        confusion[true_label][
            predicted_label
        ] += 1

    for label in labels:
        support = true_distribution[label]
        true_positive = confusion[label][label]

        recalls[label] = safe_div(
            true_positive,
            support,
        )

    balanced_accuracy = sum(
        recalls.values()
    ) / len(labels)

    first_listed_count = sum(
        row["predicted_first_listed"]
        for row in subset
        if row["legal_label"]
    )

    return {
        "pair_name": pair_name,
        "labels": labels,
        "sample_count": len(subset),
        "legal_count": legal_count,
        "legal_rate": safe_div(
            legal_count,
            len(subset),
        ),
        "correct_count": correct_count,
        "accuracy": safe_div(
            correct_count,
            len(subset),
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "recall_by_label": recalls,
        "true_distribution": dict(
            sorted(true_distribution.items())
        ),
        "predicted_distribution": dict(
            sorted(
                predicted_distribution.items()
            )
        ),
        "confusion_matrix": confusion,
        "first_listed_choice_count": (
            first_listed_count
        ),
        "first_listed_choice_rate": safe_div(
            first_listed_count,
            legal_count,
        ),
    }


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    actual_stage1_hash = sha256_file(
        STAGE1_PREDICTIONS
    )

    if (
        actual_stage1_hash
        != EXPECTED_STAGE1_SHA256
    ):
        raise RuntimeError(
            "calibration stage1 SHA256 不匹配：\n"
            f"expected={EXPECTED_STAGE1_SHA256}\n"
            f"actual={actual_stage1_hash}"
        )

    stage1_rows = load_jsonl(
        STAGE1_PREDICTIONS
    )

    if len(stage1_rows) != 452:
        raise RuntimeError(
            f"stage1 预测应为 452 条，"
            f"实际为 {len(stage1_rows)}"
        )

    cases = build_cases(stage1_rows)
    completed = load_completed()

    expected_case_ids = {
        row["_case_id"]
        for row in cases
    }

    unexpected = (
        set(completed) - expected_case_ids
    )

    if unexpected:
        raise RuntimeError(
            "已有结果包含当前实验之外的 case："
            f"{sorted(unexpected)[:5]}"
        )

    print("=" * 100)
    print(
        "英文 neutral 相关情感 pairwise 能力诊断"
    )
    print("=" * 100)
    print("Prompt version       :", PROMPT_VERSION)
    print("stage1 SHA256        :", actual_stage1_hash)
    print("pair 数量            :", len(PAIR_CONFIGS))
    print("总推理 case 数量     :", len(cases))
    print("已有结果并跳过       :", len(completed))
    print(
        "本次待推理           :",
        len(cases) - len(completed),
    )
    print("reserve accessed     : False")

    if len(completed) < len(cases):
        base.core.build_prompt = (
            build_pairwise_prompt
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

        print("\n===== pairwise 推理 =====")

        for index, case in enumerate(
            cases,
            start=1,
        ):
            case_id = case["_case_id"]

            if case_id in completed:
                continue

            audio_path = base.resolve_audio_path(
                case["audio_path"]
            )

            actual_audio_hash, audio_metadata = (
                base.normalized_audio_sha256(
                    audio_path
                )
            )

            expected_audio_hash = (
                case["audio_sha256"].lower()
            )

            if (
                actual_audio_hash
                != expected_audio_hash
            ):
                raise RuntimeError(
                    "规范化音频 SHA256 不一致："
                    f"{case['sample_id']}"
                )

            inputs = base.prepare_inputs(
                processor,
                case,
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
                    thinker_max_new_tokens=32,
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

            prediction = None

            if parsed is not None:
                prediction = str(
                    parsed.get(
                        "predicted_emotion",
                        "",
                    )
                ).strip().lower()

            legal_label = (
                prediction in case["_pair_labels"]
            )

            correct = (
                prediction
                == case["ground_truth_emotion"]
                if legal_label
                else False
            )

            first_label = (
                case["_displayed_labels"][0]
            )

            result = {
                "case_index": index,
                "case_id": case_id,
                "pair_name": case["_pair_name"],
                "sample_id": case["sample_id"],
                "language": case["language"],
                "audio_path": case["audio_path"],
                "audio_sha256": case["audio_sha256"],
                "audio_hash_verified": True,
                "audio_metadata": audio_metadata,
                "transcript": case.get(
                    "transcript",
                    "",
                ),
                "ground_truth_emotion": case[
                    "ground_truth_emotion"
                ],
                "pair_labels": case[
                    "_pair_labels"
                ],
                "displayed_labels": case[
                    "_displayed_labels"
                ],
                "first_listed_label": first_label,
                "predicted_emotion": prediction,
                "legal_json": parsed is not None,
                "legal_label": legal_label,
                "correct": correct,
                "predicted_first_listed": (
                    prediction == first_label
                    if legal_label
                    else False
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
                "stage1_predictions_sha256": (
                    EXPECTED_STAGE1_SHA256
                ),
            }

            completed[case_id] = result

            ordered_results = [
                completed[row["_case_id"]]
                for row in cases
                if row["_case_id"] in completed
            ]

            write_jsonl(
                PREDICTIONS_FILE,
                ordered_results,
            )

            print(
                f"[{index:02d}/{len(cases)}] "
                f"{case['_pair_name']:<21s} "
                f"true="
                f"{case['ground_truth_emotion']:<8s} "
                f"pred="
                f"{str(prediction):<8s} "
                f"correct="
                f"{str(correct):<5s} "
                f"first="
                f"{first_label:<8s} "
                f"time="
                f"{inference_seconds:.3f}s",
                flush=True,
            )

            del inputs
            del output
            torch.cuda.empty_cache()

    if len(completed) != len(cases):
        raise RuntimeError(
            "pairwise 结果不完整："
            f"{len(completed)}/{len(cases)}"
        )

    ordered_results = [
        completed[row["_case_id"]]
        for row in cases
    ]

    pair_metrics = {
        pair_name: calculate_pair_metrics(
            pair_name,
            ordered_results,
        )
        for pair_name in PAIR_CONFIGS
    }

    summary = {
        "prompt_version": PROMPT_VERSION,
        "stage1_predictions": str(
            STAGE1_PREDICTIONS
        ),
        "stage1_predictions_sha256": (
            EXPECTED_STAGE1_SHA256
        ),
        "case_count": len(ordered_results),
        "pair_metrics": pair_metrics,
        "reserve_accessed": False,
        "predictions_sha256": sha256_file(
            PREDICTIONS_FILE
        ),
    }

    atomic_write_json(
        SUMMARY_FILE,
        summary,
    )

    print("\n" + "=" * 100)
    print("英文 pairwise 能力诊断结果")
    print("=" * 100)

    for pair_name, metrics in (
        pair_metrics.items()
    ):
        print()
        print(pair_name)
        print(
            "  样本数量             :",
            metrics["sample_count"],
        )
        print(
            "  合法标签率           :",
            f"{metrics['legal_rate']:.6f}",
        )
        print(
            "  准确率               :",
            f"{metrics['correct_count']}/"
            f"{metrics['sample_count']} "
            f"= {metrics['accuracy']:.6f}",
        )
        print(
            "  balanced accuracy    :",
            f"{metrics['balanced_accuracy']:.6f}",
        )
        print(
            "  各标签 recall        :",
            metrics["recall_by_label"],
        )
        print(
            "  真实标签分布         :",
            metrics["true_distribution"],
        )
        print(
            "  预测标签分布         :",
            metrics[
                "predicted_distribution"
            ],
        )
        print(
            "  选择首列标签比例     :",
            f"{metrics['first_listed_choice_rate']:.6f}",
        )
        print(
            "  混淆矩阵             :",
            metrics["confusion_matrix"],
        )

    print()
    print("预测文件               :", PREDICTIONS_FILE)
    print("汇总文件               :", SUMMARY_FILE)
    print(
        "预测文件 SHA256        :",
        sha256_file(PREDICTIONS_FILE),
    )


if __name__ == "__main__":
    main()
