from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# 复用已经通过验证的二阶段运行、断点续跑和指标计算逻辑。
import run_stage2_en_happy_vs_excited as engine  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]

PROMPT_VERSION = (
    "stage2_zh_happy_vs_surprise_playfulness_v1"
)

RESULT_DIR = (
    ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/"
      "calibration452_stage2/"
      "zh_happy_vs_surprise_playfulness_v1"
)


def build_stage2_prompt(row: dict) -> str:
    transcript = row.get("transcript", "")

    return f"""
你正在进行一次聚焦的第二阶段语音情感判断。

第一阶段系统预测为 happy，但该预测可能正确，也可能把 surprise 或
playfulness 错判成了 happy。请独立重新判断音频，只能从以下三个标签中选择：

- happy
- surprise
- playfulness

转写文本只用于对齐语音内容。不得根据句子表达的事件、词语含义、正面内容、
负面内容、人物关系或对话情境推断情感。判断必须主要依据声音本身。

请重点分析：

- 音高是否突然升高或发生突变
- 能量和音量是否突然增强
- 语速、停顿和节奏是否出现意外变化
- 声音是否有“带笑感”、戏谑感或夸张的旋律轮廓
- 情感是持续的愉快，还是短暂而突发的惊讶
- 语气是轻松调侃，还是普通的积极情绪

类别定义：

happy：
持续的愉快、温暖、满足、友好或积极感。声音可以明亮、有活力，但整体比较
自然、稳定，不具有明显的突然受惊反应，也不以戏谑和调侃为主。

surprise：
突然的惊讶、意外或受惊反应。通常具有短暂而明显的音高、能量、时长或节奏
突变，例如突然提高音调、突然加重、短促感叹或明显的惊异起伏。

判断 surprise 时必须存在声音层面的“突然性”。仅仅因为句子内容描述了意外、
疑问或新消息，不能选择 surprise。

playfulness：
调侃、开玩笑、淘气、轻松讽刺或故意夸张。声音通常具有带笑感、轻松感、
戏谑性的旋律起伏、拖长音、节奏变化或夸张表达，但没有真正的受惊反应。

关键区分规则：

1. surprise 与 happy：
   突然且明显的音高、能量或节奏变化选择 surprise；
   持续、稳定的积极愉快选择 happy。

2. playfulness 与 happy：
   明显存在调侃、戏谑、带笑或故意夸张的表达时选择 playfulness；
   普通友好、满足或愉快选择 happy。

3. surprise 与 playfulness：
   短暂、突发的惊异反应选择 surprise；
   持续的调侃、戏谑或夸张表演感选择 playfulness。

4. 不要仅因为音频声音明亮或语义积极就选择 happy。

转写文本：
{transcript}

只允许输出以下三个 JSON 之一：

{{"predicted_emotion":"happy"}}

{{"predicted_emotion":"surprise"}}

{{"predicted_emotion":"playfulness"}}

不得输出置信度、解释、Markdown 或任何额外文字。
""".strip()


def configure_engine() -> None:
    engine.PROMPT_VERSION = PROMPT_VERSION

    engine.TRIGGER_LANGUAGE = "zh"
    engine.TRIGGER_LABEL = "happy"

    engine.CANDIDATE_LABELS = [
        "happy",
        "surprise",
        "playfulness",
    ]

    engine.EXPECTED_TRIGGER_COUNT = 83

    engine.RESULT_DIR = RESULT_DIR

    engine.STAGE2_PREDICTIONS = (
        RESULT_DIR / "stage2_predictions.jsonl"
    )

    engine.COMBINED_PREDICTIONS = (
        RESULT_DIR / "combined_predictions.jsonl"
    )

    engine.METRICS_FILE = (
        RESULT_DIR / "metrics.json"
    )

    engine.build_stage2_prompt = (
        build_stage2_prompt
    )


def add_focused_metrics() -> dict:
    metrics = json.loads(
        engine.METRICS_FILE.read_text(
            encoding="utf-8"
        )
    )

    baseline = metrics["baseline_metrics"]
    hybrid = metrics["hybrid_metrics"]

    baseline_zh = baseline[
        "language_metrics"
    ]["zh"]

    hybrid_zh = hybrid[
        "language_metrics"
    ]["zh"]

    labels = [
        "happy",
        "surprise",
        "playfulness",
    ]

    class_metrics = {}

    for label in labels:
        baseline_class = baseline_zh[
            "per_class"
        ][label]

        hybrid_class = hybrid_zh[
            "per_class"
        ][label]

        class_metrics[label] = {
            "baseline_precision": (
                baseline_class["precision"]
            ),
            "baseline_recall": (
                baseline_class["recall"]
            ),
            "baseline_f1": (
                baseline_class["f1"]
            ),
            "hybrid_precision": (
                hybrid_class["precision"]
            ),
            "hybrid_recall": (
                hybrid_class["recall"]
            ),
            "hybrid_f1": (
                hybrid_class["f1"]
            ),
            "f1_delta": (
                hybrid_class["f1"]
                - baseline_class["f1"]
            ),
        }

    focused_deltas = {
        "zh_macro_f1": (
            hybrid_zh["macro_f1"]
            - baseline_zh["macro_f1"]
        ),
        "selection_score": (
            hybrid["selection_score"]
            - baseline["selection_score"]
        ),
        "overall_accuracy": (
            hybrid["accuracy"]
            - baseline["accuracy"]
        ),
        "classes": class_metrics,
    }

    metrics[
        "focused_zh_happy_surprise_playfulness"
    ] = focused_deltas

    engine.atomic_write_json(
        engine.METRICS_FILE,
        metrics,
    )

    return metrics


def print_focused_summary(metrics: dict) -> None:
    trigger = metrics["trigger"]
    baseline = metrics["baseline_metrics"]
    hybrid = metrics["hybrid_metrics"]

    baseline_zh = baseline[
        "language_metrics"
    ]["zh"]

    hybrid_zh = hybrid[
        "language_metrics"
    ]["zh"]

    focused = metrics[
        "focused_zh_happy_surprise_playfulness"
    ]

    print("\n" + "=" * 100)
    print(
        "中文 happy–surprise–playfulness "
        "二阶段实验结果"
    )
    print("=" * 100)

    print(
        "触发样本数量           :",
        trigger["sample_count"],
    )
    print(
        "二阶段合法标签         :",
        f"{trigger['legal_stage2_count']}/"
        f"{trigger['sample_count']} "
        f"= {trigger['legal_stage2_rate']:.6f}",
    )
    print(
        "预测发生改变           :",
        trigger["changed_count"],
    )
    print(
        "第一阶段触发集正确     :",
        f"{trigger['stage1_correct_count']}/"
        f"{trigger['sample_count']} "
        f"= {trigger['stage1_accuracy']:.6f}",
    )
    print(
        "二阶段后触发集正确     :",
        f"{trigger['final_correct_count']}/"
        f"{trigger['sample_count']} "
        f"= {trigger['final_accuracy']:.6f}",
    )
    print(
        "修复成功数量 gained    :",
        trigger["gained_count"],
    )
    print(
        "破坏正确数量 lost      :",
        trigger["lost_count"],
    )
    print(
        "净增益 net gain        :",
        trigger["net_gain_count"],
    )
    print(
        "二阶段预测分布         :",
        trigger[
            "stage2_prediction_distribution"
        ],
    )

    print()
    print(
        "baseline 总体准确率    :",
        f"{baseline['accuracy']:.6f}",
    )
    print(
        "hybrid 总体准确率      :",
        f"{hybrid['accuracy']:.6f}",
    )
    print(
        "总体准确率变化         :",
        f"{focused['overall_accuracy']:+.6f}",
    )

    print()
    print(
        "baseline ZH macro-F1   :",
        f"{baseline_zh['macro_f1']:.6f}",
    )
    print(
        "hybrid ZH macro-F1     :",
        f"{hybrid_zh['macro_f1']:.6f}",
    )
    print(
        "ZH macro-F1 变化       :",
        f"{focused['zh_macro_f1']:+.6f}",
    )

    for label in [
        "happy",
        "surprise",
        "playfulness",
    ]:
        result = focused["classes"][label]

        print()
        print(
            f"baseline {label:11s} F1 :",
            f"{result['baseline_f1']:.6f}",
        )
        print(
            f"hybrid {label:13s} F1 :",
            f"{result['hybrid_f1']:.6f}",
        )
        print(
            f"{label:11s} F1 变化 :",
            f"{result['f1_delta']:+.6f}",
        )

    print()
    print(
        "baseline selection     :",
        f"{baseline['selection_score']:.6f}",
    )
    print(
        "hybrid selection       :",
        f"{hybrid['selection_score']:.6f}",
    )
    print(
        "selection 变化         :",
        f"{focused['selection_score']:+.6f}",
    )

    print()
    print(
        "二阶段预测文件         :",
        engine.STAGE2_PREDICTIONS,
    )
    print(
        "组合预测文件           :",
        engine.COMBINED_PREDICTIONS,
    )
    print(
        "指标文件               :",
        engine.METRICS_FILE,
    )


def main() -> None:
    configure_engine()

    # 复用原有二阶段引擎。其内部旧标题可以忽略，
    # 最终以本脚本新增的中文聚焦汇总为准。
    engine.main()

    metrics = add_focused_metrics()
    print_focused_summary(metrics)


if __name__ == "__main__":
    main()
