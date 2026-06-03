"""
副语言准确率（Paralinguistic Accuracy, PA）

判断编辑后语音是否完成了目标副语言事件的添加或删除：
  - add  操作：目标事件在输出音频中应 present（分值 >= DETECT_THRESHOLD）
  - remove 操作：目标事件在输出音频中应 absent（分值 <= ABSENT_THRESHOLD）

【评估方案】Gemini 多模态 Judge（与标注阶段同模型）
  直接将编辑后音频发给 Gemini，使用同一套 4 维副语言事件打分（0–3）。
  根据 operation 和 DETECT_THRESHOLD / ABSENT_THRESHOLD 判断是否达标。
  优点：与标注阶段使用同一模型，标准一致，无需本地分类器部署。
  缺点：依赖 API，不可离线运行。

当前 predict() 已接入 eval/metrics/llm_multimodal.py；运行前需配置 GEMINI_API_KEY。
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
from eval.resources.paralinguistic_taxonomy import (
    ALL_EVENTS, DETECT_THRESHOLD, ABSENT_THRESHOLD
)
from eval.metrics.llm_multimodal import call_multimodal_llm_json

_PROMPT_PATH = _REPO_ROOT / "eval" / "resources" / "paralinguistic_annotation_prompt.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8").strip() if _PROMPT_PATH.exists() else ""


def _parse_event_scores(obj: dict) -> dict[str, int]:
    raw = obj.get("event_scores", {}) or {}
    parsed: dict[str, int] = {}
    for e in ALL_EVENTS:
        try:
            v = int(round(float(raw.get(e, 0))))
        except Exception:  # noqa: BLE001
            v = 0
        parsed[e] = max(0, min(3, v))
    return parsed


# ── 预测接口 ─────────────────────────────────────────────────────────────────

def predict(
    audio_path: Path,
    language: str,
) -> dict[str, int]:
    """
    检测编辑后语音中的副语言事件，返回 4 维打分字典。

    当前使用 Gemini-compatible 多模态 judge：
      - 加载 paralinguistic_annotation_prompt.txt
      - 将 audio_path 的音频编码为 base64
      - 调用 API，获取 event_scores（4 维，0–3）
      - 返回 {"breath": int, "laugh": int, "cough": int, "sigh": int}

    参数：
        audio_path: 编辑后语音文件路径
        language:   语言（"zh" 或 "en"），保留给未来 prompt 分流

    返回：{event: score} 字典，score 为 0–3 整数
    """
    prompt = _PROMPT_TEMPLATE
    obj = call_multimodal_llm_json(
        audio_path,
        prompt,
        caller_tag="eval_paralinguistic_accuracy",
    )
    return _parse_event_scores(obj)


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def compute_pa(sample: dict, predicted_scores: dict[str, int]) -> dict:
    """
    计算单条样本的副语言准确率。

    add  操作：score[target_event] >= DETECT_THRESHOLD → correct=True
    remove 操作：score[target_event] <= ABSENT_THRESHOLD → correct=True

    返回：
        {
            "sample_id":      str,
            "operation":      str,       # add / remove
            "target_event":   str,       # breath / laugh / cough / sigh
            "predicted_score": int,      # Gemini 给出的目标事件分值
            "correct":        bool,
            "source_dataset": str,
            "language":       str,
        }
    """
    anchor    = sample.get("anchor", {})
    operation = anchor.get("operation", "")
    event     = anchor.get("event", "")
    score     = predicted_scores.get(event, 0)

    if operation == "add":
        correct = score >= DETECT_THRESHOLD
    elif operation == "remove":
        correct = score <= ABSENT_THRESHOLD
    else:
        correct = False

    return {
        "sample_id":       sample["sample_id"],
        "operation":       operation,
        "target_event":    event,
        "predicted_score": score,
        "correct":         correct,
        "source_dataset":  sample.get("source_dataset", "unknown"),
        "language":        sample.get("language", "unknown"),
    }
