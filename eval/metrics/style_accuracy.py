"""
风格准确率（Style Accuracy, SA）

判断编辑后语音的说话风格是否与 anchor 中 target_style 一致。

【候选评估方案】

  方案 A — Gemini 多模态 Judge（推荐）
    直接将编辑后音频发给 Gemini，要求其输出 6 维风格打分（0–4）。
    以打分最高的维度作为预测风格，与 target_style 比较。
    优点：与标注阶段使用同一模型，标准一致，且无需音频分类模型部署。
    缺点：依赖 API，不可离线运行；Gemini 输出存在随机性（可用 temperature=0 缓解）。

  方案 B — 专项音频风格分类器
    训练或使用开源说话风格识别模型，对音频做 6 类分类。
    优点：本地运行，无 API 依赖，可批量处理。
    缺点：目前没有现成的对齐本项目 6 类标签体系的预训练模型，需要微调。

  方案 C — LLM Judge（文本）
    将编辑指令 + ASR 转写文本发给 LLM，询问"该文本更可能以哪种风格朗读"。
    优点：无音频 API 依赖，简单快速。
    缺点：说话风格主要体现在声学层面，仅靠文本判断准确性有限；
          对 standard set 尤其不可靠（文本中性时文本无法区分风格）。

  方案 D — A + C 组合
    Gemini 对音频打分（方案 A）+ LLM 文本判断（方案 C），两者综合报告。

【当前实现】方案 A 的 target-conditioned 版本（Gemini-compatible 多模态 Judge）。
主指标不再使用开放式 top-1 风格分类，而是直接判断输出语音是否明显
呈现目标风格；同时保留 6 维 style_scores 作为诊断字段。
当前 predict() 已接入 eval/metrics/llm_multimodal.py；运行前需配置 GEMINI_API_KEY。
注意：Gemini 同时用于部分标注与当前 smoke 评测时，报告中应标注自我验证偏差风险。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).parent.parent.parent
from eval.resources.style_taxonomy import ALL_STYLES, STYLE_DESCRIPTIONS
from eval.metrics.llm_multimodal import call_multimodal_llm_json

TARGET_STYLE_SUCCESS_THRESHOLD = 3


def _style_definition(target_style: str, language: str) -> str:
    lang = "zh" if language == "zh" else "en"
    return STYLE_DESCRIPTIONS.get(target_style, {}).get(lang, target_style)


def _build_style_prompt(target_style: str, language: str, transcript: str | None) -> str:
    transcript = (transcript or "").strip()
    target_definition = _style_definition(target_style, language)
    if transcript:
        transcript_block = f"\n\n[Transcript / 文本转录]\n{transcript}"
    else:
        transcript_block = "\n\n[Transcript / 文本转录]\n(Not available. Judge based on audio delivery alone.)"

    if language == "zh":
        return (
            "你是语音风格编辑评测专家。请听音频，判断编辑后的语音是否明显呈现目标说话风格。\n\n"
            "重要原则：\n"
            "- 只评估说话方式、声音投射、韵律形态、节奏和表达方式。\n"
            "- 不要根据文本内容、话题、词语含义来判断风格。\n"
            "- 不要判断情感类别；风格不是 happy/sad/angry/fearful。\n"
            "- 即使文本内容像故事、新闻或对话，也必须以音频表达方式为准。\n\n"
            f"目标风格: {target_style}\n"
            f"目标风格定义: {target_definition}"
            f"{transcript_block}\n\n"
            "请同时给出：\n"
            "1. target_style_score：目标风格在音频中的强度，整数 0-4。\n"
            "2. target_style_success：只有当目标风格清楚存在并达到 strong 级别时才为 true。通常 target_style_score >= 3 才能为 true。\n"
            "3. style_scores：6 种风格的诊断分数，整数 0-4。\n"
            "4. dominant_style：音频中最主要的风格标签。\n\n"
            "仅输出 JSON，不要输出 markdown 或额外解释：\n"
            "{\n"
            '  "target_style_score": <0-4>,\n'
            '  "target_style_success": <true|false>,\n'
            '  "dominant_style": "<one of the 6 labels or uncertain>",\n'
            '  "style_scores": {\n'
            '    "public-broadcast": <0-4>,\n'
            '    "intimate": <0-4>,\n'
            '    "dramatic": <0-4>,\n'
            '    "restrained-flat": <0-4>,\n'
            '    "storytelling": <0-4>,\n'
            '    "conversational": <0-4>\n'
            '  },\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "rationale": "<一句话说明听到的具体声学/韵律线索>"\n'
            "}"
        )

    return (
        "You are an expert evaluator for speech style editing. Listen to the audio and judge "
        "whether the edited speech clearly exhibits the target speaking style.\n\n"
        "Important rules:\n"
        "- Evaluate only vocal delivery: projection, prosody shape, rhythm, pacing, and speaking manner.\n"
        "- Do NOT judge from the text content, topic, or word meanings.\n"
        "- Do NOT annotate emotion categories; style is not happy/sad/angry/fearful.\n"
        "- Even if the transcript sounds like a story, news, or dialogue, base your judgment on the audio delivery.\n\n"
        f"Target style: {target_style}\n"
        f"Target style definition: {target_definition}"
        f"{transcript_block}\n\n"
        "Return the following fields:\n"
        "1. target_style_score: integer 0-4 for how strongly the target style is present.\n"
        "2. target_style_success: true only when the target style is clearly present at a strong level. "
        "Normally this requires target_style_score >= 3.\n"
        "3. style_scores: diagnostic 0-4 scores for all six style labels.\n"
        "4. dominant_style: the dominant style in the audio.\n\n"
        "Return JSON only, with no markdown or extra text:\n"
        "{\n"
        '  "target_style_score": <0-4>,\n'
        '  "target_style_success": <true|false>,\n'
        '  "dominant_style": "<one of the 6 labels or uncertain>",\n'
        '  "style_scores": {\n'
        '    "public-broadcast": <0-4>,\n'
        '    "intimate": <0-4>,\n'
        '    "dramatic": <0-4>,\n'
        '    "restrained-flat": <0-4>,\n'
        '    "storytelling": <0-4>,\n'
        '    "conversational": <0-4>\n'
        '  },\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "rationale": "<one sentence explaining the concrete acoustic/prosodic cues you heard>"\n'
        "}"
    )


def _parse_style_scores(obj: dict) -> dict[str, int]:
    raw = obj.get("style_scores", {}) or {}
    parsed: dict[str, int] = {}
    for s in ALL_STYLES:
        try:
            v = int(round(float(raw.get(s, 0))))
        except Exception:  # noqa: BLE001
            v = 0
        parsed[s] = max(0, min(4, v))
    return parsed


def _parse_score(value: object) -> int:
    try:
        score = int(round(float(value)))
    except Exception:  # noqa: BLE001
        score = 0
    return max(0, min(4, score))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "success", "successful"}
    return False


# ── 预测接口 ─────────────────────────────────────────────────────────────────

def predict(
    audio_path: Path,
    language: Literal["zh", "en"],
    instruction: str,
    transcript: str | None,
    target_style: str,
) -> dict:
    """
    预测编辑后语音的说话风格标签。

    当前使用 Gemini-compatible 多模态 judge：
      - 根据 target_style 构造 target-conditioned prompt
      - 将 audio_path 的音频编码为 base64
      - 调用 API，获取 target_style_success 与 6 维 style_scores
      - 返回完整诊断字典

    参数：
        audio_path:  编辑后语音文件路径
        language:    语言（"zh" 或 "en"），保留给未来 prompt 分流
        instruction: 原始编辑指令，保留给接口兼容，不进入当前 judge prompt
        transcript:  编辑后语音的 ASR 转写文本，可为 None
        target_style: 样本 anchor 中的目标风格

    返回：target-conditioned judge 结果字典。
    """
    prompt = _build_style_prompt(target_style, language, transcript)
    obj = call_multimodal_llm_json(
        audio_path,
        prompt,
        caller_tag="eval_style_target_success",
    )
    scores = _parse_style_scores(obj)
    target_score = _parse_score(obj.get("target_style_score", scores.get(target_style, 0)))
    dominant_style = str(obj.get("dominant_style", "")).strip().lower()
    if dominant_style not in ALL_STYLES:
        dominant_style = max(scores, key=lambda k: scores[k]) if scores else "uncertain"

    success = _parse_bool(obj.get("target_style_success")) and target_score >= TARGET_STYLE_SUCCESS_THRESHOLD
    if not success and obj.get("target_style_success") is None:
        success = target_score >= TARGET_STYLE_SUCCESS_THRESHOLD

    return {
        "target_style_score": target_score,
        "target_style_success": success,
        "dominant_style": dominant_style,
        "style_scores": scores,
        "confidence": obj.get("confidence"),
        "rationale": obj.get("rationale", ""),
    }


def _legacy_prediction_to_result(predicted: str | dict, target: str) -> dict:
    if isinstance(predicted, dict):
        scores = predicted.get("style_scores") or {}
        predicted_style = predicted.get("dominant_style") or (
            max(scores, key=lambda k: scores[k]) if scores else "uncertain"
        )
        target_score = predicted.get("target_style_score")
        target_success = bool(predicted.get("target_style_success"))
        return {
            "predicted_style": predicted_style,
            "correct": target_success,
            "target_style_score": target_score,
            "target_style_success": target_success,
            "dominant_style": predicted_style,
            "style_scores": scores,
            "judge_confidence": predicted.get("confidence"),
            "judge_rationale": predicted.get("rationale"),
        }

    scores: dict[str, int] = {}
    predicted_style = str(predicted)
    return {
        "predicted_style": predicted_style,
        "correct": predicted_style.strip().lower() == target.strip().lower(),
        "target_style_score": None,
        "target_style_success": predicted_style.strip().lower() == target.strip().lower(),
        "dominant_style": predicted_style,
        "style_scores": scores,
        "judge_confidence": None,
        "judge_rationale": None,
    }


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def compute_style_accuracy(sample: dict, predicted_style: str | dict) -> dict:
    """
    计算单条样本的风格准确率。

    参数：
        sample:          samples.jsonl 中的一条记录
        predicted_style: predict() 返回的 target-conditioned judge 结果，
                         或旧版预测风格标签

    返回：
        {
            "sample_id":       str,
            "source_style":    str,
            "target_style":    str,
            "predicted_style": str,
            "correct":         bool,   # predicted_style == target_style
            "source_dataset":  str,
            "subset":          str,    # standard / challenging
        }
    """
    target = sample["anchor"]["target_style"]
    parsed = _legacy_prediction_to_result(predicted_style, target)
    return {
        "sample_id":       sample["sample_id"],
        "source_style":    sample["anchor"]["source_style"],
        "target_style":    target,
        "predicted_style": parsed["predicted_style"],
        "correct":         parsed["correct"],
        "target_style_score": parsed["target_style_score"],
        "target_style_success": parsed["target_style_success"],
        "dominant_style": parsed["dominant_style"],
        "style_scores": parsed["style_scores"],
        "judge_confidence": parsed["judge_confidence"],
        "judge_rationale": parsed["judge_rationale"],
        "source_dataset":  sample.get("source_dataset", "unknown"),
        "subset":          sample.get("subset", "unknown"),
    }
