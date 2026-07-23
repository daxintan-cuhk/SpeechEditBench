"""
情感准确率（Emotion Accuracy, EA）

判断编辑后语音的情感是否与 anchor 中 target_emotion 一致。

【当前实现】可切换的多模态 Judge
  将编辑后音频、编辑指令、转写文本一并提交给多模态 judge，
  要求输出 predicted_emotion/confidence/rationale JSON，并将预测标签与
  anchor.target_emotion 比对得到 EA。

【后续可选增强】
  方案 A — 情感识别模型（Audio Classifier）
    直接对编辑后音频做情感分类，判断预测标签是否与 target_emotion 一致。
    候选模型（待验证）：
      中文：待调研（如 Chinese speech emotion recognition 相关模型）
      英文：待调研（如 wav2vec2-based SER 模型）
    优点：客观、可复现；缺点：模型标签体系可能与数据集不对齐，需要手动映射。

  方案 B — LLM Judge
    将编辑指令 + 编辑后语音的 ASR 转写文本发给 LLM，由 LLM 判断情感是否达到预期。
    优点：灵活，能理解指令语义；缺点：依赖 LLM，随模型版本变化，可复现性稍差。

  方案 C — 两者结合
    音频分类器给出客观分数，LLM judge 给出语义判断，两者综合报告。

当前 predict() 支持 Gemini-compatible 与冻结的本地 Qwen2.5-Omni Hybrid v2 后端。通过环境变量 SPEECHEDITBENCH_EMOTION_JUDGE 选择；默认保持 Gemini。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal


from eval.metrics.llm_multimodal import call_multimodal_llm_json


EMOTION_JUDGE_ENV = (
    "SPEECHEDITBENCH_EMOTION_JUDGE"
)

DEFAULT_EMOTION_JUDGE_BACKEND = "gemini"

SUPPORTED_EMOTION_JUDGE_BACKENDS = {
    "gemini",
    "qwen25_omni_hybrid_v2",
}


def get_emotion_judge_backend() -> str:
    """
    返回当前情感评测后端。

    环境变量：
        SPEECHEDITBENCH_EMOTION_JUDGE=gemini
        SPEECHEDITBENCH_EMOTION_JUDGE=qwen25_omni_hybrid_v2

    默认值保持为 gemini，确保现有评测命令向后兼容。
    """
    backend = os.getenv(
        EMOTION_JUDGE_ENV,
        DEFAULT_EMOTION_JUDGE_BACKEND,
    ).strip().lower()

    if (
        backend
        not in SUPPORTED_EMOTION_JUDGE_BACKENDS
    ):
        supported = ", ".join(
            sorted(
                SUPPORTED_EMOTION_JUDGE_BACKENDS
            )
        )

        raise ValueError(
            f"不支持的情感评测后端："
            f"{backend!r}。"
            f"可选值：{supported}。"
        )

    return backend

EMOTION_ALIASES = {
    "fear": "fearful",
    "fearful": "fearful",
    "surprised": "surprise",
    "playful": "playfulness",
    "playfulness": "playfulness",
    "frustration": "frustrated",
    "neutral": "neutral",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "excited": "excited",
    "frustrated": "frustrated",
    "surprise": "surprise",
}


def _normalize_emotion(label: str) -> str:
    key = re.sub(r"[\s_]+", "-", (label or "").strip().lower())
    return EMOTION_ALIASES.get(key, key)


ALLOWED_BY_TAXONOMY = {
    "iemocap_en": [
        "angry", "excited", "fearful", "frustrated",
        "happy", "neutral", "sad", "surprise",
    ],
    "csemotions_zh": [
        "angry", "fearful", "happy", "neutral",
        "playfulness", "sad", "surprise",
    ],
}


def allowed_emotions_for_sample(sample: dict | None = None, language: str = "en") -> list[str]:
    if sample:
        taxonomy = (
            sample.get("emotion_taxonomy")
            or (sample.get("anchor") or {}).get("emotion_taxonomy")
        )
        if taxonomy in ALLOWED_BY_TAXONOMY:
            return ALLOWED_BY_TAXONOMY[taxonomy]
        source_dataset = sample.get("source_dataset")
        if source_dataset in {"iemocap", "libritts_test_clean"}:
            return ALLOWED_BY_TAXONOMY["iemocap_en"]
        if source_dataset in {"csemotions", "aishell3_test"}:
            return ALLOWED_BY_TAXONOMY["csemotions_zh"]
    if language == "zh":
        return ALLOWED_BY_TAXONOMY["csemotions_zh"]
    return ALLOWED_BY_TAXONOMY["iemocap_en"]


def _emotion_judge_prompt(language: Literal["zh", "en"], transcript: str, allowed: list[str]) -> str:
    allowed_str = ", ".join(allowed)
    if language == "zh":
        return (
            "你是语音情感评测专家。请仅根据音频实际表达判断主导情感。"
            "不要根据文本内容臆测情感。\n\n"
            f"转写文本: {transcript}\n"
            f"可选标签: {allowed_str}\n\n"
            "仅输出 JSON:\n"
            '{"predicted_emotion":"<label>","confidence":<0.0-1.0>,"rationale":"<一句话>"}'
        )
    return (
        "You are an expert speech-emotion evaluator. Judge dominant emotion from audio delivery only. "
        "Do not infer from text semantics alone.\n\n"
        f"Transcript: {transcript}\n"
        f"Allowed labels: {allowed_str}\n\n"
        "Return JSON only:\n"
        '{"predicted_emotion":"<label>","confidence":<0.0-1.0>,"rationale":"<one sentence>"}'
    )


# ── 预测接口 ─────────────────────────────────────────────────────────────────

def predict(
    audio_path: Path,
    language: Literal["zh", "en"],
    instruction: str,
    transcript: str,
    sample: dict | None = None,
) -> str:
    """
    预测编辑后语音的情感标签。

    参数：
        audio_path:  编辑后语音文件路径
        language:    语言（"zh" 或 "en"）
        instruction: 原始编辑指令；情感 judge 不使用该字段
        transcript:  编辑后语音的 ASR 转写文本
        sample:      用于确定语言标签体系和稳定 sample_id

    返回：
        情感标签字符串，如 "angry"、"happy" 等；
        非法输出统一返回 "unknown"。

    后端选择：
        SPEECHEDITBENCH_EMOTION_JUDGE=gemini
        SPEECHEDITBENCH_EMOTION_JUDGE=qwen25_omni_hybrid_v2

    两种后端均不读取 anchor.target_emotion，保持 blind
    emotion classification。
    """
    del instruction

    allowed = allowed_emotions_for_sample(
        sample,
        language,
    )

    backend = get_emotion_judge_backend()

    if backend == "gemini":
        prompt = _emotion_judge_prompt(
            language,
            transcript,
            allowed,
        )

        obj = call_multimodal_llm_json(
            audio_path,
            prompt,
            caller_tag=(
                "eval_emotion_accuracy"
            ),
        )

        pred = _normalize_emotion(
            str(
                obj.get(
                    "predicted_emotion",
                    "",
                )
            ).strip()
        )

        return (
            pred
            if pred in allowed
            else "unknown"
        )

    # 延迟导入：只有明确选择本地 Qwen 后端时，
    # 才导入生产模块。导入本身仍不会加载模型；
    # 模型在首次实际推理时才懒加载。
    from eval.metrics.qwen25_omni_emotion_judge import (
        predict_emotion,
    )

    sample_id = str(
        (sample or {}).get(
            "sample_id",
            "",
        )
    )

    pred = _normalize_emotion(
        predict_emotion(
            audio_path,
            language=language,
            transcript=transcript,
            allowed_labels=allowed,
            sample_id=sample_id,
        )
    )

    return (
        pred
        if pred in allowed
        else "unknown"
    )


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def compute_emotion_accuracy(sample: dict, predicted_emotion: str) -> dict:
    """
    计算单条样本的情感准确率。

    参数：
        sample:            samples.jsonl 中的一条记录
        predicted_emotion: predict() 返回的预测情感标签

    返回：
        {
            "sample_id":         str,
            "source_emotion":    str,
            "target_emotion":    str,
            "predicted_emotion": str,
            "correct":           bool,
            "source_dataset":    str,  # 顶层字段 source_dataset，用于 by_dataset 汇总
            "subset":            str,  # 顶层字段 subset，用于 by_subset 汇总
        }
    """
    target_raw = sample["anchor"]["target_emotion"]
    source_raw = sample["anchor"]["source_emotion"]
    target = _normalize_emotion(target_raw)
    source = _normalize_emotion(source_raw)
    pred = _normalize_emotion(predicted_emotion)
    return {
        "sample_id":         sample["sample_id"],
        "source_emotion":    source,
        "target_emotion":    target,
        "predicted_emotion": pred,
        "correct":           pred == target,
        "source_dataset":    sample.get("source_dataset", "unknown"),
        "subset":            sample.get("subset", "unknown"),
        "emotion_taxonomy":   sample.get("emotion_taxonomy")
                              or sample["anchor"].get("emotion_taxonomy", "unknown"),
    }
