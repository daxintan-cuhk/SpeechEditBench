"""
内容准确率（Content Accuracy, CA）

判断编辑后语音的文字内容是否与 anchor 中 transcript_target 一致。

【当前实现】本地 ASR + 文本指标

  步骤一 — ASR 转写
    - 英文：Transformers Whisper large-v3
      eval_models/asr/whisper-large-v3
    - 中文：FunASR Paraformer zh
      eval_models/asr/paraformer-zh

  步骤二 — 文本比对
    将 transcript_predicted 与 anchor.transcript_target 做整句比对。

  主要指标：
    - Exact Match（EM）：normalize 后的整句 ASR 结果与 transcript_target 完全一致
    - Edit Success（ES）：局部编辑片段是否满足 replace / insert / delete 预期
    - WER（Word Error Rate）：整句词错率，英文主诊断指标
    - CER（Character Error Rate）：整句字错率，中文主诊断指标

【注意】
  - 不计算 LF（Local Fidelity）：编辑区域外的语音不变性指标需要时间戳对齐，
    对 insert / delete 操作尤其困难（时间轴会整体偏移），暂列为未来扩展方向。
  - EM 和 ES 分开报告：EM 衡量整句严格一致；ES 衡量目标编辑是否达成。
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any


# ── 文本规范化 ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_MODEL_DIR = REPO_ROOT / "eval_models" / "asr" / "whisper-large-v3"
PARAFORMER_MODEL_DIR = REPO_ROOT / "eval_models" / "asr" / "paraformer-zh"


def normalize(text: str, language: str = "en") -> str:
    """
    统一化文本，用于 ASR 结果与 transcript_target 的字符串对比。

    处理项：
      - Unicode NFKC 规范化
      - 全角字符转半角
      - 去除首尾空白
      - 英文：统一小写，去除标点
      - 中文：去除空格与标点，保留汉字和数字
    """
    text = unicodedata.normalize("NFKC", text).strip()
    if language == "en":
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)
    return text


# ── ASR 预测接口 ──────────────────────────────────────────────────────────────

def _require_model_dir(path: Path, name: str, required_files: list[str]) -> None:
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{name} evaluation model is incomplete under {path}; missing: {missing}"
        )


@lru_cache(maxsize=1)
def _get_whisper_model() -> tuple[Any, Any]:
    """Lazy-load Whisper large-v3 through Transformers."""
    _require_model_dir(
        WHISPER_MODEL_DIR,
        "Whisper large-v3",
        ["config.json", "preprocessor_config.json", "tokenizer_config.json"],
    )

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:*")
        cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_DIR, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "use_safetensors": True,
        "variant": "fp32",
        "dtype": torch.float32,
    }
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_DIR, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_DIR, **model_kwargs)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, processor


@lru_cache(maxsize=1)
def _get_paraformer_model() -> Any:
    """Lazy-load Chinese Paraformer through FunASR."""
    _require_model_dir(
        PARAFORMER_MODEL_DIR,
        "Paraformer zh",
        ["config.yaml", "configuration.json", "model.pt", "tokens.json"],
    )

    from funasr import AutoModel

    return AutoModel(model=str(PARAFORMER_MODEL_DIR), disable_update=True)


def asr_predict(audio_path: Path, language: str = "en") -> str:
    """
    对给定音频执行 ASR，返回转写文本。

    参数：
        audio_path: 待转写的音频文件路径
        language:   语言代码（"en" 或 "zh"）

    返回：ASR 转写的纯文本字符串
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if language == "zh":
        model = _get_paraformer_model()
        output = model.generate(input=str(audio_path), batch_size_s=300)
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
            return str(first).strip()
        return str(output).strip()

    import librosa
    model, processor = _get_whisper_model()
    import torch

    device = next(model.parameters()).device
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            language="english",
            task="transcribe",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    """最小编辑距离（Levenshtein），用于 WER/CER 计算。"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """词错率（Word Error Rate）。适合英文。"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """字错率（Character Error Rate）。适合中文。"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def compute_edit_success(sample: dict, transcript_predicted: str) -> tuple[bool, str]:
    """
    判断局部编辑是否达成。

    该指标只检查 anchor 中的目标片段/删除片段，不要求整句完全一致：
      - replace：目标片段出现，且原片段不再出现
      - insert：插入片段出现；若有 insert_after，则检查二者顺序正确
      - delete：被删除片段不再出现
    """
    anchor = sample["anchor"]
    language = sample.get("language", "en")
    edit_type = anchor.get("edit_type", "unknown")
    hyp = normalize(transcript_predicted, language)

    edit_original_raw = anchor.get("edit_original")
    edit_target_raw = anchor.get("edit_target")
    insert_after_raw = anchor.get("insert_after")

    edit_original = normalize(str(edit_original_raw), language) if edit_original_raw else ""
    edit_target = normalize(str(edit_target_raw), language) if edit_target_raw else ""
    insert_after = normalize(str(insert_after_raw), language) if insert_after_raw else ""

    if edit_type == "replace":
        if not edit_target:
            return False, "replace_missing_edit_target"
        target_ok = edit_target in hyp
        original_gone = (not edit_original) or (edit_original not in hyp)
        return target_ok and original_gone, (
            "replace_target_present_original_absent"
            if target_ok and original_gone
            else f"replace_target_present={target_ok};original_absent={original_gone}"
        )

    if edit_type == "insert":
        if not edit_target:
            return False, "insert_missing_edit_target"
        target_ok = edit_target in hyp
        if not insert_after:
            return target_ok, "insert_target_present" if target_ok else "insert_target_absent"
        after_idx = hyp.find(insert_after)
        target_idx = hyp.find(edit_target)
        order_ok = after_idx >= 0 and target_idx >= 0 and after_idx <= target_idx
        return target_ok and order_ok, (
            "insert_target_present_after_anchor"
            if target_ok and order_ok
            else f"insert_target_present={target_ok};after_before_target={order_ok}"
        )

    if edit_type == "delete":
        if not edit_original:
            return False, "delete_missing_edit_original"
        original_gone = edit_original not in hyp
        return original_gone, "delete_original_absent" if original_gone else "delete_original_present"

    return False, f"unsupported_edit_type={edit_type}"


def compute_content_accuracy(
    sample: dict,
    transcript_predicted: str,
) -> dict:
    """
    计算单条样本的内容准确率。

    参数：
        sample:               samples.jsonl 中的一条记录
        transcript_predicted: ASR 转写的预测文本

    返回：
        {
            "sample_id":             str,
            "edit_type":             str,    # replace / insert / delete
            "transcript_target":     str,    # ground truth
            "transcript_predicted":  str,    # ASR 输出
            "norm_target":           str,    # normalize 后的 ground truth
            "norm_predicted":        str,    # normalize 后的 ASR 输出
            "exact_match":           bool,   # normalize 后完全一致
            "edit_success":          bool,   # 局部编辑是否达成
            "wer":                   float,  # 词错率（英文）
            "cer":                   float,  # 字错率（中文）
            "source_dataset":        str,
            "language":              str,
        }
    """
    anchor   = sample["anchor"]
    language = sample.get("language", "en")

    tgt_raw  = anchor.get("transcript_target", "")
    norm_tgt = normalize(tgt_raw, language)
    norm_hyp = normalize(transcript_predicted, language)

    exact_match = norm_tgt == norm_hyp
    edit_success, edit_success_reason = compute_edit_success(sample, transcript_predicted)

    if language == "en":
        wer = compute_wer(norm_tgt, norm_hyp)
        cer = compute_cer(norm_tgt, norm_hyp)   # 英文 CER 仅供参考
    else:
        cer = compute_cer(norm_tgt, norm_hyp)
        wer = compute_wer(norm_tgt, norm_hyp)   # 中文 WER 仅供参考

    return {
        "sample_id":            sample["sample_id"],
        "edit_type":            anchor.get("edit_type", "unknown"),
        "transcript_target":    tgt_raw,
        "transcript_predicted": transcript_predicted,
        "norm_target":          norm_tgt,
        "norm_predicted":       norm_hyp,
        "exact_match":          exact_match,
        "edit_success":         edit_success,
        "edit_success_reason":  edit_success_reason,
        "wer":                  round(wer, 4),
        "cer":                  round(cer, 4),
        "source_dataset":       sample.get("source_dataset", "unknown"),
        "language":             language,
    }
