"""
ASR with timestamps for prosody/stress evaluation.

Text-only ASR for content/speaker evaluation remains in content_accuracy.py.
This module is intentionally separate because stress scoring needs token time
windows, and the Chinese timestamp backend uses a different Paraformer variant.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_MODEL_DIR = REPO_ROOT / "eval_models" / "asr" / "whisper-large-v3"
PARAFORMER_TIMESTAMP_MODEL_DIR = (
    REPO_ROOT / "eval_models" / "asr" / "paraformer-zh-vad-punc-timestamp"
)
TIMESTAMP_CACHE_DIR = REPO_ROOT / "eval_cache" / "asr_timestamps"


def _require_model_dir(path: Path, name: str, required_files: list[str]) -> None:
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{name} evaluation model is incomplete under {path}; missing: {missing}"
        )


@lru_cache(maxsize=1)
def _get_whisper_pipeline() -> Any:
    """Lazy-load Transformers Whisper pipeline with word timestamps enabled."""
    _require_model_dir(
        WHISPER_MODEL_DIR,
        "Whisper large-v3",
        ["config.json", "preprocessor_config.json", "tokenizer_config.json"],
    )

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:*")
        cuda_available = torch.cuda.is_available()

    device = 0 if cuda_available else -1
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
    model.eval()
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )


@lru_cache(maxsize=1)
def _get_paraformer_timestamp_model() -> Any:
    """Lazy-load FunASR Paraformer zh VAD/PUNC timestamp model."""
    _require_model_dir(
        PARAFORMER_TIMESTAMP_MODEL_DIR,
        "Paraformer zh VAD/PUNC timestamp",
        ["config.yaml", "configuration.json", "model.pt", "tokens.json"],
    )

    from funasr import AutoModel

    return AutoModel(model=str(PARAFORMER_TIMESTAMP_MODEL_DIR), disable_update=True)


def _clean_en_token(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9']+", "", text)
    return text


def _timestamp_to_seconds(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric / 1000.0 if numeric > 100.0 else numeric


def _parse_whisper_chunks(result: dict) -> list[dict]:
    tokens: list[dict] = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp")
        if not timestamp or len(timestamp) != 2:
            continue
        start = _timestamp_to_seconds(timestamp[0])
        end = _timestamp_to_seconds(timestamp[1])
        token_text = str(chunk.get("text", "")).strip()
        clean_text = _clean_en_token(token_text)
        if start is None or end is None or end <= start or not clean_text:
            continue
        tokens.append(
            {
                "text": token_text,
                "normalized": clean_text,
                "start": round(start, 4),
                "end": round(end, 4),
            }
        )
    return tokens


def _parse_paraformer_tokens(result: list | dict) -> tuple[str, list[dict]]:
    if isinstance(result, list) and result:
        first = result[0]
    elif isinstance(result, dict):
        first = result
    else:
        return "", []

    if not isinstance(first, dict):
        return str(first), []

    text = str(first.get("text", "")).strip()
    timestamps = first.get("timestamp") or []
    pieces = [piece for piece in text.split() if piece.strip()]
    if len(pieces) != len(timestamps):
        compact = re.sub(r"\s+", "", text)
        pieces = list(compact)

    tokens: list[dict] = []
    for piece, timestamp in zip(pieces, timestamps, strict=False):
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue
        start = _timestamp_to_seconds(timestamp[0])
        end = _timestamp_to_seconds(timestamp[1])
        if start is None or end is None or end <= start:
            continue
        tokens.append(
            {
                "text": piece,
                "normalized": re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", piece),
                "start": round(start, 4),
                "end": round(end, 4),
            }
        )
    return text, tokens


def _cache_backend_name(language: str) -> str:
    return "paraformer_zh_timestamp" if language == "zh" else "whisper_large_v3_word_timestamps"


def _cache_file(audio_path: Path, language: str) -> Path:
    resolved = audio_path.resolve()
    stat = resolved.stat()
    backend = _cache_backend_name(language)
    cache_key = hashlib.sha1(
        "||".join(
            [
                backend,
                language,
                str(resolved),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return TIMESTAMP_CACHE_DIR / backend / f"{cache_key}.json"


def _load_cached_result(audio_path: Path, language: str) -> dict | None:
    cache_file = _cache_file(audio_path, language)
    if not cache_file.exists():
        return None
    with cache_file.open(encoding="utf-8") as f:
        return json.load(f)


def _save_cached_result(audio_path: Path, language: str, payload: dict) -> None:
    cache_file = _cache_file(audio_path, language)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_file.replace(cache_file)


def asr_predict_with_timestamps(
    audio_path: str | Path,
    language: str = "en",
    *,
    use_cache: bool = True,
) -> dict:
    """
    Transcribe audio and return normalized token timestamps.

    Returns:
        {
          "text": str,
          "tokens": [{"text": str, "normalized": str, "start": float, "end": float}],
          "backend": str,
        }
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if use_cache:
        cached = _load_cached_result(audio_path, language)
        if cached is not None:
            return cached

    if language == "zh":
        model = _get_paraformer_timestamp_model()
        output = model.generate(input=str(audio_path), batch_size_s=300)
        text, tokens = _parse_paraformer_tokens(output)
        result = {
            "text": text,
            "tokens": tokens,
            "backend": "funasr_paraformer_zh_vad_punc_timestamp",
        }
        if use_cache:
            _save_cached_result(audio_path, language, result)
        return result

    pipe = _get_whisper_pipeline()
    result = pipe(
        str(audio_path),
        return_timestamps="word",
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    parsed = {
        "text": str(result.get("text", "")).strip(),
        "tokens": _parse_whisper_chunks(result),
        "backend": "transformers_whisper_large_v3_word_timestamps",
    }
    if use_cache:
        _save_cached_result(audio_path, language, parsed)
    return parsed
