"""
Prosody editing metrics.

Implemented metrics:
  - speed: duration-ratio direction accuracy
  - pitch: median F0 semitone-shift direction accuracy
  - stress: timestamp-ASR target-window prominence gain
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

MIN_SPEED_DURATION_CHANGE = 0.05
MIN_PITCH_SHIFT_SEMITONE = 0.3
MIN_PROMINENCE_GAIN = 0.0
EPS = 1e-8


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def _duration(audio_path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return float(info.frames / info.samplerate)


@lru_cache(maxsize=256)
def _load_audio(audio_path: str) -> tuple[object, int]:
    import librosa

    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    return wav, sr


@lru_cache(maxsize=256)
def _rms_track(audio_path: str) -> tuple[object, object]:
    import librosa

    wav, sr = _load_audio(audio_path)
    hop_length = 256
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times


@lru_cache(maxsize=256)
def _f0_track(audio_path: str) -> tuple[object, object]:
    import librosa

    wav, sr = _load_audio(audio_path)
    hop_length = 256
    f0, _, _ = librosa.pyin(
        wav,
        fmin=50.0,
        fmax=500.0,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)
    return f0, times


def _values_in_window(values, times, start: float, end: float):
    import numpy as np

    values = np.asarray(values)
    times = np.asarray(times)
    mask = (times >= start) & (times <= end)
    return values[mask]


def _nanmedian(values) -> float | None:
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.median(values))


def _nanmean(values) -> float | None:
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median_f0(audio_path: str | Path, start: float | None = None, end: float | None = None) -> float | None:
    f0, times = _f0_track(str(Path(audio_path)))
    if start is not None and end is not None:
        f0 = _values_in_window(f0, times, start, end)
    return _nanmedian(f0)


def _voiced_ratio(audio_path: str | Path) -> float:
    import numpy as np

    f0, _ = _f0_track(str(Path(audio_path)))
    values = np.asarray(f0, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sum(~np.isnan(values) & (values > 0)) / values.size)


def _mean_rms(audio_path: str | Path, start: float | None = None, end: float | None = None) -> float | None:
    rms, times = _rms_track(str(Path(audio_path)))
    if start is not None and end is not None:
        rms = _values_in_window(rms, times, start, end)
    return _nanmean(rms)


def _normalize_en(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    return [piece for piece in text.split() if piece]


def _normalize_zh(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)


def _find_target_window(tokens: list[dict], target: str, language: str) -> dict | None:
    if language == "zh":
        target_norm = _normalize_zh(target)
        if not target_norm:
            return None

        joined = ""
        char_to_token: list[int] = []
        for idx, token in enumerate(tokens):
            token_norm = _normalize_zh(str(token.get("normalized") or token.get("text", "")))
            for char in token_norm:
                joined += char
                char_to_token.append(idx)

        start_char = joined.find(target_norm)
        if start_char < 0:
            return None
        token_indices = char_to_token[start_char:start_char + len(target_norm)]
        if not token_indices:
            return None
        matched = [tokens[idx] for idx in sorted(set(token_indices))]
    else:
        target_words = _normalize_en(target)
        token_words = [str(token.get("normalized", "")).lower() for token in tokens]
        matched = []
        for idx in range(0, len(token_words) - len(target_words) + 1):
            if token_words[idx:idx + len(target_words)] == target_words:
                matched = tokens[idx:idx + len(target_words)]
                break
        if not matched:
            return None

    return {
        "start": min(float(token["start"]) for token in matched),
        "end": max(float(token["end"]) for token in matched),
        "matched_text": " ".join(str(token.get("text", "")) for token in matched).strip(),
    }


def _prominence(audio_path: str | Path, start: float, end: float) -> dict:
    duration = _duration(audio_path)
    start = max(0.0, min(start, duration))
    end = max(start + 0.02, min(end, duration))
    segment_duration = end - start

    segment_rms = _mean_rms(audio_path, start, end)
    global_rms = _mean_rms(audio_path)
    segment_f0 = _median_f0(audio_path, start, end)
    global_f0 = _median_f0(audio_path)

    energy_score = math.log((segment_rms or EPS) / ((global_rms or EPS) + EPS) + EPS)
    f0_score = 0.0
    if segment_f0 and global_f0:
        f0_score = math.log2((segment_f0 + EPS) / (global_f0 + EPS))
    duration_score = math.log((segment_duration / max(duration, EPS)) + EPS)

    score = 0.5 * energy_score + 0.3 * f0_score + 0.2 * duration_score
    return {
        "score": score,
        "energy_score": energy_score,
        "f0_score": f0_score,
        "duration_score": duration_score,
        "rms": segment_rms,
        "median_f0": segment_f0,
        "duration": segment_duration,
        "start": start,
        "end": end,
    }


def predict_speed(
    input_wav: str | Path,
    output_wav: str | Path,
    direction: str,
    *,
    input_transcript: Optional[str] = None,
    output_transcript: Optional[str] = None,
    language: str = "en",
) -> dict:
    """Evaluate speed editing with output/source duration ratio."""
    del input_transcript, output_transcript, language

    input_duration = _duration(input_wav)
    output_duration = _duration(output_wav)
    duration_ratio = output_duration / input_duration if input_duration > 0 else None
    if duration_ratio is None:
        raise ValueError("input duration is zero")

    direction = direction.lower()
    if direction == "faster":
        direction_correct = duration_ratio < 1.0
        passed = duration_ratio <= (1.0 - MIN_SPEED_DURATION_CHANGE)
    elif direction == "slower":
        direction_correct = duration_ratio > 1.0
        passed = duration_ratio >= (1.0 + MIN_SPEED_DURATION_CHANGE)
    else:
        raise ValueError(f"unsupported speed direction: {direction!r}")

    return {
        "metric_type": "speed_duration_ratio",
        "passed": bool(passed),
        "direction_correct": bool(direction_correct),
        "input_duration": _round(input_duration),
        "output_duration": _round(output_duration),
        "duration_ratio": _round(duration_ratio),
        "change_ratio": _round(duration_ratio - 1.0),
        "threshold": MIN_SPEED_DURATION_CHANGE,
    }


def predict_pitch(
    input_wav: str | Path,
    output_wav: str | Path,
    direction: str,
    *,
    language: str = "en",
) -> dict:
    """Evaluate pitch editing with median F0 semitone shift."""
    del language

    input_f0 = _median_f0(input_wav)
    output_f0 = _median_f0(output_wav)
    if not input_f0 or not output_f0:
        raise ValueError("unable to compute voiced F0 median")

    shift_semitone = 12.0 * math.log2(output_f0 / input_f0)
    direction = direction.lower()
    if direction == "higher":
        direction_correct = shift_semitone > 0
        passed = shift_semitone >= MIN_PITCH_SHIFT_SEMITONE
    elif direction == "lower":
        direction_correct = shift_semitone < 0
        passed = shift_semitone <= -MIN_PITCH_SHIFT_SEMITONE
    else:
        raise ValueError(f"unsupported pitch direction: {direction!r}")

    return {
        "metric_type": "pitch_f0_shift",
        "passed": bool(passed),
        "direction_correct": bool(direction_correct),
        "input_f0_median": _round(input_f0),
        "output_f0_median": _round(output_f0),
        "shift_semitone": _round(shift_semitone),
        "input_voiced_ratio": _round(_voiced_ratio(input_wav)),
        "output_voiced_ratio": _round(_voiced_ratio(output_wav)),
        "threshold": MIN_PITCH_SHIFT_SEMITONE,
    }


def predict_stress(
    input_wav: str | Path,
    output_wav: str | Path,
    stress_words: list[str],
    transcript: str,
    *,
    language: str = "en",
) -> dict:
    """Evaluate stress editing via ASR timestamp target-window prominence gain."""
    del transcript

    from eval.metrics.asr_timestamp import asr_predict_with_timestamps

    if not stress_words:
        raise ValueError("stress_words is empty")

    input_asr = asr_predict_with_timestamps(input_wav, language)
    output_asr = asr_predict_with_timestamps(output_wav, language)

    word_details: list[dict] = []
    passed_count = 0
    for word in stress_words:
        input_window = _find_target_window(input_asr["tokens"], word, language)
        output_window = _find_target_window(output_asr["tokens"], word, language)
        detail = {
            "word": word,
            "input_found": input_window is not None,
            "output_found": output_window is not None,
        }
        if input_window is None or output_window is None:
            detail["passed"] = False
            detail["error"] = "target_not_found_in_timestamp_asr"
            word_details.append(detail)
            continue

        input_prom = _prominence(input_wav, input_window["start"], input_window["end"])
        output_prom = _prominence(output_wav, output_window["start"], output_window["end"])
        delta = output_prom["score"] - input_prom["score"]
        passed = delta > MIN_PROMINENCE_GAIN
        passed_count += int(passed)
        detail.update(
            {
                "passed": bool(passed),
                "input_window": input_window,
                "output_window": output_window,
                "input_prominence": _round(input_prom["score"]),
                "output_prominence": _round(output_prom["score"]),
                "prominence_delta": _round(delta),
                "input_energy_score": _round(input_prom["energy_score"]),
                "output_energy_score": _round(output_prom["energy_score"]),
                "input_f0_score": _round(input_prom["f0_score"]),
                "output_f0_score": _round(output_prom["f0_score"]),
                "input_duration_score": _round(input_prom["duration_score"]),
                "output_duration_score": _round(output_prom["duration_score"]),
            }
        )
        word_details.append(detail)

    rpg = passed_count / len(stress_words)
    found_all = all(item["input_found"] and item["output_found"] for item in word_details)
    return {
        "metric_type": "stress_timestamp_prominence_gain",
        "passed": bool(found_all and rpg == 1.0),
        "rpg": _round(rpg),
        "stress_words_total": len(stress_words),
        "stress_words_passed": passed_count,
        "target_found_all": bool(found_all),
        "input_timestamp_backend": input_asr["backend"],
        "output_timestamp_backend": output_asr["backend"],
        "input_timestamp_text": input_asr["text"],
        "output_timestamp_text": output_asr["text"],
        "input_timestamp_token_count": len(input_asr["tokens"]),
        "output_timestamp_token_count": len(output_asr["tokens"]),
        "word_details": word_details,
    }


def predict(
    input_wav: str | Path,
    output_wav: str | Path,
    anchor: dict,
    *,
    input_transcript: Optional[str] = None,
    output_transcript: Optional[str] = None,
    language: str = "en",
) -> dict:
    """Route to the metric required by anchor.prosody_type."""
    ptype = anchor.get("prosody_type")
    if ptype == "speed":
        return predict_speed(
            input_wav,
            output_wav,
            anchor["direction"],
            input_transcript=input_transcript,
            output_transcript=output_transcript,
            language=language,
        )
    if ptype == "pitch":
        return predict_pitch(
            input_wav,
            output_wav,
            anchor["direction"],
            language=language,
        )
    if ptype == "stress":
        return predict_stress(
            input_wav,
            output_wav,
            anchor.get("stress_words", []),
            anchor.get("transcript", input_transcript or ""),
            language=language,
        )
    raise ValueError(f"unknown prosody_type: {ptype!r}")
