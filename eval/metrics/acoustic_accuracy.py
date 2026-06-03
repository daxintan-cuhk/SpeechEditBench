"""
Acoustic-editing metrics.

The task has three different acoustic goals, so the evaluator keeps the
primary metric matched to each goal:

- enhancement: DNSMOS P.835 improvement plus content preservation, with
  optional PESQ/STOI diagnostics when a clean reference is available.
- env_transfer/reverb: a source->output transfer-response RT60 estimate checked
  against the sample's anchor.rt60_target_range.
- env_transfer/noise: PANNs CNN14 AudioSet tagging, grouped into the benchmark
  env_subtype labels outdoor/crowd/music.

PANNs source:
  qiuqiangkong/panns_inference, Cnn14_mAP=0.431.pth trained on AudioSet.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DNSMOS_OVRL_GAIN_THRESHOLD = 0.0
DNSMOS_BAK_GAIN_THRESHOLD = 0.0
EN_PRESERVATION_ERROR_THRESHOLD = 0.10
ZH_PRESERVATION_ERROR_THRESHOLD = 0.10
SCENE_PASS_THRESHOLD = 0.1
SCENE_GAIN_THRESHOLD = 0.03

RT60_TOLERANCE_LO = 0.8
RT60_TOLERANCE_HI = 1.2

PANN_CHECKPOINT = (
    REPO_ROOT / "eval_models" / "acoustic" / "panns-cnn14" / "Cnn14_mAP=0.431.pth"
)

SCENE_TO_AUDIOSET_LABELS: dict[str, list[str]] = {
    "outdoor": [
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Traffic noise, roadway noise",
        "Vehicle",
        "Motor vehicle (road)",
        "Wind",
        "Rain",
        "Bird",
        "Water",
    ],
    "crowd": [
        "Crowd",
        "Hubbub, speech noise, speech babble",
        "Cheering",
        "Applause",
        "Children shouting",
    ],
    "music": [
        "Music",
        "Background music",
        "Musical instrument",
        "Singing",
        "Pop music",
        "Electronic music",
    ],
}


def _load_audio_mono(path: str | Path, *, sr: int | None = None) -> tuple[np.ndarray, int]:
    import librosa

    audio, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"empty audio: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio, int(sample_rate)


def _align_pair(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(reference), len(degraded))
    if length <= 0:
        raise ValueError("empty audio pair")
    return reference[:length], degraded[:length]


def pesq_predict(audio_path: str | Path, ref_path: str | Path) -> float:
    """Compute wide-band PESQ at 16 kHz."""
    from pesq import pesq

    sr = 16000
    ref, _ = _load_audio_mono(ref_path, sr=sr)
    deg, _ = _load_audio_mono(audio_path, sr=sr)
    ref, deg = _align_pair(ref, deg)
    return float(pesq(sr, ref, deg, "wb"))


def stoi_predict(audio_path: str | Path, ref_path: str | Path) -> float:
    """Compute STOI at 16 kHz."""
    from pystoi import stoi

    sr = 16000
    ref, _ = _load_audio_mono(ref_path, sr=sr)
    deg, _ = _load_audio_mono(audio_path, sr=sr)
    ref, deg = _align_pair(ref, deg)
    return float(stoi(ref, deg, sr, extended=False))


def _rt60_from_ir(ir: np.ndarray, sr: int) -> float:
    peak = int(np.argmax(np.abs(ir)))
    tail = np.asarray(ir[peak:], dtype=np.float32)
    if tail.size < int(0.05 * sr):
        raise ValueError("impulse response too short for RT60 estimate")
    energy = np.cumsum(tail[::-1] ** 2)[::-1]
    if not np.any(energy > 0):
        raise ValueError("silent impulse response")
    decay_db = 10 * np.log10(np.maximum(energy, 1e-12) / np.max(energy))
    t = np.arange(len(decay_db), dtype=np.float32) / sr
    mask = (decay_db <= -5.0) & (decay_db >= -25.0)
    if np.count_nonzero(mask) < int(0.025 * sr):
        mask = (decay_db <= -3.0) & (decay_db >= -15.0)
    if np.count_nonzero(mask) < int(0.015 * sr):
        return 0.03
    slope, _ = np.polyfit(t[mask], decay_db[mask], deg=1)
    if slope >= -3.0:
        return 0.03
    return float(np.clip(-60.0 / slope, 0.03, 4.0))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _deconvolved_rt60(
    output_path: str | Path,
    source_path: str | Path,
    *,
    sr: int = 16000,
) -> float:
    source, _ = _load_audio_mono(source_path, sr=sr)
    output, _ = _load_audio_mono(output_path, sr=sr)
    source, output = _align_pair(source, output)
    if float(np.mean((source - output) ** 2)) <= 1e-8:
        return 0.03

    ir_len = int(1.6 * sr)
    n_fft = _next_power_of_two(len(source) + ir_len - 1)
    source_fft = np.fft.rfft(source, n=n_fft)
    output_fft = np.fft.rfft(output, n=n_fft)
    power = np.abs(source_fft) ** 2
    reg = max(float(np.max(power)) * 1e-3, 1e-8)
    transfer_fft = output_fft * np.conj(source_fft) / (power + reg)
    ir = np.fft.irfft(transfer_fft, n=n_fft)[:ir_len]
    peak = float(np.max(np.abs(ir)))
    if peak <= 1e-8:
        raise ValueError("unable to estimate source-output transfer response")
    ir = ir / peak
    return _rt60_from_ir(ir, sr)


def rt60_measure(audio_path: str | Path, source_path: str | Path | None = None) -> float:
    """
    Estimate RT60 in seconds for a generated speech waveform.

    True RT60 is defined on an impulse response. When the clean source is
    available, we estimate the source->output transfer response by regularized
    deconvolution and measure its energy decay. If source_path is omitted, this
    falls back to a blind speech-tail proxy.
    """
    import librosa

    if source_path is not None:
        return _deconvolved_rt60(audio_path, source_path)

    audio, sr = _load_audio_mono(audio_path, sr=16000)
    if len(audio) < int(0.25 * sr):
        raise ValueError("audio too short for RT60 estimate")

    frame_length = int(0.03 * sr)
    hop_length = int(0.01 * sr)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    if not np.any(rms > 0):
        raise ValueError("silent audio")

    db = 20 * np.log10(np.maximum(rms, 1e-8) / np.max(rms))
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)

    strong_threshold = max(float(np.percentile(db, 75)), -18.0)
    strong = np.flatnonzero(db >= strong_threshold)
    if strong.size == 0:
        strong = np.array([int(np.argmax(db))])

    candidates: list[float] = []
    min_gap_frames = max(1, int(0.08 / 0.01))
    last_peak = -min_gap_frames

    for peak in strong:
        if peak - last_peak < min_gap_frames:
            continue
        last_peak = int(peak)
        stop = min(len(db), peak + int(1.2 / 0.01))
        tail = db[peak:stop] - db[peak]
        tail_times = times[peak:stop] - times[peak]
        mask = (tail <= -5.0) & (tail >= -25.0)
        if np.count_nonzero(mask) < 6:
            continue
        slope, _ = np.polyfit(tail_times[mask], tail[mask], deg=1)
        if slope >= -3.0:
            continue
        rt60 = float(-60.0 / slope)
        if 0.03 <= rt60 <= 4.0:
            candidates.append(rt60)

    if candidates:
        return float(np.median(candidates))

    # Fallback: fit the final global energy-decay curve.
    energy = np.cumsum(audio[::-1] ** 2)[::-1]
    energy_db = 10 * np.log10(np.maximum(energy, 1e-12) / np.max(energy))
    t = np.arange(len(energy_db), dtype=np.float32) / sr
    mask = (energy_db <= -5.0) & (energy_db >= -25.0)
    if np.count_nonzero(mask) < sr * 0.05:
        raise ValueError("unable to find a stable decay region")
    slope, _ = np.polyfit(t[mask], energy_db[mask], deg=1)
    if slope >= -3.0:
        raise ValueError("unable to estimate a negative decay slope")
    return float(np.clip(-60.0 / slope, 0.03, 4.0))


@lru_cache(maxsize=2)
def _get_panns_tagger(device: str = "auto"):
    from panns_inference import AudioTagging

    resolved_device = device
    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    PANN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    return AudioTagging(checkpoint_path=str(PANN_CHECKPOINT), device=resolved_device)


def acoustic_scene_predict(audio_path: str | Path, *, device: str = "auto") -> dict[str, float]:
    """Return AudioSet clipwise scores from PANNs CNN14."""
    audio, _ = _load_audio_mono(audio_path, sr=32000)
    tagger = _get_panns_tagger(device)
    clipwise_output, _ = tagger.inference(audio[None, :])
    scores = np.asarray(clipwise_output[0], dtype=np.float32)
    return {
        label: float(scores[idx])
        for idx, label in enumerate(tagger.labels)
    }


def scene_group_scores(scene_scores: dict[str, float]) -> dict[str, float]:
    """Map AudioSet scores into the benchmark noise env_subtype groups."""
    grouped: dict[str, float] = {}
    for subtype, labels in SCENE_TO_AUDIOSET_LABELS.items():
        grouped[subtype] = max((scene_scores.get(label, 0.0) for label in labels), default=0.0)
    return grouped


def top_scene_labels(scene_scores: dict[str, float], *, k: int = 5) -> list[dict]:
    ordered = sorted(scene_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [{"label": label, "score": round(float(score), 4)} for label, score in ordered]


def _target_rt60_range(anchor: dict) -> tuple[float, float] | tuple[None, None]:
    value = anchor.get("rt60_target_range")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None, None


def _preservation_threshold(language: str) -> float:
    return ZH_PRESERVATION_ERROR_THRESHOLD if language == "zh" else EN_PRESERVATION_ERROR_THRESHOLD


def compute_acoustic_accuracy(
    sample: dict,
    *,
    dnsmos_sig: float | None = None,
    dnsmos_bak: float | None = None,
    dnsmos_ovrl: float | None = None,
    source_dnsmos_sig: float | None = None,
    source_dnsmos_bak: float | None = None,
    source_dnsmos_ovrl: float | None = None,
    transcript_target: str | None = None,
    transcript_predicted: str | None = None,
    norm_target: str | None = None,
    norm_predicted: str | None = None,
    wer: float | None = None,
    cer: float | None = None,
    content_preservation_error: float | None = None,
    pesq: float | None = None,
    stoi: float | None = None,
    source_pesq: float | None = None,
    source_stoi: float | None = None,
    utmos: float | None = None,
    rt60_measured: float | None = None,
    scene_scores: dict[str, float] | None = None,
    source_scene_scores: dict[str, float] | None = None,
) -> dict:
    """Build the per-sample acoustic metric row."""
    anchor = sample.get("anchor", {})
    subtask = anchor.get("subtask", "unknown")
    env_type = anchor.get("env_type")
    env_subtype = anchor.get("env_subtype")

    row: dict = {
        "sample_id": sample["sample_id"],
        "subtask": subtask,
        "language": sample.get("language", "unknown"),
        "source_dataset": sample.get("source_dataset", "unknown"),
        "degradation_type": anchor.get("degradation_type"),
        "env_type": env_type,
        "env_subtype": env_subtype,
        "passed": None,
    }

    if subtask == "enhancement":
        dnsmos_sig_gain = (
            dnsmos_sig - source_dnsmos_sig
            if dnsmos_sig is not None and source_dnsmos_sig is not None
            else None
        )
        dnsmos_bak_gain = (
            dnsmos_bak - source_dnsmos_bak
            if dnsmos_bak is not None and source_dnsmos_bak is not None
            else None
        )
        dnsmos_ovrl_gain = (
            dnsmos_ovrl - source_dnsmos_ovrl
            if dnsmos_ovrl is not None and source_dnsmos_ovrl is not None
            else None
        )
        pesq_gain = pesq - source_pesq if pesq is not None and source_pesq is not None else None
        stoi_gain = stoi - source_stoi if stoi is not None and source_stoi is not None else None
        preservation_threshold = _preservation_threshold(row["language"])
        target_passed = None
        if dnsmos_ovrl_gain is not None and dnsmos_bak_gain is not None:
            target_passed = (
                dnsmos_ovrl_gain > DNSMOS_OVRL_GAIN_THRESHOLD
                and dnsmos_bak_gain > DNSMOS_BAK_GAIN_THRESHOLD
            )
        row.update({
            "evaluation_protocol": "no_reference_perceptual",
            "dnsmos_sig": round(dnsmos_sig, 4) if dnsmos_sig is not None else None,
            "dnsmos_bak": round(dnsmos_bak, 4) if dnsmos_bak is not None else None,
            "dnsmos_ovrl": round(dnsmos_ovrl, 4) if dnsmos_ovrl is not None else None,
            "source_dnsmos_sig": (
                round(source_dnsmos_sig, 4) if source_dnsmos_sig is not None else None
            ),
            "source_dnsmos_bak": (
                round(source_dnsmos_bak, 4) if source_dnsmos_bak is not None else None
            ),
            "source_dnsmos_ovrl": (
                round(source_dnsmos_ovrl, 4) if source_dnsmos_ovrl is not None else None
            ),
            "dnsmos_sig_gain": round(dnsmos_sig_gain, 4) if dnsmos_sig_gain is not None else None,
            "dnsmos_bak_gain": round(dnsmos_bak_gain, 4) if dnsmos_bak_gain is not None else None,
            "dnsmos_ovrl_gain": (
                round(dnsmos_ovrl_gain, 4) if dnsmos_ovrl_gain is not None else None
            ),
            "transcript_target": transcript_target,
            "transcript_predicted": transcript_predicted,
            "norm_target": norm_target,
            "norm_predicted": norm_predicted,
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "content_preservation_metric": "cer" if row["language"] == "zh" else "wer",
            "content_preservation_error": (
                round(content_preservation_error, 4)
                if content_preservation_error is not None
                else None
            ),
            "dnsmos_ovrl_gain_threshold": DNSMOS_OVRL_GAIN_THRESHOLD,
            "dnsmos_bak_gain_threshold": DNSMOS_BAK_GAIN_THRESHOLD,
            "content_preservation_error_threshold": preservation_threshold,
            "pesq": round(pesq, 4) if pesq is not None else None,
            "stoi": round(stoi, 4) if stoi is not None else None,
            "source_pesq": round(source_pesq, 4) if source_pesq is not None else None,
            "source_stoi": round(source_stoi, 4) if source_stoi is not None else None,
            "pesq_gain": round(pesq_gain, 4) if pesq_gain is not None else None,
            "stoi_gain": round(stoi_gain, 4) if stoi_gain is not None else None,
            "utmos": round(utmos, 4) if utmos is not None else None,
            "passed": target_passed,
        })
        return row

    if subtask == "env_transfer" and env_type == "reverb":
        target_min, target_max = _target_rt60_range(anchor)
        tolerated_min = target_min * RT60_TOLERANCE_LO if target_min is not None else None
        tolerated_max = target_max * RT60_TOLERANCE_HI if target_max is not None else None
        passed = None
        if rt60_measured is not None and tolerated_min is not None and tolerated_max is not None:
            passed = tolerated_min <= rt60_measured <= tolerated_max
        row.update({
            "rt60_measured": round(rt60_measured, 4) if rt60_measured is not None else None,
            "rt60_target_min": target_min,
            "rt60_target_max": target_max,
            "rt60_tolerated_min": round(tolerated_min, 4) if tolerated_min is not None else None,
            "rt60_tolerated_max": round(tolerated_max, 4) if tolerated_max is not None else None,
            "rt60_reference": anchor.get("rt60_reference"),
            "passed": passed,
        })
        return row

    if subtask == "env_transfer" and env_type == "noise":
        if scene_scores is None or source_scene_scores is None:
            row.update({
                "scene_group_scores": None,
                "source_scene_group_scores": None,
                "predicted_env_subtype": None,
                "target_scene_score": None,
                "source_target_scene_score": None,
                "target_scene_gain": None,
                "scene_pass_threshold": SCENE_PASS_THRESHOLD,
                "scene_gain_threshold": SCENE_GAIN_THRESHOLD,
                "top_scene_labels": None,
                "passed": None,
            })
            return row

        grouped = scene_group_scores(scene_scores or {})
        source_grouped = scene_group_scores(source_scene_scores or {})
        predicted = max(grouped, key=grouped.get) if grouped else None
        target_score = grouped.get(str(env_subtype), 0.0)
        source_target_score = source_grouped.get(str(env_subtype), 0.0)
        target_gain = target_score - source_target_score
        passed = (
            predicted == env_subtype
            and target_score >= SCENE_PASS_THRESHOLD
            and target_gain >= SCENE_GAIN_THRESHOLD
            if predicted is not None
            else None
        )
        row.update({
            "scene_group_scores": {k: round(v, 4) for k, v in sorted(grouped.items())},
            "source_scene_group_scores": {
                k: round(v, 4) for k, v in sorted(source_grouped.items())
            },
            "predicted_env_subtype": predicted,
            "target_scene_score": round(target_score, 4),
            "source_target_scene_score": round(source_target_score, 4),
            "target_scene_gain": round(target_gain, 4),
            "scene_pass_threshold": SCENE_PASS_THRESHOLD,
            "scene_gain_threshold": SCENE_GAIN_THRESHOLD,
            "top_scene_labels": top_scene_labels(scene_scores or {}, k=5),
            "passed": passed,
        })
        return row

    return row
