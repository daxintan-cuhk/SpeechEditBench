"""
DNSMOS P.835 prediction for no-reference enhancement evaluation.

This wrapper follows the official Microsoft DNS Challenge local inference path:
  - primary MOS model: DNSMOS/sig_bak_ovr.onnx
  - P.808 model: DNSMOS/model_v8.onnx

Model files are downloaded on demand from the official repository if missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DNSMOS_DIR = REPO_ROOT / "eval_models" / "mos" / "DNSMOS"
DNSMOS_PRIMARY_MODEL = DNSMOS_DIR / "sig_bak_ovr.onnx"
DNSMOS_P808_MODEL = DNSMOS_DIR / "model_v8.onnx"

DNSMOS_PRIMARY_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
    "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_P808_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
    "DNSMOS/DNSMOS/model_v8.onnx"
)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def _download_file(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp_path.replace(dest)


def _ensure_models() -> None:
    if not DNSMOS_PRIMARY_MODEL.exists():
        _download_file(DNSMOS_PRIMARY_URL, DNSMOS_PRIMARY_MODEL)
    if not DNSMOS_P808_MODEL.exists():
        _download_file(DNSMOS_P808_URL, DNSMOS_P808_MODEL)


def _audio_melspec(
    audio,
    *,
    n_mels: int = 120,
    frame_size: int = 320,
    hop_length: int = 160,
    sr: int = SAMPLING_RATE,
):
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=frame_size + 1,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40.0) / 40.0
    return mel_spec.T


def _polyfit_scores(sig_raw: float, bak_raw: float, ovr_raw: float) -> tuple[float, float, float]:
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return float(p_sig(sig_raw)), float(p_bak(bak_raw)), float(p_ovr(ovr_raw))


@lru_cache(maxsize=1)
def _get_sessions():
    import onnxruntime as ort

    _ensure_models()
    primary_session = ort.InferenceSession(str(DNSMOS_PRIMARY_MODEL))
    p808_session = ort.InferenceSession(str(DNSMOS_P808_MODEL))
    return primary_session, p808_session


def _load_audio(audio_path: str | Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, input_fs = sf.read(str(audio_path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if input_fs != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=input_fs, target_sr=SAMPLING_RATE)
    if audio.size == 0:
        raise ValueError(f"empty audio: {audio_path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def predict(audio_path: str | Path) -> dict[str, float]:
    primary_session, p808_session = _get_sessions()
    audio = _load_audio(audio_path)

    len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)

    num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
    hop_len_samples = SAMPLING_RATE

    sig_scores: list[float] = []
    bak_scores: list[float] = []
    ovr_scores: list[float] = []
    p808_scores: list[float] = []

    for idx in range(num_hops):
        start = int(idx * hop_len_samples)
        end = int((idx + INPUT_LENGTH) * hop_len_samples)
        audio_seg = audio[start:end]
        if len(audio_seg) < len_samples:
            continue

        input_features = np.asarray(audio_seg, dtype=np.float32)[np.newaxis, :]
        p808_input = np.asarray(
            _audio_melspec(audio=audio_seg[:-160]),
            dtype=np.float32,
        )[np.newaxis, :, :]

        p808_mos = float(p808_session.run(None, {"input_1": p808_input})[0][0][0])
        sig_raw, bak_raw, ovr_raw = primary_session.run(None, {"input_1": input_features})[0][0]
        sig, bak, ovr = _polyfit_scores(sig_raw, bak_raw, ovr_raw)

        sig_scores.append(sig)
        bak_scores.append(bak)
        ovr_scores.append(ovr)
        p808_scores.append(p808_mos)

    if not sig_scores:
        raise ValueError(f"unable to compute DNSMOS segments for {audio_path}")

    return {
        "SIG": round(float(np.mean(sig_scores)), 4),
        "BAK": round(float(np.mean(bak_scores)), 4),
        "OVRL": round(float(np.mean(ovr_scores)), 4),
        "P808_MOS": round(float(np.mean(p808_scores)), 4),
    }


def predict_many(audio_paths: list[str | Path]) -> list[dict[str, float]]:
    return [predict(path) for path in audio_paths]
