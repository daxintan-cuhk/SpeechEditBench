"""
UTMOS naturalness prediction for speech-editing evaluation.

This wrapper follows the official UTMOS22 quick-prediction path:

    sarulab-speech/UTMOS22 -> Hugging Face UTMOS-demo -> score.Score

The demo has legacy dependencies, so we run it in the dedicated local venv under
eval_models/mos/utmos22_env and keep the benchmark's main Python environment
unchanged.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UTMOS_DEMO_DIR = REPO_ROOT / "eval_models" / "mos" / "UTMOS-demo"
UTMOS_ENV_PYTHON = REPO_ROOT / "eval_models" / "mos" / "utmos22_env" / "bin" / "python"
UTMOS_CHECKPOINT = UTMOS_DEMO_DIR / "epoch=3-step=7459.ckpt"


def _require_utmos_files() -> None:
    missing = [
        str(path)
        for path in [
            UTMOS_DEMO_DIR / "score.py",
            UTMOS_DEMO_DIR / "lightning_module.py",
            UTMOS_CHECKPOINT,
            UTMOS_ENV_PYTHON,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "UTMOS22 evaluator is incomplete. Missing: " + ", ".join(missing)
        )


def predict_many(audio_paths: list[str | Path], *, device: str = "auto") -> list[float]:
    """Predict UTMOS scores for a list of audio files."""
    _require_utmos_files()
    paths = [str(Path(path).resolve()) for path in audio_paths]
    missing_audio = [path for path in paths if not Path(path).exists()]
    if missing_audio:
        raise FileNotFoundError("Audio file not found: " + ", ".join(missing_audio))
    if not paths:
        return []

    helper_code = f"""
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(UTMOS_DEMO_DIR)!r})

import numpy as np
import soundfile as sf
import torch
from score import Score

paths = json.load(sys.stdin)
requested_device = {device!r}
device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
if device == "auto":
    device = "cpu"

scores = []
scorers = {{}}
with contextlib.redirect_stdout(sys.stderr):
    for path in paths:
        wav, sr = sf.read(Path(path), always_2d=True)
        sr = int(sr)
        wav = wav.astype(np.float32, copy=False).T
        wav = torch.from_numpy(wav)
        if sr not in scorers:
            scorers[sr] = Score(
                ckpt_path={str(UTMOS_CHECKPOINT)!r},
                input_sample_rate=sr,
                device=device,
            )
        score = scorers[sr].score(wav.to(device))
        scores.append(float(score[0]))

sys.stdout.write(json.dumps(scores))
"""

    completed = subprocess.run(
        [str(UTMOS_ENV_PYTHON), "-c", helper_code],
        input=json.dumps(paths),
        text=True,
        cwd=str(UTMOS_DEMO_DIR),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "UTMOS22 subprocess failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    try:
        return [float(score) for score in json.loads(completed.stdout)]
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "UTMOS22 subprocess returned non-JSON output.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        ) from exc


def predict(audio_path: str | Path, *, device: str = "auto") -> float:
    """Predict the UTMOS naturalness score for one audio file."""
    return predict_many([audio_path], device=device)[0]
