"""
Speaker Similarity (SS) for speaker_editing.

This evaluator follows the WavLM large speaker verification setup referenced by
Microsoft UniSpeech downstream speaker_verification:

    WavLM-large SSL hidden states -> ECAPA-TDNN speaker head -> cosine similarity

The downloaded speaker-verification checkpoint is expected at:

    eval_models/speaker/wavlm-large-sv/wavlm-large.pt

The WavLM large SSL model is loaded from:

    eval_models/speaker/wavlm-large-pretrained
"""

from __future__ import annotations

import importlib.util
import gc
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKER_MODEL_DIR = REPO_ROOT / "eval_models" / "speaker" / "wavlm-large-sv"
WAVLM_MODEL_DIR = REPO_ROOT / "eval_models" / "speaker" / "wavlm-large-pretrained"
SPEAKER_CHECKPOINT = SPEAKER_MODEL_DIR / "wavlm-large.pt"
ECAPA_CODE = SPEAKER_MODEL_DIR / "ecapa_tdnn.py"


def _require_files() -> None:
    missing = [
        str(path)
        for path in [SPEAKER_CHECKPOINT, ECAPA_CODE, WAVLM_MODEL_DIR / "config.json"]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "WavLM speaker verification evaluator is incomplete. Missing: "
            + ", ".join(missing)
        )


def _load_ecapa_module() -> Any:
    spec = importlib.util.spec_from_file_location("wavlm_sv_ecapa_tdnn", ECAPA_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ECAPA code from {ECAPA_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested for speaker similarity but is unavailable; using CPU.")
        return "cpu"
    return device


def _cleanup_device(device: str) -> None:
    import torch

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


@lru_cache(maxsize=8)
def _load_models(device: str) -> tuple[Any, Any]:
    _require_files()

    from transformers.models.wavlm.configuration_wavlm import WavLMConfig
    from transformers.models.wavlm.modeling_wavlm import WavLMModel
    import torch
    import torch.nn as nn

    resolved_device = torch.device(_resolve_device(device))
    wavlm_config = WavLMConfig.from_pretrained(
        WAVLM_MODEL_DIR,
        local_files_only=True,
        output_hidden_states=True,
    )

    wavlm = WavLMModel.from_pretrained(
        WAVLM_MODEL_DIR,
        config=wavlm_config,
        local_files_only=True,
    ).to(resolved_device)
    wavlm.eval()

    ecapa_module = _load_ecapa_module()
    head = ecapa_module.ECAPA_TDNN(
        feat_dim=1024,
        channels=512,
        emb_dim=256,
        feat_type="fbank",
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
    )
    # The UniSpeech checkpoint learns a weighted sum over 25 WavLM hidden states.
    head.feature_weight = nn.Parameter(torch.zeros(25))

    checkpoint = torch.load(SPEAKER_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith("feature_extract.")
    }
    head.load_state_dict(state_dict, strict=False)
    head.to(resolved_device)
    head.eval()

    return wavlm, head


def _load_audio(audio_path: str | Path) -> Any:
    import librosa

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    return wav


def _embed(audio_path: str | Path, *, device: str = "auto") -> Any:
    wavlm, head = _load_models(device)

    import torch

    torch_device = next(head.parameters()).device
    wav = torch.from_numpy(_load_audio(audio_path)).float().unsqueeze(0).to(torch_device)

    with torch.inference_mode():
        ssl_outputs = wavlm(wav, output_hidden_states=True)
        hidden_states = torch.stack(tuple(ssl_outputs.hidden_states), dim=0)
        weights = torch.softmax(head.feature_weight, dim=-1).view(-1, 1, 1, 1)
        feats = (weights * hidden_states).sum(dim=0).transpose(1, 2) + 1e-6

        x = head.instance_norm(feats)
        out1 = head.layer1(x)
        out2 = head.layer2(out1)
        out3 = head.layer3(out2)
        out4 = head.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = torch.relu(head.conv(out))
        emb = head.linear(head.bn(head.pooling(out))).squeeze(0).detach().cpu()

    del wav
    del ssl_outputs
    del hidden_states
    del feats
    del x
    del out1
    del out2
    del out3
    del out4
    del out
    _cleanup_device(str(torch_device))
    return emb


def predict(
    output_wav: str | Path,
    reference_wav: str | Path,
    *,
    model_name: str = "wavlm_large",
    device: str = "auto",
) -> float:
    """
    Compute speaker cosine similarity between output_wav and target reference_wav.

    Args:
        output_wav: model output after speaker editing
        reference_wav: target-speaker reference audio
        model_name: kept for API compatibility; only "wavlm_large" is supported
        device: kept for API compatibility; model auto-selects CUDA when available
    """
    if model_name != "wavlm_large":
        raise ValueError("Only model_name='wavlm_large' is supported by this evaluator.")

    emb_out = _embed(output_wav, device=device)
    emb_ref = _embed(reference_wav, device=device)

    import torch.nn.functional as F

    return float(F.cosine_similarity(emb_out.unsqueeze(0), emb_ref.unsqueeze(0)).item())


def batch_predict(
    pairs: list[dict],
    *,
    model_name: str = "wavlm_large",
    device: str = "auto",
) -> list[float | None]:
    """Batch helper for {"output_wav": ..., "reference_wav": ...} pairs."""
    if model_name != "wavlm_large":
        raise ValueError("Only model_name='wavlm_large' is supported by this evaluator.")

    path_order: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        for key in ("output_wav", "reference_wav"):
            path = str(Path(pair[key]).resolve())
            if path not in seen:
                seen.add(path)
                path_order.append(path)

    embeddings: dict[str, Any | None] = {}
    total_paths = len(path_order)
    for index, path in enumerate(path_order, start=1):
        try:
            embeddings[path] = _embed(path, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] speaker embedding failed for {path}: {e}", flush=True)
            embeddings[path] = None
        if index == 1 or index % 25 == 0 or index == total_paths:
            print(f"[PROGRESS] speaker embedding {index}/{total_paths}", flush=True)

    import torch.nn.functional as F

    results: list[float | None] = []
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        output_path = str(Path(pair["output_wav"]).resolve())
        reference_path = str(Path(pair["reference_wav"]).resolve())
        emb_out = embeddings.get(output_path)
        emb_ref = embeddings.get(reference_path)
        if emb_out is None or emb_ref is None:
            results.append(None)
        else:
            results.append(float(F.cosine_similarity(emb_out.unsqueeze(0), emb_ref.unsqueeze(0)).item()))
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[PROGRESS] speaker similarity {index}/{total}", flush=True)
    return results
