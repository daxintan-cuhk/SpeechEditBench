from __future__ import annotations

import time
from pathlib import Path

from eval.metrics.content_accuracy import asr_predict, compute_cer, compute_wer, normalize
from eval.metrics.speaker_similarity import batch_predict as speaker_batch_predict


REPO_ROOT = Path(__file__).resolve().parents[2]

EN_CONTENT_PRESERVATION_THRESHOLD = 0.10
ZH_CONTENT_PRESERVATION_THRESHOLD = 0.10
SPEAKER_TARGET_THRESHOLD = 0.50
SPEAKER_PRESERVATION_THRESHOLD = 0.75
DEFAULT_PROGRESS_EVERY = 25
OUTPUT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")


def progress_log(task_id: str, message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{task_id}] {message}", flush=True)


def should_log_progress(index: int, total: int, every: int = DEFAULT_PROGRESS_EVERY) -> bool:
    return index == 1 or index % every == 0 or index == total


def resolve_sample_path(path_value: str, samples_jsonl_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    by_samples_dir = samples_jsonl_path.parent / path
    if by_samples_dir.exists():
        return by_samples_dir

    return REPO_ROOT / path


def find_output_audio(output_dir: Path, sample_id: str) -> Path | None:
    """Find a model output by sample id in root or audio/ subdirectory."""
    search_dirs = [output_dir]
    audio_dir = output_dir / "audio"
    if audio_dir.is_dir():
        search_dirs.append(audio_dir)

    for directory in search_dirs:
        for ext in OUTPUT_AUDIO_EXTENSIONS:
            candidate = directory / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def content_preservation_threshold(language: str) -> float:
    return ZH_CONTENT_PRESERVATION_THRESHOLD if language == "zh" else EN_CONTENT_PRESERVATION_THRESHOLD


def mean_or_none(values: list[float | None], *, digits: int = 4) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return round(sum(valid) / len(valid), digits) if valid else None


def rate_or_none(values: list[bool | None], *, digits: int = 4) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(sum(1 for value in valid if value) / len(valid), digits)


def content_preservation_metrics(
    output_wav: str | Path,
    transcript_target: str,
    language: str,
) -> dict:
    transcript_predicted = asr_predict(Path(output_wav), language)
    norm_target = normalize(transcript_target, language)
    norm_predicted = normalize(transcript_predicted, language)
    wer = round(compute_wer(norm_target, norm_predicted), 4)
    cer = round(compute_cer(norm_target, norm_predicted), 4)
    metric = "cer" if language == "zh" else "wer"
    primary_error = cer if language == "zh" else wer
    threshold = content_preservation_threshold(language)
    return {
        "transcript_target": transcript_target,
        "transcript_predicted": transcript_predicted,
        "norm_target": norm_target,
        "norm_predicted": norm_predicted,
        "wer": wer,
        "cer": cer,
        "content_preservation_metric": metric,
        "content_preservation_error": round(primary_error, 4),
        "content_preservation_threshold": threshold,
        "content_preservation_pass": primary_error <= threshold,
    }


def batch_source_output_speaker_similarity(
    pairs: list[dict],
    *,
    device: str = "auto",
) -> dict[str, float | None]:
    if not pairs:
        return {}

    scores = speaker_batch_predict(pairs, device=device)
    return {
        pair["sample_id"]: (round(score, 4) if score is not None else None)
        for pair, score in zip(pairs, scores, strict=True)
    }


def batch_utmos_scores(
    pairs: list[tuple[str, str | Path]],
    *,
    device: str = "auto",
    batch_size: int | None = None,
) -> dict[str, float | None]:
    if not pairs:
        return {}

    from eval.metrics.utmos import predict_many as utmos_predict_many

    results: dict[str, float | None] = {}
    step = len(pairs) if batch_size is None else max(1, batch_size)
    for start in range(0, len(pairs), step):
        batch = pairs[start:start + step]
        sample_ids = [sample_id for sample_id, _ in batch]
        audio_paths = [audio_path for _, audio_path in batch]
        scores = utmos_predict_many(audio_paths, device=device)
        results.update({
            sample_id: round(score, 4) if score is not None else None
            for sample_id, score in zip(sample_ids, scores, strict=True)
        })
    return results
