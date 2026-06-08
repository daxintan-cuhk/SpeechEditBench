#!/usr/bin/env python3
"""Download SpeechEditBench data assets from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
PATH_KEYS = {
    "audio_path",
    "reference_audio_path",
    "target_reference_path",
    "reference_wav",
}


def default_repo_root() -> Path:
    """Parent of ``scripts/`` when this file lives in ``scripts/download_hf_dataset.py``."""
    return Path(__file__).resolve().parents[1]


def _is_data_audio_path(value: str) -> bool:
    return value.startswith("data/") and value.lower().endswith(AUDIO_SUFFIXES)


def _collect_data_audio_paths(value: object, *, key: str | None = None) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            paths.update(_collect_data_audio_paths(child_value, key=str(child_key)))
        return paths
    if isinstance(value, list):
        for child in value:
            paths.update(_collect_data_audio_paths(child))
        return paths
    if isinstance(value, str) and (key in PATH_KEYS or _is_data_audio_path(value)):
        if _is_data_audio_path(value):
            paths.add(value)
    return paths


def benchmark_allow_patterns(repo_root: Path) -> list[str]:
    """Return sample metadata plus the audio files referenced by those samples."""
    patterns: set[str] = set()
    for samples_file in sorted((repo_root / "data").glob("*/samples.jsonl")):
        patterns.add(samples_file.relative_to(repo_root).as_posix())
        with samples_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                patterns.update(_collect_data_audio_paths(row))
    return sorted(patterns) or ["data/**"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="DiscreteSpeech/SpeechEditBench",
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit sha) for reproducible snapshots.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Download destination root (creates/updates data/ underneath). "
        "Default: this repository's root (directory that contains scripts/). "
        "Use e.g. '.' to use the current working directory instead.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token; defaults to HF_TOKEN or cached `huggingface-cli login`.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download the full HF snapshot, including repository files such as README.md.",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Download every file under data/**, including extra files not referenced "
        "by the benchmark samples. By default only benchmark-referenced data files "
        "are downloaded.",
    )
    args = parser.parse_args()

    repo_root = default_repo_root()
    target = (args.local_dir or default_repo_root()).resolve()
    target.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    patterns: list[str] | None = None
    if args.all_files:
        if args.all_data:
            print("--all-files was set; ignoring --all-data.")
    elif args.all_data:
        patterns = ["data/**"]
    else:
        patterns = benchmark_allow_patterns(repo_root)

    if patterns is not None:
        kwargs["allow_patterns"] = patterns
        print(f"Downloading {len(patterns)} file pattern(s) from {args.repo_id!r}.")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(target),
        token=args.token,
        **kwargs,
    )
    print(f"Downloaded {args.repo_id!r} into {target}")


if __name__ == "__main__":
    main()
