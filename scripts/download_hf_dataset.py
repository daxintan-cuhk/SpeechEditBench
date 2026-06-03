#!/usr/bin/env python3
"""Download SpeechEditBench data assets from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def default_repo_root() -> Path:
    """Parent of ``scripts/`` when this file lives in ``scripts/download_hf_dataset.py``."""
    return Path(__file__).resolve().parents[1]


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
        help="Download the full HF snapshot. By default only data/** is downloaded "
        "so repository files such as README.md are not overwritten.",
    )
    args = parser.parse_args()

    target = (args.local_dir or default_repo_root()).resolve()
    target.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if not args.all_files:
        kwargs["allow_patterns"] = ["data/**"]

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
