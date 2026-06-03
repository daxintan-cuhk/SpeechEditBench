#!/usr/bin/env python3
"""Create a lightweight manifest for released SpeechEditBench data assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _task_manifest(task_dir: Path, repo_root: Path) -> dict[str, Any]:
    samples_path = task_dir / "samples.jsonl"
    audio_dir = task_dir / "audio"
    audio_files = sorted(
        path for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    return {
        "task_id": task_dir.name,
        "samples_jsonl": str(samples_path.relative_to(repo_root)),
        "samples": _count_jsonl(samples_path),
        "samples_sha256": _sha256(samples_path),
        "audio_files": len(audio_files),
        "audio_bytes": sum(path.stat().st_size for path in audio_files),
    }


def build_manifest(version: str, data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    tasks = [
        _task_manifest(task_dir, REPO_ROOT)
        for task_dir in sorted(data_root.iterdir())
        if task_dir.is_dir() and (task_dir / "samples.jsonl").is_file()
    ]
    return {
        "benchmark": "SpeechEditBench",
        "version": version,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head_at_manifest_creation": _git_head(),
        "asset_scope": [
            f"{data_root.relative_to(REPO_ROOT)}/*/samples.jsonl",
            f"{data_root.relative_to(REPO_ROOT)}/*/audio/**",
        ],
        "notes": [
            "This manifest records release data assets only.",
            "Caches, local model outputs, eval_results, and generated reports are not release data assets.",
        ],
        "totals": {
            "tasks": len(tasks),
            "samples": sum(task["samples"] for task in tasks),
            "audio_files": sum(task["audio_files"] for task in tasks),
            "audio_bytes": sum(task["audio_bytes"] for task in tasks),
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Dataset release version, e.g. v1.0")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Dataset root containing <task>/samples.jsonl and <task>/audio/.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.version, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
