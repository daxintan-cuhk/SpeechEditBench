#!/usr/bin/env python3
"""Audit SpeechEditBench samples/audio before dataset repair releases."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}
SHORT_THRESHOLDS = (0.8, 1.0, 1.5, 2.0, 3.0)

PATTERNS = {
    "position_or_scope": re.compile(
        r"\b(after|before|between|pause|pauses|boundary|boundaries|throughout|"
        r"at the beginning|at the end|natural pauses|position|word|phrase)\b|"
        r"在|之前|之后|开头|结尾|停顿|位置|全程|贯穿",
        re.I,
    ),
    "intensity_or_extra_manner": re.compile(
        r"\b(intense|strong|subtle|slightly|clearly|heightened|overwhelming|"
        r"warm|authoritative|dramatic|casual|urgent|nervous|unease|dread)\b|"
        r"紧张|颤抖|强烈|轻微|明显|急促|压抑|氛围|温暖|权威|自然",
        re.I,
    ),
    "content_preservation_text": re.compile(
        r"保留|保持|content unchanged|content and speaking style remain unchanged|"
        r"preserve|语义|内容",
        re.I,
    ),
    "meta_instruction": re.compile(
        r"语音编辑系统|speech editing system|重录|重新录制|请你|you need to",
        re.I,
    ),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_path(path_value: str | None, samples_file: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    for base in (REPO_ROOT, samples_file.parent, samples_file.parent.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def _duration_seconds(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if not info.samplerate:
            return None
        return float(info.frames / info.samplerate)
    except Exception:  # noqa: BLE001
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[idx]


def _quote_issue(text: str) -> bool:
    return (
        text.count('"') % 2 == 1
        or text.count("“") != text.count("”")
        or text.count("‘") != text.count("’")
    )


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): v for k, v in counter.items()}


def _task_audit(task_dir: Path) -> dict[str, Any]:
    samples_file = task_dir / "samples.jsonl"
    rows = _load_jsonl(samples_file)

    sample_ids = Counter(row.get("sample_id") for row in rows)
    audio_paths = Counter(row.get("audio_path") for row in rows)
    durations: list[float] = []
    missing_audio: list[str] = []
    missing_duration: list[str] = []

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        audio_path = _resolve_path(row.get("audio_path"), samples_file)
        if audio_path is None or not audio_path.exists():
            missing_audio.append(sample_id)
            continue
        duration = _duration_seconds(audio_path)
        if duration is None:
            missing_duration.append(sample_id)
            continue
        durations.append(duration)

    instructions = [str(row.get("instruction") or "") for row in rows]
    transcripts = [str(row.get("transcript") or "") for row in rows]
    anchors = [row.get("anchor") or {} for row in rows]

    pattern_counts = {
        name: sum(1 for text in instructions if pattern.search(text))
        for name, pattern in PATTERNS.items()
    }

    duration_summary = {
        "min": round(min(durations), 4) if durations else None,
        "p05": round(_percentile(durations, 0.05), 4) if durations else None,
        "median": round(statistics.median(durations), 4) if durations else None,
        "p95": round(_percentile(durations, 0.95), 4) if durations else None,
        "max": round(max(durations), 4) if durations else None,
        "short_counts": {
            f"lt_{str(threshold).replace('.', '_')}s": sum(1 for value in durations if value < threshold)
            for threshold in SHORT_THRESHOLDS
        },
    }

    label_counters: dict[str, dict[str, int]] = {}
    for key in sorted({k for anchor in anchors for k in anchor}):
        if key in {
            "source_emotion", "target_emotion", "source_style", "target_style",
            "operation", "event", "prosody_type", "direction", "subtask",
            "env_type", "env_subtype", "degradation_type", "edit_type",
        }:
            label_counters[key] = _counter_dict(Counter(anchor.get(key) for anchor in anchors))

    return {
        "task_id": task_dir.name,
        "samples": len(rows),
        "languages": _counter_dict(Counter(row.get("language") for row in rows)),
        "source_datasets": _counter_dict(Counter(row.get("source_dataset") for row in rows)),
        "empty_instruction_count": sum(1 for text in instructions if not text.strip()),
        "empty_transcript_count": sum(1 for text in transcripts if not text.strip()),
        "duplicate_sample_id_count": sum(1 for value in sample_ids.values() if value > 1),
        "duplicate_audio_path_count": sum(1 for value in audio_paths.values() if value > 1),
        "missing_audio_count": len(missing_audio),
        "missing_duration_count": len(missing_duration),
        "duration_seconds": duration_summary,
        "instruction_unique_count": len(set(instructions)),
        "instruction_length": {
            "min": min((len(text) for text in instructions), default=0),
            "median": statistics.median([len(text) for text in instructions]) if instructions else 0,
            "max": max((len(text) for text in instructions), default=0),
        },
        "instruction_pattern_counts": pattern_counts,
        "instruction_unbalanced_quote_count": sum(1 for text in instructions if _quote_issue(text)),
        "transcript_unbalanced_quote_count": sum(1 for text in transcripts if _quote_issue(text)),
        "missing_anchor_count": sum(1 for row in rows if row.get("anchor") is None),
        "anchor_key_sets": {
            "|".join(keys): count
            for keys, count in Counter(tuple(sorted(anchor.keys())) for anchor in anchors).items()
        },
        "label_counters": label_counters,
        "top_instruction_templates": _top_instruction_templates(instructions),
    }


def _top_instruction_templates(instructions: list[str], limit: int = 8) -> list[dict[str, Any]]:
    normalized = []
    for text in instructions:
        item = re.sub(r'"[^"]+"', '"<q>"', text)
        item = re.sub(
            r"\b(angry|happy|sad|neutral|fear|fearful|surprise|playfulness|"
            r"excited|frustrated|dramatic|storytelling|conversational|intimate|"
            r"public[- ]broadcast|restrained[- ]flat|breath|laugh|cough|sigh)\b",
            "<label>",
            item,
            flags=re.I,
        )
        normalized.append(item)
    return [
        {"template": template, "count": count}
        for template, count in Counter(normalized).most_common(limit)
    ]


def build_audit(data_root: Path, version: str) -> dict[str, Any]:
    task_dirs = [
        path for path in sorted(data_root.iterdir())
        if path.is_dir() and (path / "samples.jsonl").is_file()
    ]
    tasks = [_task_audit(task_dir) for task_dir in task_dirs]
    return {
        "benchmark": "SpeechEditBench",
        "version": version,
        "data_root": str(data_root.relative_to(REPO_ROOT)),
        "totals": {
            "tasks": len(tasks),
            "samples": sum(task["samples"] for task in tasks),
            "missing_audio_count": sum(task["missing_audio_count"] for task in tasks),
            "empty_instruction_count": sum(task["empty_instruction_count"] for task in tasks),
            "duplicate_sample_id_count": sum(task["duplicate_sample_id_count"] for task in tasks),
        },
        "tasks": tasks,
    }


def write_markdown(audit: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# SpeechEditBench {audit['version']} Dataset Audit",
        "",
        f"- Data root: `{audit['data_root']}`",
        f"- Tasks: {audit['totals']['tasks']}",
        f"- Samples: {audit['totals']['samples']}",
        f"- Missing audio: {audit['totals']['missing_audio_count']}",
        f"- Empty instructions: {audit['totals']['empty_instruction_count']}",
        "",
        "| Task | Samples | Langs | Duration median | <2s | Empty instr | Position/scope | Extra manner | Unbalanced transcript quotes |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in audit["tasks"]:
        short_2s = task["duration_seconds"]["short_counts"]["lt_2_0s"]
        langs = ", ".join(f"{k}:{v}" for k, v in sorted(task["languages"].items()))
        lines.append(
            f"| `{task['task_id']}` | {task['samples']} | {langs} | "
            f"{task['duration_seconds']['median']} | {short_2s} | "
            f"{task['empty_instruction_count']} | "
            f"{task['instruction_pattern_counts']['position_or_scope']} | "
            f"{task['instruction_pattern_counts']['intensity_or_extra_manner']} | "
            f"{task['transcript_unbalanced_quote_count']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Position/scope` and `Extra manner` are heuristic pattern counts for triage, not hard errors.")
    lines.append("- Duration is measured from local audio files with `soundfile`.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--version", default="v1.1")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    audit = build_audit(args.data_root.resolve(), args.version)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_json)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(audit, args.output_md)
        print(args.output_md)


if __name__ == "__main__":
    main()
