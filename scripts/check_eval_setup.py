#!/usr/bin/env python3
"""Check local SpeechEditBench data, packages, and evaluator model readiness."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = REPO_ROOT / "release_manifests" / "v1.1" / "dataset_manifest.json"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
PATH_KEYS = {
    "audio_path",
    "reference_audio_path",
    "target_reference_path",
    "reference_wav",
}

PACKAGE_SPECS = {
    "huggingface_hub": ("huggingface-hub", ">=1.12,<1.13"),
    "requests": ("requests", ">=2.33,<2.34"),
    "numpy": ("numpy", ">=2.4,<2.5"),
    "soundfile": ("soundfile", ">=0.13,<0.14"),
    "packaging": ("packaging", ">=26,<27"),
    "librosa": ("librosa", ">=0.11,<0.12"),
    "torch": ("torch", ">=2.11,<2.12"),
    "transformers": ("transformers", ">=5.5,<5.6"),
    "funasr": ("funasr", ">=1.3,<1.4"),
    "modelscope": ("modelscope", ">=1.36,<1.37"),
    "onnxruntime": ("onnxruntime", ">=1.25,<1.26"),
    "panns-inference": ("panns-inference", "==0.1.1"),
    "pesq": ("pesq", "==0.0.4"),
    "pystoi": ("pystoi", ">=0.4,<0.5"),
}

MODEL_GROUPS = {
    "asr_en_whisper": {
        "description": "English ASR and word timestamps",
        "paths": [
            "eval_models/asr/whisper-large-v3/config.json",
            "eval_models/asr/whisper-large-v3/generation_config.json",
            "eval_models/asr/whisper-large-v3/preprocessor_config.json",
            "eval_models/asr/whisper-large-v3/model.safetensors.index.fp32.json",
            "eval_models/asr/whisper-large-v3/model.fp32-00001-of-00002.safetensors",
            "eval_models/asr/whisper-large-v3/model.fp32-00002-of-00002.safetensors",
            "eval_models/asr/whisper-large-v3/added_tokens.json",
            "eval_models/asr/whisper-large-v3/merges.txt",
            "eval_models/asr/whisper-large-v3/normalizer.json",
            "eval_models/asr/whisper-large-v3/special_tokens_map.json",
            "eval_models/asr/whisper-large-v3/tokenizer_config.json",
            "eval_models/asr/whisper-large-v3/vocab.json",
        ],
    },
    "asr_zh_paraformer": {
        "description": "Chinese ASR",
        "paths": [
            "eval_models/asr/paraformer-zh/config.yaml",
            "eval_models/asr/paraformer-zh/configuration.json",
            "eval_models/asr/paraformer-zh/am.mvn",
            "eval_models/asr/paraformer-zh/model.pt",
            "eval_models/asr/paraformer-zh/seg_dict",
            "eval_models/asr/paraformer-zh/tokens.json",
        ],
    },
    "asr_zh_timestamp": {
        "description": "Chinese stress timestamps",
        "paths": [
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/config.yaml",
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/configuration.json",
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/am.mvn",
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/model.pt",
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/seg_dict",
            "eval_models/asr/paraformer-zh-vad-punc-timestamp/tokens.json",
        ],
    },
    "speaker_wavlm": {
        "description": "WavLM speaker verification",
        "paths": [
            "eval_models/speaker/wavlm-large-pretrained/config.json",
            "eval_models/speaker/wavlm-large-pretrained/preprocessor_config.json",
            "eval_models/speaker/wavlm-large-pretrained/pytorch_model.bin",
            "eval_models/speaker/wavlm-large-sv/wavlm-large.pt",
            "eval_models/speaker/wavlm-large-sv/ecapa_tdnn.py",
        ],
    },
    "utmos": {
        "description": "UTMOS naturalness diagnostics",
        "paths": [
            "eval_models/mos/UTMOS-demo/score.py",
            "eval_models/mos/UTMOS-demo/lightning_module.py",
            "eval_models/mos/UTMOS-demo/epoch=3-step=7459.ckpt",
            "eval_models/mos/utmos22_env/bin/python",
        ],
    },
    "panns": {
        "description": "PANNs acoustic scene classifier",
        "paths": [
            "eval_models/acoustic/panns-cnn14/Cnn14_mAP=0.431.pth",
        ],
    },
    "dnsmos": {
        "description": "DNSMOS ONNX models, downloaded on demand if missing",
        "paths": [
            "eval_models/mos/DNSMOS/sig_bak_ovr.onnx",
            "eval_models/mos/DNSMOS/model_v8.onnx",
        ],
        "auto_download": True,
    },
}

TASK_REQUIREMENTS = {
    "content_editing": ["asr_en_whisper", "asr_zh_paraformer"],
    "speaker_editing": ["speaker_wavlm", "asr_en_whisper", "asr_zh_paraformer"],
    "emotion_editing": ["gemini", "asr_en_whisper", "asr_zh_paraformer"],
    "style_editing": ["gemini", "asr_en_whisper", "asr_zh_paraformer"],
    "paralinguistic_editing": ["gemini", "asr_en_whisper", "asr_zh_paraformer"],
    "prosody_editing": [
        "asr_en_whisper",
        "asr_zh_paraformer",
        "asr_zh_timestamp",
    ],
    "acoustic_editing": [
        "asr_en_whisper",
        "asr_zh_paraformer",
        "dnsmos",
        "panns",
    ],
    "compositional_editing": [
        "gemini",
        "speaker_wavlm",
        "asr_en_whisper",
        "asr_zh_paraformer",
        "asr_zh_timestamp",
        "dnsmos",
        "panns",
    ],
}


def _package_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def _version_satisfies(version: str | None, spec: str) -> bool:
    if version is None:
        return False
    try:
        return Version(version) in SpecifierSet(spec)
    except InvalidVersion:
        return False


def _count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _manifest_tasks() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {task["task_id"]: task for task in payload.get("tasks", [])}


def _audio_count(task_dir: Path) -> int:
    audio_dir = task_dir / "audio"
    if not audio_dir.exists():
        return 0
    return sum(
        1
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


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


def _declared_audio_paths(samples: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for sample in samples:
        paths.update(_collect_data_audio_paths(sample))
    return paths


def _sample_languages(samples: list[dict[str, Any]]) -> set[str]:
    languages: set[str] = set()
    for sample in samples:
        language = sample.get("language")
        if isinstance(language, str) and language:
            languages.add(language)
        for component in sample.get("components") or []:
            if not isinstance(component, dict):
                continue
            component_language = component.get("source_language")
            if isinstance(component_language, str) and component_language:
                languages.add(component_language)
    return languages or {"en"}


def _add_content_asr_requirements(required: set[str], languages: set[str]) -> None:
    if "en" in languages or not languages:
        required.add("asr_en_whisper")
    if "zh" in languages:
        required.add("asr_zh_paraformer")
    if languages - {"en", "zh"}:
        required.add("asr_en_whisper")


def _needs_panns(anchor: dict[str, Any]) -> bool:
    if anchor.get("subtask") != "env_transfer":
        return False
    return anchor.get("env_type") == "noise" or anchor.get("env_subtype") in {
        "outdoor",
        "crowd",
        "music",
    }


def _needs_zh_timestamp(anchor: dict[str, Any], languages: set[str]) -> bool:
    return anchor.get("prosody_type") == "stress" and "zh" in languages


def _component_type(component: dict[str, Any]) -> str:
    value = component.get("component_type") or component.get("edit_type") or ""
    return str(value).replace("_editing", "")


def required_model_groups_for_samples(task_id: str, samples: list[dict[str, Any]]) -> list[str]:
    """Infer blocking evaluator model groups for the selected samples."""
    required: set[str] = set()
    languages = _sample_languages(samples)
    _add_content_asr_requirements(required, languages)

    llm_multimodal_tasks = {
        "emotion_editing",
        "style_editing",
        "paralinguistic_editing",
    }
    if task_id in llm_multimodal_tasks:
        required.add("gemini")
    if task_id == "speaker_editing":
        required.add("speaker_wavlm")

    if task_id == "prosody_editing":
        for sample in samples:
            anchor = sample.get("anchor") or {}
            if isinstance(anchor, dict) and _needs_zh_timestamp(anchor, languages):
                required.add("asr_zh_timestamp")

    if task_id == "acoustic_editing":
        for sample in samples:
            anchor = sample.get("anchor") or {}
            if not isinstance(anchor, dict):
                continue
            if anchor.get("subtask") == "enhancement":
                required.add("dnsmos")
            if _needs_panns(anchor):
                required.add("panns")

    if task_id == "compositional_editing":
        for sample in samples:
            for component in sample.get("components") or []:
                if not isinstance(component, dict):
                    continue
                ctype = _component_type(component)
                anchor = component.get("anchor") or {}
                if not isinstance(anchor, dict):
                    anchor = {}
                if ctype == "speaker":
                    required.add("speaker_wavlm")
                elif ctype in {"emotion", "style", "paralinguistic"}:
                    required.add("gemini")
                elif ctype == "prosody" and _needs_zh_timestamp(anchor, languages):
                    required.add("asr_zh_timestamp")
                elif ctype == "acoustic":
                    if anchor.get("subtask") == "enhancement":
                        required.add("dnsmos")
                    if _needs_panns(anchor):
                        required.add("panns")

    return sorted(required)


def required_model_groups_for_task(task_id: str, samples_file: Path | None = None) -> list[str]:
    samples_path = samples_file or DATA_ROOT / task_id / "samples.jsonl"
    samples = _load_jsonl(samples_path)
    if samples:
        return required_model_groups_for_samples(task_id, samples)
    return sorted(TASK_REQUIREMENTS.get(task_id, []))


def check_packages() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for display_name, (dist_name, spec) in PACKAGE_SPECS.items():
        version = _package_version(dist_name)
        packages[display_name] = {
            "ok": _version_satisfies(version, spec),
            "version": version,
            "required": spec,
        }
    return packages


def check_data() -> dict[str, Any]:
    manifest_tasks = _manifest_tasks()
    tasks: dict[str, Any] = {}
    for task_id, manifest in sorted(manifest_tasks.items()):
        task_dir = DATA_ROOT / task_id
        samples_file = task_dir / "samples.jsonl"
        sample_count = _count_jsonl(samples_file)
        samples = _load_jsonl(samples_file)
        declared_audio_paths = _declared_audio_paths(samples)
        missing_audio_paths = sorted(
            path for path in declared_audio_paths if not (REPO_ROOT / path).exists()
        )
        declared_audio_count = len(declared_audio_paths)
        present_audio_count = declared_audio_count - len(missing_audio_paths)
        expected_audio_count = manifest.get("audio_files")
        audio_manifest_ok = declared_audio_count == expected_audio_count
        local_audio_count = _audio_count(task_dir)
        tasks[task_id] = {
            "samples_file": str(samples_file.relative_to(REPO_ROOT)),
            "samples_ok": sample_count == manifest.get("samples"),
            "samples": sample_count,
            "expected_samples": manifest.get("samples"),
            "audio_files": present_audio_count,
            "expected_audio_files": expected_audio_count,
            "declared_audio_files": declared_audio_count,
            "audio_manifest_ok": audio_manifest_ok,
            "audio_complete": audio_manifest_ok and not missing_audio_paths,
            "missing_audio_count": len(missing_audio_paths),
            "missing_audio_files": missing_audio_paths[:20],
            "extra_audio_files": max(0, local_audio_count - present_audio_count),
        }
    return {
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)) if MANIFEST_PATH.exists() else None,
        "tasks": tasks,
    }


def check_model_groups() -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group_name, spec in MODEL_GROUPS.items():
        missing = [
            path
            for path in spec["paths"]
            if not (REPO_ROOT / path).exists()
        ]
        auto_download = bool(spec.get("auto_download"))
        groups[group_name] = {
            "ok": not missing or auto_download,
            "local_files_ok": not missing,
            "description": spec["description"],
            "missing": missing,
            "auto_download": auto_download,
        }
    groups["gemini"] = {
        "ok": bool(os.getenv("GEMINI_API_KEY")),
        "description": "Gemini-compatible multimodal judge",
        "missing": [] if os.getenv("GEMINI_API_KEY") else ["GEMINI_API_KEY"],
        "auto_download": False,
    }
    return groups


def check_task_readiness(model_groups: dict[str, Any]) -> dict[str, Any]:
    readiness: dict[str, Any] = {}
    for task_id in TASK_REQUIREMENTS:
        requirements = required_model_groups_for_task(task_id)
        missing = [
            requirement
            for requirement in requirements
            if not model_groups.get(requirement, {}).get("ok")
        ]
        readiness[task_id] = {
            "ok": not missing,
            "required_groups": requirements,
            "missing_groups": missing,
        }
    return readiness


def check_hf_access(repo_id: str, revision: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, list_repo_files
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"huggingface_hub import failed: {exc}"}

    try:
        info = HfApi().dataset_info(repo_id)
        files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
        return {
            "ok": True,
            "repo_id": getattr(info, "id", repo_id),
            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "sha": getattr(info, "sha", None),
            "revision": revision,
            "file_count": len(files),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "repo_id": repo_id, "revision": revision, "error": str(exc)}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packages = check_packages()
    data = check_data()
    model_groups = check_model_groups()
    report = {
        "python": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "packages": packages,
        "data": data,
        "model_groups": model_groups,
        "tasks": check_task_readiness(model_groups),
    }
    if args.check_hf:
        report["huggingface"] = check_hf_access(args.hf_repo, args.revision)
    return report


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def print_text_report(report: dict[str, Any]) -> None:
    print("SpeechEditBench setup check")
    print(f"Python    : {report['python']}")
    print(f"Repo root : {report['repo_root']}")

    print("\nPackages")
    for name, row in report["packages"].items():
        version = row["version"] or "-"
        required = row.get("required", "")
        print(f"  {_status(row['ok']):7s} {name:20s} {version:12s} required={required}")

    print("\nData")
    for task_id, row in report["data"]["tasks"].items():
        audio = f"{row['audio_files']}/{row['expected_audio_files']}"
        samples = f"{row['samples']}/{row['expected_samples']}"
        samples_status = _status(row["samples_ok"])
        audio_status = _status(row["audio_complete"])
        extra = row.get("extra_audio_files", 0)
        extra_note = f" extra={extra}" if extra else ""
        print(
            f"  {task_id:24s} "
            f"samples={samples:11s}({samples_status:7s}) "
            f"audio={audio:11s}({audio_status:7s}){extra_note}"
        )

    print("\nEvaluator model groups")
    for name, row in report["model_groups"].items():
        suffix = " (auto-download)" if row.get("auto_download") else ""
        print(f"  {_status(row['ok']):7s} {name:18s} {row['description']}{suffix}")
        for missing in row.get("missing", []):
            label = "will download" if row.get("auto_download") else "missing"
            print(f"           {label}: {missing}")

    print("\nTask readiness")
    for task_id, row in report["tasks"].items():
        missing = ", ".join(row["missing_groups"]) or "-"
        print(f"  {_status(row['ok']):7s} {task_id:24s} missing={missing}")

    if "huggingface" in report:
        hf = report["huggingface"]
        if hf.get("ok"):
            print(
                "\nHugging Face\n"
                f"  OK      {hf['repo_id']} revision={hf['revision']} "
                f"private={hf['private']} gated={hf['gated']} files={hf['file_count']}"
            )
        else:
            print(f"\nHugging Face\n  MISSING {hf.get('error')}")


def _has_strict_failure(report: dict[str, Any]) -> bool:
    if any(not row["ok"] for row in report["packages"].values()):
        return True
    if any(not row["samples_ok"] for row in report["data"]["tasks"].values()):
        return True
    if any(not row["audio_complete"] for row in report["data"]["tasks"].values()):
        return True
    if any(not row["ok"] for row in report["model_groups"].values()):
        return True
    if "huggingface" in report and not report["huggingface"].get("ok"):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on missing packages, samples, or evaluator assets")
    parser.add_argument("--check-hf", action="store_true", help="Check anonymous Hugging Face dataset access")
    parser.add_argument("--hf-repo", default="DiscreteSpeech/SpeechEditBench")
    parser.add_argument("--revision", default="v1.1")
    args = parser.parse_args()

    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)

    if args.strict and _has_strict_failure(report):
        sys.exit(1)


if __name__ == "__main__":
    main()
