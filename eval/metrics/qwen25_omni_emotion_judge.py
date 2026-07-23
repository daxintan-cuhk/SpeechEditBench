from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal


REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_DIR = (
    REPO_ROOT
    / "local_eval/emotion_judge/scripts"
)

SOURCE_LOCK_FILE = (
    REPO_ROOT
    / "eval/metrics/"
      "qwen25_omni_hybrid_v2_source_lock.json"
)

EXPECTED_SOURCE_LOCK_SHA256 = (
    "de3d4bd7423419206ac3ff3b761c9714"
    "e835282da2e80369c39c8ed9406c9f46"
)

JUDGE_VERSION = "qwen25_omni_hybrid_v2"

EXPECTED_MODEL_REVISION = (
    "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
    "@6d33b6bb5114a84de7efd38310779242520e7d4e"
)

EXPECTED_SOURCE_REVISION = (
    "QwenLM/Qwen2.5-Omni"
    "@d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42"
)

LABEL_CANONICALIZATION = {
    "surprised": "surprise",
}


# emotion_editing.py 可能通过 ThreadPoolExecutor
# 并发调用 predict()。模型加载与 generate() 必须串行。
_RUNTIME_LOCK = threading.RLock()

_RUNTIME: SimpleNamespace | None = None

# 进程内确定性缓存。缓存键包含规范化音频哈希、
# transcript、标签体系、语言和 sample_id。
_CACHE: dict[str, dict[str, Any]] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_frozen_source_lock() -> dict[str, Any]:
    """
    在加载模型前，校验生产源码锁及锁中的九个文件。

    任何 Prompt、GPTQModel 注册代码、低显存源码、
    Hybrid v2 规则或冻结记录发生改变，均直接终止。
    """
    if not SOURCE_LOCK_FILE.is_file():
        raise FileNotFoundError(
            f"缺少源码锁：{SOURCE_LOCK_FILE}"
        )

    actual_lock_hash = _sha256_file(
        SOURCE_LOCK_FILE
    )

    if (
        actual_lock_hash
        != EXPECTED_SOURCE_LOCK_SHA256
    ):
        raise RuntimeError(
            "源码锁 SHA256 不一致：\n"
            f"expected={EXPECTED_SOURCE_LOCK_SHA256}\n"
            f"actual={actual_lock_hash}"
        )

    lock = json.loads(
        SOURCE_LOCK_FILE.read_text(
            encoding="utf-8"
        )
    )

    expected_metadata = {
        "judge_version": JUDGE_VERSION,
        "status": "frozen",
        "model_revision": (
            EXPECTED_MODEL_REVISION
        ),
        "source_revision": (
            EXPECTED_SOURCE_REVISION
        ),
    }

    for key, expected_value in (
        expected_metadata.items()
    ):
        actual_value = lock.get(key)

        if actual_value != expected_value:
            raise RuntimeError(
                "源码锁元数据不一致：\n"
                f"field={key}\n"
                f"expected={expected_value!r}\n"
                f"actual={actual_value!r}"
            )

    entries = lock.get("files")

    if (
        not isinstance(entries, list)
        or len(entries) != 9
    ):
        raise RuntimeError(
            "源码锁应包含 9 个冻结文件"
        )

    for entry in entries:
        path = REPO_ROOT / entry["path"]

        if not path.is_file():
            raise FileNotFoundError(
                f"冻结文件不存在：{path}"
            )

        actual_hash = _sha256_file(path)
        expected_hash = entry["sha256"]

        if actual_hash != expected_hash:
            raise RuntimeError(
                "冻结源码已发生改变：\n"
                f"role={entry.get('role')}\n"
                f"path={path}\n"
                f"expected={expected_hash}\n"
                f"actual={actual_hash}"
            )

    return lock


def _load_runtime() -> SimpleNamespace:
    """
    延迟加载模型。

    导入 emotion_accuracy.py 时不会导入 torch，
    也不会加载本地 Qwen 模型。第一次调用本地
    predict() 时才进行加载。
    """
    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        verify_frozen_source_lock()

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen2.5-Omni 情感评测需要 CUDA"
            )

        script_dir = str(SCRIPT_DIR)

        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # 直接复用已经被 SHA256 锁定的冻结实现，
        # 避免人工复制 Prompt 时产生字符级漂移。
        micro = importlib.import_module(
            "run_qwen25_omni_micro5"
        )

        stage1 = importlib.import_module(
            "run_qwen25_omni_micro5_prompt_v2"
        )

        en_happy = importlib.import_module(
            "run_stage2_en_happy_vs_excited"
        )

        zh_happy = importlib.import_module(
            "run_stage2_zh_happy_vs_surprise_playfulness"
        )

        pairwise = importlib.import_module(
            "probe_en_neutral_pairwise_capability"
        )

        if (
            micro.core.MODEL_REVISION
            != EXPECTED_MODEL_REVISION
        ):
            raise RuntimeError(
                "模型 revision 与冻结配置不一致"
            )

        if (
            micro.core.SOURCE_REVISION
            != EXPECTED_SOURCE_REVISION
        ):
            raise RuntimeError(
                "源码 revision 与冻结配置不一致"
            )

        free_bytes, total_bytes = (
            torch.cuda.mem_get_info(0)
        )

        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3

        minimum_free_gib = float(
            os.getenv(
                "SPEECHEDITBENCH_QWEN_MIN_FREE_GIB",
                "17",
            )
        )

        if free_gib < minimum_free_gib:
            raise RuntimeError(
                "当前可见 GPU 空闲显存不足：\n"
                f"device={torch.cuda.get_device_name(0)}\n"
                f"free={free_gib:.3f} GiB\n"
                f"total={total_gib:.3f} GiB\n"
                f"required={minimum_free_gib:.3f} GiB\n"
                "请通过 CUDA_VISIBLE_DEVICES "
                "选择空闲 GPU。"
            )

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.empty_cache()

        device_map = {
            "thinker.model": "cuda",
            "thinker.lm_head": "cuda",
            "thinker.visual": "cpu",
            "thinker.audio_tower": "cpu",
            "talker": "cuda",
            "token2wav": "cuda",
        }

        print(
            f"[emotion-judge] loading "
            f"{JUDGE_VERSION}; "
            f"CUDA_VISIBLE_DEVICES="
            f"{os.getenv('CUDA_VISIBLE_DEVICES')!r}",
            flush=True,
        )

        model = micro.core.GPTQModel.load(
            str(micro.core.MODEL_DIR),
            device_map=device_map,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        model.eval()

        processor = (
            micro.core.Qwen2_5OmniProcessor
            .from_pretrained(
                str(micro.core.MODEL_DIR),
                local_files_only=True,
            )
        )

        _RUNTIME = SimpleNamespace(
            torch=torch,
            micro=micro,
            stage1=stage1,
            en_happy=en_happy,
            zh_happy=zh_happy,
            pairwise=pairwise,
            model=model,
            processor=processor,
        )

        print(
            "[emotion-judge] model loaded; "
            f"allocated="
            f"{torch.cuda.memory_allocated() / 1024**3:.3f} GiB",
            flush=True,
        )

        return _RUNTIME


def _prepare_inputs(
    runtime: SimpleNamespace,
    audio_path: Path,
    prompt: str,
):
    """
    复现冻结实验的多模态输入构造，但不修改
    micro.core.build_prompt 全局变量。
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a careful and "
                        "deterministic speech-emotion "
                        "classification system."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": str(audio_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    chat_text = (
        runtime.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    audios, images, videos = (
        runtime.micro.core.process_mm_info(
            messages,
            use_audio_in_video=False,
        )
    )

    inputs = runtime.processor(
        text=chat_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    ).to("cuda")

    for key, value in inputs.items():
        if (
            runtime.torch.is_tensor(value)
            and runtime.torch.is_floating_point(
                value
            )
        ):
            inputs[key] = value.to(
                dtype=runtime.torch.float16
            )

    return inputs


def _infer(
    runtime: SimpleNamespace,
    audio_path: Path,
    prompt: str,
    allowed_labels: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = _prepare_inputs(
        runtime,
        audio_path,
        prompt,
    )

    runtime.torch.cuda.empty_cache()
    runtime.torch.cuda.synchronize()
    runtime.torch.cuda.reset_peak_memory_stats()

    with runtime.torch.inference_mode():
        output = runtime.model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            thinker_do_sample=False,
            thinker_max_new_tokens=max_new_tokens,
        )

    runtime.torch.cuda.synchronize()

    raw_output, token_metadata = (
        runtime.micro.decode_generated_text(
            runtime.processor,
            inputs,
            output,
        )
    )

    parsed = (
        runtime.micro.core.parse_json_object(
            raw_output
        )
    )

    prediction = None

    if parsed is not None:
        prediction = str(
            parsed.get(
                "predicted_emotion",
                "",
            )
        ).strip().lower()

    result = {
        "prediction": prediction,
        "legal_json": parsed is not None,
        "legal_label": (
            prediction in allowed_labels
        ),
        "raw_output": raw_output,
        "token_metadata": token_metadata,
    }

    del inputs
    del output

    runtime.torch.cuda.empty_cache()

    return result


def _cache_key(
    audio_hash: str,
    language: str,
    transcript: str,
    allowed_labels: list[str],
    sample_id: str,
) -> str:
    payload = {
        "judge": JUDGE_VERSION,
        "source_lock": (
            EXPECTED_SOURCE_LOCK_SHA256
        ),
        "audio": audio_hash,
        "language": language,
        "transcript": transcript,
        "allowed_labels": allowed_labels,
        # pairwise 标签顺序取决于 sample_id，
        # 因而 sample_id 必须进入缓存键。
        "sample_id": sample_id,
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def predict_with_details(
    audio_path: Path,
    *,
    language: Literal["zh", "en"],
    transcript: str,
    allowed_labels: list[str],
    sample_id: str = "",
) -> dict[str, Any]:
    """
    运行冻结的 Qwen2.5-Omni Hybrid v2 judge。

    该接口不读取 target_emotion，也不读取 instruction，
    始终进行 blind emotion classification。
    """
    audio_path = Path(audio_path).resolve()
    language = str(language).strip().lower()
    transcript = str(transcript or "")
    sample_id = str(sample_id or "")

    allowed = [
        str(label).strip().lower()
        for label in allowed_labels
    ]

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    if language not in {"en", "zh"}:
        raise ValueError(
            f"不支持的语言：{language!r}"
        )

    if (
        not allowed
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError(
            "allowed_labels 为空或包含重复标签"
        )

    runtime = _load_runtime()

    audio_hash, audio_metadata = (
        runtime.micro.normalized_audio_sha256(
            audio_path
        )
    )

    key = _cache_key(
        audio_hash,
        language,
        transcript,
        allowed,
        sample_id,
    )

    # ThreadPoolExecutor 可以并发进入本函数，
    # 但模型加载、缓存读写及所有 generate 均串行执行。
    with _RUNTIME_LOCK:
        cached = _CACHE.get(key)

        if cached is not None:
            return dict(cached)

        row = {
            "sample_id": sample_id,
            "language": language,
            "transcript": transcript,
            "allowed_labels": allowed,
        }

        stage1_call = _infer(
            runtime,
            audio_path,
            runtime.stage1.build_prompt_v2(row),
            allowed,
            96,
        )

        raw_stage1 = stage1_call[
            "prediction"
        ]

        stage1_prediction = (
            LABEL_CANONICALIZATION.get(
                raw_stage1,
                raw_stage1,
            )
        )

        if stage1_prediction not in allowed:
            result = {
                "judge_version": JUDGE_VERSION,
                "source_lock_sha256": (
                    EXPECTED_SOURCE_LOCK_SHA256
                ),
                "sample_id": sample_id,
                "language": language,
                "normalized_audio_sha256": (
                    audio_hash
                ),
                "audio_metadata": audio_metadata,
                "stage1_prediction": (
                    stage1_prediction
                ),
                "applied_rule": None,
                "final_prediction": "unknown",
                "final_legal_label": False,
                "stage_calls": {
                    "stage1": stage1_call,
                },
            }

            _CACHE[key] = result
            return dict(result)

        final_prediction = stage1_prediction
        applied_rule = None

        calls: dict[str, Any] = {
            "stage1": stage1_call,
        }

        # -------------------------------------------------
        # 规则 1：EN happy → happy / excited
        # -------------------------------------------------
        if (
            language == "en"
            and stage1_prediction == "happy"
        ):
            candidates = [
                "happy",
                "excited",
            ]

            call = _infer(
                runtime,
                audio_path,
                runtime.en_happy
                .build_stage2_prompt(row),
                candidates,
                48,
            )

            if call["legal_label"]:
                final_prediction = call[
                    "prediction"
                ]

            applied_rule = (
                "en_happy_vs_excited_v1"
            )

            calls["en_happy"] = call

        # -------------------------------------------------
        # 规则 2：ZH happy
        # → happy / surprise / playfulness
        # -------------------------------------------------
        elif (
            language == "zh"
            and stage1_prediction == "happy"
        ):
            candidates = [
                "happy",
                "surprise",
                "playfulness",
            ]

            call = _infer(
                runtime,
                audio_path,
                runtime.zh_happy
                .build_stage2_prompt(row),
                candidates,
                48,
            )

            if call["legal_label"]:
                final_prediction = call[
                    "prediction"
                ]

            applied_rule = (
                "zh_happy_vs_surprise_"
                "playfulness_v1"
            )

            calls["zh_happy"] = call

        # -------------------------------------------------
        # 规则 3：EN neutral 双 pairwise 检测器
        # -------------------------------------------------
        elif (
            language == "en"
            and stage1_prediction == "neutral"
        ):
            detector_calls = {}

            for pair_name in [
                "neutral_vs_fearful",
                "neutral_vs_sad",
            ]:
                candidates = list(
                    runtime.pairwise
                    .PAIR_CONFIGS[pair_name][
                        "labels"
                    ]
                )

                displayed_labels = (
                    runtime.pairwise
                    .counterbalanced_label_order(
                        pair_name,
                        sample_id,
                        candidates,
                    )
                )

                pair_row = {
                    **row,
                    "_pair_name": pair_name,
                    "_pair_labels": candidates,
                    "_displayed_labels": (
                        displayed_labels
                    ),
                    "allowed_labels": candidates,
                }

                detector_calls[pair_name] = (
                    _infer(
                        runtime,
                        audio_path,
                        runtime.pairwise
                        .build_pairwise_prompt(
                            pair_row
                        ),
                        candidates,
                        32,
                    )
                )

            fearful_call = detector_calls[
                "neutral_vs_fearful"
            ]

            sad_call = detector_calls[
                "neutral_vs_sad"
            ]

            fearful_fire = (
                fearful_call["legal_label"]
                and fearful_call["prediction"]
                == "fearful"
            )

            sad_fire = (
                sad_call["legal_label"]
                and sad_call["prediction"]
                == "sad"
            )

            if fearful_fire and not sad_fire:
                final_prediction = "fearful"
                decision = "fearful_only"

            elif sad_fire and not fearful_fire:
                final_prediction = "sad"
                decision = "sad_only"

            elif fearful_fire and sad_fire:
                final_prediction = "neutral"
                decision = (
                    "conflict_fallback_neutral"
                )

            else:
                final_prediction = "neutral"
                decision = (
                    "no_detector_fired"
                )

            applied_rule = (
                "en_neutral_dual_pairwise_v1"
            )

            calls.update(detector_calls)

            calls["pairwise_fusion"] = {
                "fearful_detector_fired": (
                    fearful_fire
                ),
                "sad_detector_fired": (
                    sad_fire
                ),
                "decision": decision,
            }

        result = {
            "judge_version": JUDGE_VERSION,
            "source_lock_sha256": (
                EXPECTED_SOURCE_LOCK_SHA256
            ),
            "model_revision": (
                EXPECTED_MODEL_REVISION
            ),
            "source_revision": (
                EXPECTED_SOURCE_REVISION
            ),
            "sample_id": sample_id,
            "language": language,
            "normalized_audio_sha256": (
                audio_hash
            ),
            "audio_metadata": audio_metadata,
            "stage1_prediction": (
                stage1_prediction
            ),
            "applied_rule": applied_rule,
            "final_prediction": (
                final_prediction
            ),
            "final_legal_label": (
                final_prediction in allowed
            ),
            "prediction_changed": (
                final_prediction
                != stage1_prediction
            ),
            "stage_calls": calls,
        }

        _CACHE[key] = result
        return dict(result)


def predict_emotion(
    audio_path: Path,
    *,
    language: Literal["zh", "en"],
    transcript: str,
    allowed_labels: list[str],
    sample_id: str = "",
) -> str:
    result = predict_with_details(
        audio_path,
        language=language,
        transcript=transcript,
        allowed_labels=allowed_labels,
        sample_id=sample_id,
    )

    return str(result["final_prediction"])


def get_backend_status() -> dict[str, Any]:
    """
    查看后端状态，但不触发模型加载。
    """
    with _RUNTIME_LOCK:
        return {
            "judge_version": JUDGE_VERSION,
            "source_lock_sha256": (
                EXPECTED_SOURCE_LOCK_SHA256
            ),
            "runtime_loaded": (
                _RUNTIME is not None
            ),
            "memory_cache_entries": (
                len(_CACHE)
            ),
            "cuda_visible_devices": (
                os.getenv(
                    "CUDA_VISIBLE_DEVICES"
                )
            ),
        }


def clear_memory_cache() -> None:
    with _RUNTIME_LOCK:
        _CACHE.clear()
