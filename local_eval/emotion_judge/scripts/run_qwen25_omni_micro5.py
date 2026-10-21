from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]

SCRIPT_DIR = (
    REPO_ROOT
    / "local_eval/emotion_judge/scripts"
)

sys.path.insert(0, str(SCRIPT_DIR))

# 复用已通过 smoke test 的模型注册、加载补丁、提示词和 JSON 解析逻辑。
import qwen25_omni_single_sample_smoke as core  # noqa: E402


MANIFEST = (
    REPO_ROOT
    / "local_eval/emotion_judge/manifests/"
      "emotion_judge_micro5.jsonl"
)

RESULT_DIR = (
    REPO_ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/micro5"
)

RESULT_JSONL = RESULT_DIR / "predictions.jsonl"
SUMMARY_JSON = RESULT_DIR / "summary.json"


def resolve_audio_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def normalized_audio_sha256(
    audio_path: Path,
) -> tuple[str, dict]:
    waveform, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    mono = waveform.mean(axis=1).astype(np.float32)

    if sample_rate != 16000:
        mono = librosa.resample(
            mono,
            orig_sr=sample_rate,
            target_sr=16000,
            res_type="soxr_hq",
        ).astype(np.float32)

    peak = float(np.max(np.abs(mono)))

    if peak > 0.0:
        mono = mono / peak

    mono = np.clip(mono, -1.0, 1.0)

    pcm16 = np.round(
        mono * 32767.0
    ).astype("<i2")

    digest = hashlib.sha256(
        np.ascontiguousarray(pcm16).tobytes()
    ).hexdigest()

    metadata = {
        "original_sample_rate": sample_rate,
        "channels": waveform.shape[1],
        "original_frames": waveform.shape[0],
        "duration_sec": waveform.shape[0] / sample_rate,
        "normalized_sample_rate": 16000,
        "normalized_frames": len(mono),
        "peak_before_normalization": peak,
    }

    return digest, metadata


def load_manifest() -> list[dict]:
    rows = []

    with MANIFEST.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            row["_manifest_line"] = line_number
            rows.append(row)

    if len(rows) != 5:
        raise RuntimeError(
            f"micro-5 清单应包含 5 条，实际为 {len(rows)} 条"
        )

    return rows


def prepare_inputs(
    processor,
    row: dict,
    audio_path: Path,
):
    prompt = core.build_prompt(row)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a careful and deterministic "
                        "speech-emotion classification system."
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

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, images, videos = core.process_mm_info(
        messages,
        use_audio_in_video=False,
    )

    inputs = processor(
        text=chat_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    inputs = inputs.to("cuda")

    for key, value in inputs.items():
        if (
            torch.is_tensor(value)
            and torch.is_floating_point(value)
        ):
            inputs[key] = value.to(dtype=torch.float16)

    return inputs


def decode_generated_text(
    processor,
    inputs,
    output,
) -> tuple[str, dict]:
    if isinstance(output, tuple):
        output = output[0]

    if not torch.is_tensor(output) or output.ndim != 2:
        raise RuntimeError(
            "无法识别 generate() 返回值："
            f"type={type(output)}, "
            f"shape={getattr(output, 'shape', None)}"
        )

    prompt_length = inputs["input_ids"].shape[1]

    includes_prompt = (
        output.shape[1] >= prompt_length
        and torch.equal(
            output[:, :prompt_length].detach().cpu(),
            inputs["input_ids"].detach().cpu(),
        )
    )

    generated_ids = (
        output[:, prompt_length:]
        if includes_prompt
        else output
    )

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    metadata = {
        "prompt_tokens": prompt_length,
        "returned_tokens": output.shape[1],
        "generated_tokens": generated_ids.shape[1],
        "return_includes_prompt": includes_prompt,
    }

    return decoded[0].strip(), metadata


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    rows = load_manifest()

    print("=" * 100)
    print("Qwen2.5-Omni-7B-GPTQ-Int4 micro-5")
    print("=" * 100)
    print("manifest          :", MANIFEST)
    print("model             :", core.MODEL_DIR)
    print("model_revision    :", core.MODEL_REVISION)
    print("source_revision   :", core.SOURCE_REVISION)
    print("CUDA device       :", torch.cuda.get_device_name(0))
    print("样本数量          :", len(rows))

    print("\n===== 音频完整性预检查 =====")

    audio_metadata = {}

    for row in rows:
        audio_path = resolve_audio_path(row["audio_path"])

        actual_hash, metadata = normalized_audio_sha256(
            audio_path
        )

        expected_hash = row["audio_sha256"].lower()
        matched = actual_hash == expected_hash

        print(
            row["sample_id"],
            "|",
            row["language"],
            "/",
            row["ground_truth_emotion"],
            "| sr =",
            metadata["original_sample_rate"],
            "| hash_ok =",
            matched,
        )

        if not matched:
            raise RuntimeError(
                f"规范化波形哈希不匹配：{row['sample_id']}"
            )

        audio_metadata[row["sample_id"]] = metadata

    device_map = {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        "thinker.visual": "cpu",
        "thinker.audio_tower": "cpu",
        "talker": "cuda",
        "token2wav": "cuda",
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n===== 加载模型 =====")

    load_start = time.time()

    model = core.GPTQModel.load(
        str(core.MODEL_DIR),
        device_map=device_map,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    load_seconds = time.time() - load_start

    model.eval()

    processor = core.Qwen2_5OmniProcessor.from_pretrained(
        str(core.MODEL_DIR),
        local_files_only=True,
    )

    print("模型类型             :", type(model))
    print("模型加载耗时（秒）   :", round(load_seconds, 3))
    print(
        "加载峰值显存（GiB） :",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
    )
    print(
        "当前显存占用（GiB） :",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3,
        ),
    )

    results = []

    print("\n===== 连续推理 =====")

    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row["audio_path"])

        torch.cuda.empty_cache()

        inputs = prepare_inputs(
            processor,
            row,
            audio_path,
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        infer_start = time.time()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                thinker_do_sample=False,
                thinker_max_new_tokens=96,
            )

        torch.cuda.synchronize()
        infer_seconds = time.time() - infer_start

        raw_output, token_metadata = decode_generated_text(
            processor,
            inputs,
            output,
        )

        parsed = core.parse_json_object(raw_output)

        predicted = None
        confidence = None
        rationale = None

        if parsed is not None:
            predicted = str(
                parsed.get("predicted_emotion", "")
            ).strip().lower()

            confidence = parsed.get("confidence")
            rationale = parsed.get("rationale")

        allowed_labels = row["allowed_labels"]
        legal_label = predicted in allowed_labels
        correct = (
            predicted == row["ground_truth_emotion"]
            if legal_label
            else False
        )

        peak_gpu_gib = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        result = {
            "index": index,
            "sample_id": row["sample_id"],
            "manifest_line": row["_manifest_line"],
            "language": row["language"],
            "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"],
            "audio_metadata": audio_metadata[
                row["sample_id"]
            ],
            "transcript": row.get("transcript", ""),
            "allowed_labels": allowed_labels,
            "ground_truth_emotion": row[
                "ground_truth_emotion"
            ],
            "predicted_emotion": predicted,
            "confidence": confidence,
            "rationale": rationale,
            "legal_json": parsed is not None,
            "legal_label": legal_label,
            "correct": correct,
            "raw_output": raw_output,
            "token_metadata": token_metadata,
            "inference_seconds": round(
                infer_seconds,
                6,
            ),
            "peak_gpu_gib": round(
                peak_gpu_gib,
                6,
            ),
            "model_revision": core.MODEL_REVISION,
            "source_revision": core.SOURCE_REVISION,
            "attention_implementation": "sdpa",
            "thinker_do_sample": False,
            "thinker_max_new_tokens": 96,
        }

        results.append(result)

        print("\n" + "-" * 100)
        print(f"[{index}/5] {row['sample_id']}")
        print(
            "language / target :",
            row["language"],
            "/",
            row["ground_truth_emotion"],
        )
        print("prediction        :", predicted)
        print("confidence        :", confidence)
        print("legal_json        :", parsed is not None)
        print("legal_label       :", legal_label)
        print("correct           :", correct)
        print(
            "generated_tokens  :",
            token_metadata["generated_tokens"],
        )
        print(
            "inference_seconds :",
            round(infer_seconds, 3),
        )
        print(
            "peak_gpu_gib      :",
            round(peak_gpu_gib, 3),
        )
        print("raw_output        :", raw_output)

        del inputs
        del output
        torch.cuda.empty_cache()

    total = len(results)
    legal_json_count = sum(
        item["legal_json"] for item in results
    )
    legal_label_count = sum(
        item["legal_label"] for item in results
    )
    correct_count = sum(
        item["correct"] for item in results
    )

    by_language = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "legal_label": 0,
        }
    )

    for item in results:
        stats = by_language[item["language"]]
        stats["total"] += 1
        stats["correct"] += int(item["correct"])
        stats["legal_label"] += int(
            item["legal_label"]
        )

    language_summary = {}

    for language, stats in sorted(
        by_language.items()
    ):
        language_summary[language] = {
            **stats,
            "accuracy": (
                stats["correct"] / stats["total"]
            ),
            "legal_label_rate": (
                stats["legal_label"]
                / stats["total"]
            ),
        }

    summary = {
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "sample_count": total,
        "legal_json_count": legal_json_count,
        "legal_json_rate": legal_json_count / total,
        "legal_label_count": legal_label_count,
        "legal_label_rate": legal_label_count / total,
        "correct_count": correct_count,
        "accuracy": correct_count / total,
        "mean_inference_seconds": (
            sum(
                item["inference_seconds"]
                for item in results
            )
            / total
        ),
        "max_peak_gpu_gib": max(
            item["peak_gpu_gib"]
            for item in results
        ),
        "by_language": language_summary,
        "load_seconds": round(load_seconds, 6),
        "model_revision": core.MODEL_REVISION,
        "source_revision": core.SOURCE_REVISION,
        "attention_implementation": "sdpa",
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in results:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("micro-5 汇总")
    print("=" * 100)
    print("样本数量          :", total)
    print(
        "合法 JSON         :",
        f"{legal_json_count}/{total}",
    )
    print(
        "合法标签          :",
        f"{legal_label_count}/{total}",
    )
    print(
        "正确数量          :",
        f"{correct_count}/{total}",
    )
    print(
        "准确率            :",
        round(summary["accuracy"], 6),
    )
    print(
        "平均推理耗时（秒）:",
        round(
            summary["mean_inference_seconds"],
            3,
        ),
    )
    print(
        "最大峰值显存（GiB）:",
        round(summary["max_peak_gpu_gib"], 3),
    )

    for language, stats in language_summary.items():
        print(
            f"{language} 准确率"
            f"          : "
            f"{stats['correct']}/{stats['total']} "
            f"= {stats['accuracy']:.6f}"
        )

    print("预测结果          :", RESULT_JSONL)
    print("汇总结果          :", SUMMARY_JSON)


if __name__ == "__main__":
    main()
