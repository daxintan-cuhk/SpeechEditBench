from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from gptqmodel import GPTQModel
from gptqmodel.models._const import SUPPORTED_MODELS
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseGPTQModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file


# ---------------------------------------------------------------------
# 固定路径
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = (
    REPO_ROOT
    / "eval_models/emotion/Qwen2.5-Omni-7B-GPTQ-Int4"
)

LOW_VRAM_DIR = (
    REPO_ROOT
    / "local_eval/emotion_judge/third_party/"
      "Qwen2.5-Omni/low-VRAM-mode"
)

MANIFEST = (
    REPO_ROOT
    / "local_eval/emotion_judge/manifests/"
      "emotion_judge_dev.jsonl"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/smoke/single_sample.json"
)

MODEL_REVISION = (
    "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
    "@6d33b6bb5114a84de7efd38310779242520e7d4e"
)

SOURCE_REVISION = (
    "QwenLM/Qwen2.5-Omni"
    "@d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42"
)

if not MODEL_DIR.exists():
    raise FileNotFoundError(MODEL_DIR)

if not LOW_VRAM_DIR.exists():
    raise FileNotFoundError(LOW_VRAM_DIR)

sys.path.insert(0, str(LOW_VRAM_DIR))

from modeling_qwen2_5_omni_low_VRAM_mode import (  # noqa: E402
    Qwen2_5OmniForConditionalGeneration,
)


# ---------------------------------------------------------------------
# 注册官方 GPTQModel 自定义类型
# ---------------------------------------------------------------------
class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration

    base_modules = [
        "thinker.model.embed_tokens",
        "thinker.model.norm",
        "token2wav",
        "thinker.audio_tower",
        "thinker.model.rotary_emb",
        "thinker.visual",
        "talker",
    ]

    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"

    layer_modules = [
        [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
        ],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def preprocess_dataset(self, sample: dict) -> dict:
        return sample


MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ

if "qwen2_5_omni" not in SUPPORTED_MODELS:
    SUPPORTED_MODELS.append("qwen2_5_omni")


# ---------------------------------------------------------------------
# 修补 from_config：强制从本地加载 spk_dict.pt
# ---------------------------------------------------------------------
@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    del args
    kwargs.pop("trust_remote_code", None)

    model = cls._from_config(config, **kwargs)

    speaker_path = cached_file(
        str(MODEL_DIR),
        "spk_dict.pt",
        local_files_only=True,
    )

    if speaker_path is None:
        raise FileNotFoundError(
            MODEL_DIR / "spk_dict.pt"
        )

    model.load_speakers(speaker_path)
    return model


Qwen2_5OmniForConditionalGeneration.from_config = (
    patched_from_config
)


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def load_first_manifest_row() -> dict:
    with MANIFEST.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)

    raise RuntimeError(f"清单为空：{MANIFEST}")


def resolve_audio_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def build_prompt(row: dict) -> str:
    language = row["language"]
    transcript = row.get("transcript", "")
    labels = ", ".join(row["allowed_labels"])

    if language == "zh":
        return (
            "你是语音情感评测专家。"
            "请仅根据音频中的实际语气、韵律、音高、能量和表达方式，"
            "判断说话者的主导情感。"
            "不要仅根据文本语义推测情感。\n\n"
            f"转写文本：{transcript}\n"
            f"可选标签：{labels}\n\n"
            "必须只输出一个 JSON 对象，不要输出 Markdown 或额外解释：\n"
            '{"predicted_emotion":"<label>",'
            '"confidence":<0.0-1.0>,'
            '"rationale":"<一句简短说明>"}'
        )

    return (
        "You are an expert speech-emotion evaluator. "
        "Judge the dominant emotion from the actual vocal delivery, "
        "including prosody, pitch, energy, and speaking style. "
        "Do not infer emotion from text semantics alone.\n\n"
        f"Transcript: {transcript}\n"
        f"Allowed labels: {labels}\n\n"
        "Return exactly one JSON object. "
        "Do not return Markdown or additional commentary:\n"
        '{"predicted_emotion":"<label>",'
        '"confidence":<0.0-1.0>,'
        '"rationale":"<one short sentence>"}'
    )


def parse_json_object(text: str) -> dict | None:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)

    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    row = load_first_manifest_row()
    audio_path = resolve_audio_path(row["audio_path"])
    prompt = build_prompt(row)

    print("=" * 80)
    print("Qwen2.5-Omni-7B-GPTQ-Int4 单样本 smoke test")
    print("=" * 80)
    print("model_dir            :", MODEL_DIR)
    print("model_revision       :", MODEL_REVISION)
    print("source_revision      :", SOURCE_REVISION)
    print("sample_id            :", row["sample_id"])
    print("language             :", row["language"])
    print("ground_truth_emotion :", row["ground_truth_emotion"])
    print("allowed_labels       :", row["allowed_labels"])
    print("audio_path           :", audio_path)
    print("CUDA available       :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    print("CUDA device          :", torch.cuda.get_device_name(0))

    # 官方低显存 device_map。加载后立即禁用 Talker。
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

    print("\n开始加载 GPTQ 模型……")
    load_start = time.time()

    model = GPTQModel.load(
        str(MODEL_DIR),
        device_map=device_map,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    load_seconds = time.time() - load_start

    print("模型类型             :", type(model))
    print("模型加载耗时（秒）   :", round(load_seconds, 3))
    print(
        "加载后峰值显存（GiB）:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
    )

    if not hasattr(model, "disable_talker"):
        raise RuntimeError(
            "当前模型对象没有 disable_talker()，"
            "停止测试以避免误用语音输出路径。"
        )

    # 不调用 disable_talker()：官方 low-VRAM generate() 仍会访问 self.talker
    model.eval()

    torch.cuda.empty_cache()

    print("Talker 状态           : 保留（low-VRAM 兼容要求）")
    print(
        "当前已分配显存（GiB）:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3,
        ),
    )

    print("\n开始加载 Processor……")

    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

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

    audios, images, videos = process_mm_info(
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

    # 只转换浮点输入，input_ids 等整数张量保持原类型。
    for key, value in inputs.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            inputs[key] = value.to(dtype=torch.float16)

    print("\nProcessor 输出：")
    for key, value in inputs.items():
        if torch.is_tensor(value):
            print(
                f"  {key:24s} "
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype} "
                f"device={value.device}"
            )

    torch.cuda.reset_peak_memory_stats()

    print("\n开始确定性文本生成……")
    infer_start = time.time()

    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            thinker_do_sample=False,
            thinker_max_new_tokens=96,
        )

    torch.cuda.synchronize()
    infer_seconds = time.time() - infer_start

    if isinstance(text_ids, tuple):
        text_ids = text_ids[0]

    if not torch.is_tensor(text_ids) or text_ids.ndim != 2:
        raise RuntimeError(
            f"无法识别 generate() 返回值："
            f"type={type(text_ids)}, "
            f"shape={getattr(text_ids, 'shape', None)}"
        )

    prompt_token_count = inputs["input_ids"].shape[1]

    includes_prompt = (
        text_ids.shape[1] >= prompt_token_count
        and torch.equal(
            text_ids[:, :prompt_token_count].detach().cpu(),
            inputs["input_ids"].detach().cpu(),
        )
    )

    if includes_prompt:
        generated_ids = text_ids[:, prompt_token_count:]
    else:
        generated_ids = text_ids

    print("输入 token 数量       :", prompt_token_count)
    print("返回 token 总数       :", text_ids.shape[1])
    print("返回值包含输入 prompt :", includes_prompt)
    print("新生成 token 数量     :", generated_ids.shape[1])

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw_output = decoded[0].strip()
    parsed = parse_json_object(raw_output)

    print("\n" + "=" * 80)
    print("模型原始输出")
    print("=" * 80)
    print(raw_output)

    print("\nJSON 解析结果：")
    print(
        json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
        )
        if parsed is not None
        else "FAILED"
    )

    predicted = None

    if parsed is not None:
        predicted = str(
            parsed.get("predicted_emotion", "")
        ).strip().lower()

    legal_label = predicted in row["allowed_labels"]

    result = {
        "sample_id": row["sample_id"],
        "audio_path": row["audio_path"],
        "audio_sha256": row["audio_sha256"],
        "language": row["language"],
        "allowed_labels": row["allowed_labels"],
        "ground_truth_emotion": row[
            "ground_truth_emotion"
        ],
        "predicted_emotion": predicted,
        "legal_label": legal_label,
        "correct": (
            predicted == row["ground_truth_emotion"]
            if legal_label
            else False
        ),
        "parsed_json": parsed,
        "raw_output": raw_output,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "attention_implementation": "sdpa",
        "thinker_do_sample": False,
        "thinker_max_new_tokens": 96,
        "load_seconds": round(load_seconds, 6),
        "inference_seconds": round(infer_seconds, 6),
        "peak_inference_gpu_gib": round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            6,
        ),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Smoke test 汇总")
    print("=" * 80)
    print("predicted_emotion     :", predicted)
    print("legal_label           :", legal_label)
    print("ground_truth_emotion  :", row["ground_truth_emotion"])
    print("correct               :", result["correct"])
    print("推理耗时（秒）       :", round(infer_seconds, 3))
    print(
        "推理峰值显存（GiB） :",
        result["peak_inference_gpu_gib"],
    )
    print("结果文件             :", OUTPUT_FILE)

    if parsed is None:
        raise RuntimeError("模型输出无法解析为 JSON")

    if not legal_label:
        raise RuntimeError(
            f"模型返回非法标签：{predicted!r}"
        )


if __name__ == "__main__":
    main()
