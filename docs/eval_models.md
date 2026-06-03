# Evaluator Models

Large evaluator models are not stored in this repository. Place them under
`eval_models/` when running the full automatic metrics.

Run commands from the repository root. The main benchmark environment used for
this release is Python 3.11; the UTMOS demo keeps its own Python 3.10
environment because its upstream dependency stack is older.

```bash
conda create -n speecheditbench python=3.11
conda activate speecheditbench
pip install -r requirements.txt
```

Run the setup checker after installing packages and model assets:

```bash
python scripts/check_eval_setup.py
```

Use `--check-hf` to also verify anonymous access to the Hugging Face dataset
release, and `--strict` when using the checker in CI.

## Expected Paths

| Component | Expected path | Used for |
|---|---|---|
| Whisper large-v3 | `eval_models/asr/whisper-large-v3` | English ASR |
| Paraformer zh | `eval_models/asr/paraformer-zh` | Chinese ASR |
| Paraformer zh timestamp | `eval_models/asr/paraformer-zh-vad-punc-timestamp` | Chinese stress timestamps |
| WavLM pretrained | `eval_models/speaker/wavlm-large-pretrained` | Speaker embeddings |
| WavLM speaker head | `eval_models/speaker/wavlm-large-sv/wavlm-large.pt` | Speaker verification |
| UTMOS demo | `eval_models/mos/UTMOS-demo` | Naturalness diagnostics |
| UTMOS environment | `eval_models/mos/utmos22_env/bin/python` | UTMOS subprocess |
| DNSMOS ONNX files | `eval_models/mos/DNSMOS` | Enhancement metrics |
| PANNs CNN14 | `eval_models/acoustic/panns-cnn14/Cnn14_mAP=0.431.pth` | Acoustic scene metrics |

## Download Commands

### English ASR: Whisper Large-v3

Source: <https://huggingface.co/openai/whisper-large-v3>

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/whisper-large-v3",
    local_dir="eval_models/asr/whisper-large-v3",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "added_tokens.json",
        "model.safetensors.index.fp32.json",
        "model.fp32-*.safetensors",
    ],
)
PY
```

### Chinese ASR: Paraformer

Sources:

- <https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch>
- <https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch>

```bash
python - <<'PY'
from modelscope import snapshot_download

COMMON = [
    "config.yaml",
    "configuration.json",
    "am.mvn",
    "model.pt",
    "seg_dict",
    "tokens.json",
]

snapshot_download(
    model_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    local_dir="eval_models/asr/paraformer-zh",
    allow_patterns=COMMON,
)
snapshot_download(
    model_id="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    local_dir="eval_models/asr/paraformer-zh-vad-punc-timestamp",
    allow_patterns=COMMON,
)
PY
```

### Speaker Similarity: WavLM + ECAPA-TDNN

Sources:

- WavLM SSL model: <https://huggingface.co/microsoft/wavlm-large>
- WavLM/UniSpeech speaker-verification setup: <https://github.com/microsoft/UniSpeech>
- Public checkpoint mirror used by this benchmark layout: <https://huggingface.co/yfyeung/wavlm-large-speaker-verification>

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/wavlm-large",
    local_dir="eval_models/speaker/wavlm-large-pretrained",
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    ],
)
snapshot_download(
    repo_id="yfyeung/wavlm-large-speaker-verification",
    local_dir="eval_models/speaker/wavlm-large-sv",
    allow_patterns=[
        "wavlm-large.pt",
        "ecapa_tdnn.py",
    ],
)
PY
```

### UTMOS Diagnostics

Source: <https://huggingface.co/spaces/sarulab-speech/UTMOS-demo>

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sarulab-speech/UTMOS-demo",
    repo_type="space",
    local_dir="eval_models/mos/UTMOS-demo",
    allow_patterns=[
        "score.py",
        "lightning_module.py",
        "epoch=3-step=7459.ckpt",
        "requirements.txt",
    ],
)
PY

conda create -y -p eval_models/mos/utmos22_env python=3.10
eval_models/mos/utmos22_env/bin/python -m pip install -r eval_models/mos/UTMOS-demo/requirements.txt
```

UTMOS is a naturalness diagnostic in the released evaluators. If it is missing,
task evaluators record `null` for UTMOS fields and continue.

### DNSMOS

DNSMOS ONNX files are downloaded on demand by `eval/metrics/dnsmos.py`. To
download them explicitly:

```bash
python - <<'PY'
from eval.metrics.dnsmos import _ensure_models

_ensure_models()
PY
```

Source files are hosted in the Microsoft DNS-Challenge repository:

- <https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx>
- <https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx>

### PANNs CNN14

Source: <https://zenodo.org/records/3987831>

```bash
python - <<'PY'
from pathlib import Path
import requests

url = "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1"
dest = Path("eval_models/acoustic/panns-cnn14/Cnn14_mAP=0.431.pth")
dest.parent.mkdir(parents=True, exist_ok=True)
with requests.get(url, stream=True, timeout=60) as response:
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
PY
```

## Reference File Sizes

The commands above were checked against the evaluator assets used for release
validation. The most important local reference sizes are:

| File | Bytes |
|---|---:|
| `eval_models/asr/whisper-large-v3/model.fp32-00001-of-00002.safetensors` | 4,993,448,880 |
| `eval_models/asr/whisper-large-v3/model.fp32-00002-of-00002.safetensors` | 1,180,663,192 |
| `eval_models/asr/paraformer-zh/model.pt` | 880,502,012 |
| `eval_models/asr/paraformer-zh-vad-punc-timestamp/model.pt` | 900,702,648 |
| `eval_models/speaker/wavlm-large-pretrained/pytorch_model.bin` | 1,261,990,257 |
| `eval_models/speaker/wavlm-large-sv/wavlm-large.pt` | 1,301,926,579 |
| `eval_models/mos/UTMOS-demo/epoch=3-step=7459.ckpt` | 1,238,128,841 |
| `eval_models/acoustic/panns-cnn14/Cnn14_mAP=0.431.pth` | 327,428,481 |
| `eval_models/mos/DNSMOS/sig_bak_ovr.onnx` | 1,157,965 |
| `eval_models/mos/DNSMOS/model_v8.onnx` | 224,860 |

Use this command to compare local file sizes after download:

```bash
python - <<'PY'
from pathlib import Path

for path in [
    "eval_models/asr/whisper-large-v3/model.fp32-00001-of-00002.safetensors",
    "eval_models/asr/whisper-large-v3/model.fp32-00002-of-00002.safetensors",
    "eval_models/asr/paraformer-zh/model.pt",
    "eval_models/asr/paraformer-zh-vad-punc-timestamp/model.pt",
    "eval_models/speaker/wavlm-large-pretrained/pytorch_model.bin",
    "eval_models/speaker/wavlm-large-sv/wavlm-large.pt",
    "eval_models/mos/UTMOS-demo/epoch=3-step=7459.ckpt",
    "eval_models/acoustic/panns-cnn14/Cnn14_mAP=0.431.pth",
    "eval_models/mos/DNSMOS/sig_bak_ovr.onnx",
    "eval_models/mos/DNSMOS/model_v8.onnx",
]:
    file = Path(path)
    print(f"{path}\t{file.stat().st_size if file.exists() else 'MISSING'}")
PY
```

## Notes

If an optional diagnostic model is missing, many task evaluators will fill the
diagnostic field with `null` and continue. ASR models are required for content
preservation and content editing metrics. Gemini-compatible judge tasks require
`GEMINI_API_KEY`.
