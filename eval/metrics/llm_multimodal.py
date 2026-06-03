from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

_REPO_ROOT = Path(__file__).parent.parent.parent

MIME_TYPE_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}

_CACHE_ROOT = _REPO_ROOT / ".cache" / "eval_llm_judge"
_CACHE_LOCK = threading.Lock()
_CACHE_MEMORY: dict[Path, dict[str, dict]] = {}


def _parse_json_text(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _load_cache(cache_file: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_file.exists():
        return cache
    with cache_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("key", "")).strip()
            obj = row.get("obj")
            if key and isinstance(obj, dict):
                cache[key] = obj
    return cache


def _append_cache(cache_file: Path, key: str, obj: dict, *, model: str, audio_path: Path, caller_tag: str) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.utcnow().isoformat(),
        "key": key,
        "caller_tag": caller_tag,
        "model": model,
        "audio_path": str(audio_path),
        "obj": obj,
    }
    with _CACHE_LOCK:
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _CACHE_MEMORY.setdefault(cache_file, {})[key] = obj


def _default_cache_key(audio_path: Path, prompt: str, model: str, caller_tag: str) -> str:
    payload = f"{caller_tag}\n{model}\n{audio_path.resolve()}\n{prompt}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def call_multimodal_llm_json(
    audio_path: Path,
    prompt: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gemini-2.5-pro",
    timeout: int = 120,
    max_retries: int = 3,
    caller_tag: str = "eval_multimodal_judge",
    cache_key: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Call Gemini-compatible multimodal endpoint and return parsed JSON object.
    """
    if requests is None:
        raise RuntimeError("requests is required: pip install requests")

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY for multimodal judge")

    root = (base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = f"{root}/v1beta/models/{model}:generateContent"

    key_for_cache = cache_key or _default_cache_key(audio_path, prompt, model, caller_tag)
    cache_file = _CACHE_ROOT / f"{caller_tag}.jsonl"
    if use_cache:
        with _CACHE_LOCK:
            cache = _CACHE_MEMORY.get(cache_file)
            if cache is None:
                cache = _load_cache(cache_file)
                _CACHE_MEMORY[cache_file] = cache
        if key_for_cache in cache:
            return cache[key_for_cache]

    ext = audio_path.suffix.lower()
    mime_type = MIME_TYPE_MAP.get(ext, "audio/wav")
    with audio_path.open("rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    headers = {"Content-Type": "application/json"}
    if "generativelanguage.googleapis.com" in root:
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }

    delay = 3.0
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"retryable_http_{resp.status_code}: {resp.text[:200]}")
                raise RuntimeError(f"http_{resp.status_code}: {resp.text[:300]}")
            data = resp.json()

            text = data["candidates"][0]["content"]["parts"][0]["text"]
            obj = _parse_json_text(text)
            if obj is None:
                raise RuntimeError(f"invalid_json_output: {text[:300]}")
            if use_cache:
                _append_cache(
                    cache_file,
                    key_for_cache,
                    obj,
                    model=model,
                    audio_path=audio_path,
                    caller_tag=caller_tag,
                )
            return obj
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"multimodal_llm_failed_after_retries: {last_error}")
