"""
llm.py — LLM backend clients and JSON extraction.

Supported backends
  ollama       — local Ollama server (default, used on DGX)
  openai       — OpenAI-compatible server, e.g. llama.cpp's llama-server, for
                 models Ollama cannot run correctly (Sarvam); URL in LLM_BASE_URL
  transformers — HuggingFace causal LM (offline GPU fallback)

Key settings are read from config.py at the project root.
Change OLLAMA_MODEL or OLLAMA_HOST there, not here.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Optional

# Config values — pulled at import time; override via environment variables.
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import (
        AUDIT_ENABLE_THINKING,
        AUDIT_CONTEXT_TOKENS,
        AUDIT_MAX_TOKENS,
        AUDIT_TEMPERATURE,
        OLLAMA_HOST,
        OLLAMA_MODEL,
    )
except ImportError:
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:32b"
    AUDIT_ENABLE_THINKING = False
    AUDIT_TEMPERATURE = 0.0
    AUDIT_MAX_TOKENS = 6144
    AUDIT_CONTEXT_TOKENS = 32768

TRANSFORMERS_MODEL = "Qwen/Qwen3-8B"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8082/v1")

ENABLE_THINKING = False
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.2

# Single-pass settings. Ratings are derived and validated by Python, so stable
# extraction is preferable to sampled prose or long hidden reasoning traces.
SINGLE_PASS_THINK       = AUDIT_ENABLE_THINKING
SINGLE_PASS_TEMPERATURE = AUDIT_TEMPERATURE
SINGLE_PASS_MAX_TOKENS  = AUDIT_MAX_TOKENS
SINGLE_PASS_CONTEXT_TOKENS = AUDIT_CONTEXT_TOKENS


def single_pass_options() -> Dict[str, Any]:
    """Ollama options of the single-pass audit call (also used by Brio, so the
    model is not reloaded with a different context size)."""
    return {
        "temperature": SINGLE_PASS_TEMPERATURE,
        "num_predict": SINGLE_PASS_MAX_TOKENS,
        "num_ctx": SINGLE_PASS_CONTEXT_TOKENS,
    }


def _supports_thinking(model: str) -> bool:
    return "qwen3" in (model or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    def generate(  # pragma: no cover
        self,
        system: str,
        user: str,
        format_schema: Optional[Dict[str, Any]] = None,
        *,
        single_pass: bool = False,
    ) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA (primary — used on DGX)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient(LLMClient):
    """Talks to a local Ollama server. keep_alive=-1 keeps model resident in VRAM."""

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                 keep_alive: int = -1):
        self.model = model
        self.host = host.rstrip("/")
        self._keep_alive = keep_alive
        self._local = threading.local()

    def get_last_metrics(self) -> Dict[str, Any]:
        """Return Ollama timing/token counters for the current worker thread."""
        return dict(getattr(self._local, "last_metrics", {}))

    def _save_metrics(self, response: Dict[str, Any]) -> None:
        fields = (
            "prompt_eval_count", "eval_count", "total_duration",
            "load_duration", "prompt_eval_duration", "eval_duration",
            "done", "done_reason",
        )
        metrics = {key: response.get(key) for key in fields if response.get(key) is not None}
        for key in list(metrics):
            if key.endswith("_duration"):
                metrics[f"{key}_s"] = round(float(metrics.pop(key)) / 1_000_000_000, 3)
        self._local.last_metrics = metrics

    def generate(
        self,
        system: str,
        user: str,
        format_schema: Optional[Dict[str, Any]] = None,
        *,
        single_pass: bool = False,
    ) -> str:
        """Send one chat completion to Ollama.

        Parameters
        ----------
        format_schema : When supplied, used as the Ollama structured-output
            ``format`` value (a JSON Schema dict).  When None, falls back to
            the string ``"json"`` for backward-compatible generic JSON mode.
        single_pass   : When True, uses deterministic audit settings from
            config.py / environment variables.
        """
        fmt: Any = format_schema if format_schema is not None else "json"

        if single_pass:
            think     = SINGLE_PASS_THINK if _supports_thinking(self.model) else False
            options: Dict[str, Any] = single_pass_options()
        else:
            think   = ENABLE_THINKING
            options = {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS}

        return self._chat(
            system, user, fmt, options,
            think if _supports_thinking(self.model) else None,
        )

    def preload(self, options: Dict[str, Any]) -> None:
        """Load the model into Ollama memory without generating anything."""
        import urllib.request
        data = json.dumps({
            "model": self.model, "keep_alive": self._keep_alive, "options": options,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()

    def generate_ratings(
        self, system: str, user: str, options: Dict[str, Any], timeout: float
    ) -> str:
        """Send one request in the fine-tuned rating model's trained contract.

        The model was trained with thinking disabled and with the JSON
        requirement expressed only in prose, so no ``format`` grammar is sent.
        """
        return self._chat(system, user, None, options, False, timeout)

    def complete_raw(
        self,
        prompt: str,
        options: Dict[str, Any],
        *,
        top_logprobs: int = 0,
        fmt: Any = None,
        timeout: float = 600,
    ) -> Dict[str, Any]:
        """Continue an already-templated prompt (Ollama raw mode, no chat template).

        Used by Brio scoring: ``top_logprobs`` returns the most likely next
        tokens with their log-probabilities; ``fmt`` constrains the
        continuation to a JSON Schema. Returns Ollama's full response.
        """
        import urllib.request
        payload: Dict[str, Any] = {
            "model": self.model, "prompt": prompt, "raw": True, "stream": False,
            "options": options, "keep_alive": self._keep_alive,
        }
        if top_logprobs:
            payload.update({"logprobs": True, "top_logprobs": top_logprobs})
        if fmt is not None:
            payload["format"] = fmt
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _chat(
        self,
        system: str,
        user: str,
        fmt: Any,
        options: Dict[str, Any],
        think: Optional[bool],
        timeout: float = 600,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self._keep_alive,
        }
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think

        try:
            import ollama  # type: ignore
            # Prevent a stalled local model request from holding the only
            # audit worker indefinitely.
            client = ollama.Client(host=self.host, timeout=timeout)
            kwargs = {key: value for key, value in payload.items() if key != "stream"}
            resp = client.chat(**kwargs)
            self._save_metrics(resp)
            return resp["message"]["content"]
        except ImportError:
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
            self._save_metrics(out)
            return out["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI-COMPATIBLE SERVER (llama.cpp llama-server, vLLM, ...)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatClient(LLMClient):
    """Chat completions on an OpenAI-compatible server. The answer is the message
    content; reasoning the server returns separately (reasoning_content) is dropped.
    max_tokens covers reasoning too, so reasoning models need AUDIT_MAX_TOKENS raised."""

    def __init__(self, model: str = "", base_url: str = LLM_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._local = threading.local()

    def get_last_metrics(self) -> Dict[str, Any]:
        return dict(getattr(self._local, "last_metrics", {}))

    def generate(
        self,
        system: str,
        user: str,
        format_schema: Optional[Dict[str, Any]] = None,
        *,
        single_pass: bool = False,
    ) -> str:
        import urllib.request
        payload: Dict[str, Any] = {
            "model": self.model or "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": SINGLE_PASS_TEMPERATURE if single_pass else TEMPERATURE,
            "max_tokens": SINGLE_PASS_MAX_TOKENS if single_pass else MAX_NEW_TOKENS,
            "response_format": (
                {"type": "json_schema", "json_schema": {"name": "audit", "schema": format_schema}}
                if format_schema is not None else {"type": "json_object"}
            ),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read().decode("utf-8"))
        usage, timings = out.get("usage") or {}, out.get("timings") or {}
        choice = out["choices"][0]
        self._local.last_metrics = {
            "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_duration_s": round(float(timings.get("prompt_ms", 0)) / 1000, 3),
            "eval_duration_s": round(float(timings.get("predicted_ms", 0)) / 1000, 3),
            "done_reason": choice.get("finish_reason"),
        }
        return choice["message"].get("content") or ""


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMERS (offline GPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

class TransformersClient(LLMClient):
    """Loads an HF causal LM — use when Ollama is unavailable."""

    def __init__(self, model_id: str = TRANSFORMERS_MODEL):
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )

    def generate(
        self,
        system: str,
        user: str,
        format_schema: Optional[Dict[str, Any]] = None,
        *,
        single_pass: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        think = SINGLE_PASS_THINK if single_pass else ENABLE_THINKING
        template_kwargs: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if _supports_thinking(getattr(self, "model_id", "")):
            template_kwargs["enable_thinking"] = think
        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        temp = SINGLE_PASS_TEMPERATURE if single_pass else TEMPERATURE
        max_tok = SINGLE_PASS_MAX_TOKENS if single_pass else MAX_NEW_TOKENS
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_tok, temperature=temp,
                do_sample=temp > 0, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_client(backend: str, model: Optional[str] = None) -> LLMClient:
    backend = backend.lower()
    if backend == "ollama":
        return OllamaClient(model=model or OLLAMA_MODEL)
    if backend == "openai":
        return OpenAICompatClient(model=model or "")
    if backend == "transformers":
        return TransformersClient(model_id=model or TRANSFORMERS_MODEL)
    raise ValueError(f"Unknown backend: {backend!r} (use 'ollama', 'openai' or 'transformers')")


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from raw LLM output."""
    # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("Could not extract valid JSON from model output.")
