"""A language or vision-language model behind a chat API, shared by every method that asks a model a question:
any server with an OpenAI-compatible HTTP API (vLLM, OpenAI, LM Studio, Ollama, ...), or your own backend.

A call sends chat messages (text and images) and may pass a JSON schema; the server then enforces the shape of
the answer token by token (STRUCTURED OUTPUT), so a caller can restrict the model to, for example, the skills the
policy was trained on. The planner loop makes two kinds of calls this way: a text-only plan request once per
episode and a monitor query with images at every policy call. Implement VLMBackend.chat for another serving stack.

Choosing the backend (the proxies' --vlm-backend, plugins.py): a registered name (VLM_BACKENDS: openai =
OpenAICompatibleVLM; register_vlm_backend adds one), an entry point of the group vla_memory.vlm_backends, or an import path
module:Class. make_vlm() builds it; resolve_api_key() reads --vlm-api-key or the environment variable VLM_API_KEY.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

import numpy as np

from .plugins import construct, resolve

VLM_BACKENDS: dict[str, Any] = {"openai": "vla_memory.vlm:OpenAICompatibleVLM"}   # name -> class or import path
VLM_ENTRY_POINTS = "vla_memory.vlm_backends"
API_KEY_ENV = "VLM_API_KEY"


def encode_png(img: np.ndarray) -> str:
    """numpy RGB image -> data URL for the chat API."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VLMBackend:
    """Interface. chat() returns {"text", "json" (if a schema was given), "ms", "prompt_tokens", "completion_tokens",
    "reasoning"}; messages follow the OpenAI chat format with image_url content items."""

    def chat(self, messages: list[dict], schema: dict | None = None, name: str = "answer",
             max_tokens: int | None = None) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleVLM(VLMBackend):
    """Minimal client for /v1/chat/completions with response_format=json_schema (vLLM's structured output).

    chat_template_kwargs is passed through to the server; the default (None) disables Qwen3's "thinking" mode, which is
    what we ran. Pass {} for servers that reject unknown fields: then no chat_template_kwargs are sent. api_key is sent as
    "Authorization: Bearer <key>".
    """

    def __init__(self, base_url="http://127.0.0.1:8100/v1", model="Qwen3.8-27B-INT4", timeout=300, temperature=0.0,
                 max_tokens=512, chat_template_kwargs: dict | None = None, api_key: str | None = None,
                 extra_body: dict | None = None):
        import requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        self.extra_body = extra_body or {}
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages, schema=None, name="answer", max_tokens=None):
        """One request: the answer text, its JSON (when a schema was given), the latency in ms and the token counts."""
        body = self._request_body(messages, schema, name, max_tokens)
        started = time.perf_counter()
        reply = self._post(body)
        message = reply["choices"][0]["message"]
        text = message.get("content") or ""
        usage = reply.get("usage", {})
        out = {"text": text, "ms": 1000 * (time.perf_counter() - started),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "reasoning": message.get("reasoning_content") or message.get("reasoning")}
        if schema is not None:
            out["json"] = _parse_answer(text)
        return out

    def _request_body(self, messages, schema, name, max_tokens) -> dict:
        """The JSON body of one chat request; a schema becomes response_format=json_schema, which the server enforces."""
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens, **self.extra_body}
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        return body

    def _post(self, body: dict) -> dict:
        """Send the request to /chat/completions and return the server's reply as JSON (an HTTP error raises)."""
        response = self.session.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def _parse_answer(text: str):
    """The answer text as JSON. When the whole text is not JSON (the model wrapped it in words), the first {...} block in
    it; None when there is none."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------------------------------------------- choosing one
def register_vlm_backend(name: str, factory) -> None:
    """Make `factory` (a VLMBackend class, or a callable returning one) selectable as --vlm-backend NAME."""
    VLM_BACKENDS[name] = factory


def resolve_vlm_backend(spec):
    """The class --vlm-backend names: a registered name, an entry point of vla_memory.vlm_backends, or module:Class."""
    return resolve(spec, VLM_BACKENDS, VLM_ENTRY_POINTS, "VLM backend")


def make_vlm(spec="openai", options: dict | None = None, **standard) -> VLMBackend:
    """The backend `spec` names, built with the `standard` keyword arguments it declares (e.g. base_url, model, api_key,
    chat_template_kwargs) and every one of `options` (--vlm-arg KEY=VALUE)."""
    return construct(resolve_vlm_backend(spec), standard, options, f"VLM backend {spec!r}")


def resolve_api_key(explicit: str | None = None) -> str | None:
    """The API key of the VLM server: the explicit one (--vlm-api-key), else the environment variable VLM_API_KEY, else None."""
    return explicit or os.environ.get(API_KEY_ENV) or None
