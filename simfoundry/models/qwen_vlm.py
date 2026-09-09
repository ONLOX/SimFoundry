# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alibaba Model Studio vision-language client used by reconstruction VLM steps."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests

from simfoundry.models.qwen_image import QwenHTTPError
from simfoundry.models.remote_cache import RemoteModelCache, image_digests
from simfoundry.models.vlm import RemoteCallFailed, handle_remote_exception, load_api_keys


DEFAULT_VLM_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
INTL_VLM_ENDPOINT = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
)


def default_vlm_endpoint() -> str:
    """Prefer an explicit VLM URL, else match the image endpoint's region."""
    explicit = os.environ.get("DASHSCOPE_VLM_ENDPOINT")
    if explicit:
        return explicit
    image_ep = os.environ.get("DASHSCOPE_IMAGE_ENDPOINT", "")
    if "dashscope-intl" in image_ep:
        return INTL_VLM_ENDPOINT
    return DEFAULT_VLM_ENDPOINT


class QwenVLResult:
    """Text returned by one Qwen VL request."""

    def __init__(self, text: str):
        self.text = text


def _data_uri(path: str | Path) -> str:
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _message_text(payload: dict) -> str:
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "".join(parts)
    raise RuntimeError(f"Qwen VL response contains no text: {payload}")


class QwenVL:
    """DashScope Qwen VL adapter with the text interface used by Gemini callers."""

    VERSIONS = {
        "qwen3-vl-flash",
        "qwen-vl-plus",
        "qwen3-vl-plus",
        "qwen-vl-max",
    }

    def __init__(
        self,
        model: str = "qwen3-vl-flash",
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float | None = None,
        session=None,
    ):
        if model not in self.VERSIONS:
            raise ValueError(
                f"Unsupported Qwen VL model {model!r}; choose from {sorted(self.VERSIONS)}"
            )
        load_api_keys()
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.endpoint = endpoint or default_vlm_endpoint()
        self.timeout_s = timeout_s or float(os.environ.get("SIMFOUNDRY_QWEN_TIMEOUT_S", "300"))
        self.session = session or requests.Session()

    def __call__(
        self,
        prompt,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
        **_unused,
    ):
        image_paths = (
            []
            if image_paths is None
            else [image_paths]
            if isinstance(image_paths, (str, os.PathLike))
            else list(image_paths)
        )
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_inputs": image_digests(image_paths),
            "temperature": temperature,
            "top_p": top_p,
            "seed": int(seed),
        }
        cache_key = cache.key_for(provider="qwen-vl", model=self.model, request=cache_request)
        if cache.test_enabled:
            return QwenVLResult(cache.load_response(provider="qwen-vl", key=cache_key)["text"])
        if cache.cache_enabled:
            cached = cache.load_response_if_exists(provider="qwen-vl", key=cache_key)
            if cached is not None:
                return QwenVLResult(cached["text"])
        if not self.api_key:
            raise ValueError(
                "Qwen VL requires DASHSCOPE_API_KEY (or DASHSCOPE_API_KEY in api_keys.txt)"
            )

        content = [{"type": "image_url", "image_url": {"url": _data_uri(path)}} for path in image_paths]
        content.append({"type": "text", "text": str(prompt)})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "seed": int(seed),
            "enable_thinking": False,
        }
        if top_p and float(top_p) > 0:
            payload["top_p"] = float(top_p)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        result = None
        budget = n_retries
        last_exc = None
        attempt = 0
        while attempt < budget and result is None:
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                if not response.ok:
                    detail = response.text.strip()
                    if len(detail) > 2000:
                        detail = detail[:2000] + "..."
                    raise QwenHTTPError(
                        response.status_code,
                        detail or response.reason or "empty response body",
                    )
                response_payload = response.json()
                if response_payload.get("code"):
                    raise RuntimeError(
                        f"{response_payload.get('code')}: {response_payload.get('message', response_payload)}"
                    )
                text = _message_text(response_payload)
                result = QwenVLResult(text)
            except Exception as exc:
                last_exc = exc
                budget = handle_remote_exception(
                    exc,
                    attempt=attempt,
                    n_retries=budget,
                    provider="Qwen VL",
                    model=self.model,
                )
            attempt += 1
        if result is None:
            raise RemoteCallFailed(
                f"Qwen VL [{self.model}] failed after {budget} attempts: {last_exc}"
            ) from last_exc

        if print_results:
            print(result.text)
        if cache.cache_enabled:
            cache.store_response(
                provider="qwen-vl",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response={"text": result.text},
            )
        return result

    @staticmethod
    def get_result_text(result):
        return result.text
