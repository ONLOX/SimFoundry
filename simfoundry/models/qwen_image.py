# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alibaba Model Studio image-editing client used by reconstruction stages."""

from __future__ import annotations

import base64
import math
import mimetypes
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from simfoundry.models.remote_cache import RemoteModelCache, image_digests
from simfoundry.models.vlm import RemoteCallFailed, handle_remote_exception, load_api_keys


DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "aigc/multimodal-generation/generation"
)
MIN_OUTPUT_PIXELS = 512 * 512
MAX_OUTPUT_PIXELS = 2048 * 2048


class QwenImageResult:
    """In-memory images returned by one Qwen editing request."""

    def __init__(self, images):
        self.images = list(images)


class QwenHTTPError(RuntimeError):
    """DashScope HTTP failure retaining status for retry classification."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _serialize_result(result: QwenImageResult) -> dict:
    return {"images_base64": [_image_to_base64(image) for image in result.images]}


def _deserialize_result(payload: dict) -> QwenImageResult:
    images = []
    for encoded in payload["images_base64"]:
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        images.append(image)
    return QwenImageResult(images)


def _find_image_locations(payload) -> list[str]:
    """Find image URLs/data URIs across current DashScope response layouts."""
    locations = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"image", "url"} and isinstance(child, str):
                    if child.startswith(("http://", "https://", "data:image/")):
                        locations.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload.get("output", payload) if isinstance(payload, dict) else payload)
    return list(dict.fromkeys(locations))


def _valid_output_size(size) -> tuple[int, int]:
    """Scale to Qwen's pixel-count range while preserving aspect ratio."""
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Qwen output dimensions must be positive, got {(width, height)}")
    pixels = width * height
    if pixels < MIN_OUTPUT_PIXELS:
        scale = math.sqrt(MIN_OUTPUT_PIXELS / pixels)
        width = math.ceil(width * scale / 8) * 8
        height = math.ceil(height * scale / 8) * 8
    elif pixels > MAX_OUTPUT_PIXELS:
        scale = math.sqrt(MAX_OUTPUT_PIXELS / pixels)
        width = max(8, math.floor(width * scale / 8) * 8)
        height = max(8, math.floor(height * scale / 8) * 8)
    return width, height


class QwenImage:
    """Qwen Image 3.0 adapter with the interface used by stages 5 and 6."""

    IMAGE_SHAPES = {
        (1024, 1024),
        (864, 1184),
        (1184, 864),
        (736, 1408),
        (1408, 736),
    }
    VERSIONS = {"qwen-image-3.0", "qwen-image-3.0-pro"}

    def __init__(
        self,
        model: str = "qwen-image-3.0",
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float | None = None,
        session=None,
    ):
        if model not in self.VERSIONS:
            raise ValueError(f"Unsupported Qwen image model {model!r}; choose from {sorted(self.VERSIONS)}")
        load_api_keys()
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.endpoint = endpoint or os.environ.get("DASHSCOPE_IMAGE_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout_s = timeout_s or float(os.environ.get("SIMFOUNDRY_QWEN_TIMEOUT_S", "300"))
        self.session = session or requests.Session()

    @staticmethod
    def _input_data_uri(image_path: str | Path) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _load_location(self, location: str) -> Image.Image:
        if location.startswith("data:image/"):
            encoded = location.split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(encoded)))
        else:
            response = self.session.get(location, timeout=self.timeout_s)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    def __call__(
        self,
        prompt,
        image_path=None,
        image_paths=None,
        seed=0,
        n_retries=3,
        size=None,
        print_results=False,
        **_unused,
    ):
        """Edit one image using a natural-language instruction."""
        source = image_path or image_paths
        if isinstance(source, (list, tuple)):
            if len(source) != 1:
                raise ValueError("The minimal Qwen adapter accepts exactly one input image")
            source = source[0]
        if source is None:
            raise ValueError("Qwen image editing requires image_path or image_paths")
        source = str(source)

        if size is None:
            with Image.open(source) as input_image:
                size = input_image.size
        width, height = _valid_output_size(size)
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_input": image_digests(source),
            "seed": int(seed),
            "size": [width, height],
            "prompt_extend": False,
        }
        cache_key = cache.key_for(provider="qwen-image", model=self.model, request=cache_request)
        if cache.test_enabled:
            return _deserialize_result(cache.load_response(provider="qwen-image", key=cache_key))
        if cache.cache_enabled:
            cached = cache.load_response_if_exists(provider="qwen-image", key=cache_key)
            if cached is not None:
                return _deserialize_result(cached)
        if not self.api_key:
            raise ValueError(
                "Qwen Image requires DASHSCOPE_API_KEY (or DASHSCOPE_API_KEY in api_keys.txt)"
            )

        payload = {
            "model": self.model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image": self._input_data_uri(source)},
                        {"text": str(prompt)},
                    ],
                }],
            },
            "parameters": {
                "n": 1,
                "size": f"{width}*{height}",
                "seed": int(seed),
                "prompt_extend": False,
                "watermark": False,
            },
        }
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
                locations = _find_image_locations(response_payload)
                if not locations:
                    raise RuntimeError(f"Qwen Image response contains no image: {response_payload}")
                result = QwenImageResult([self._load_location(locations[0])])
            except Exception as exc:
                last_exc = exc
                budget = handle_remote_exception(
                    exc,
                    attempt=attempt,
                    n_retries=budget,
                    provider="Qwen Image",
                    model=self.model,
                )
            attempt += 1
        if result is None:
            raise RemoteCallFailed(
                f"Qwen Image [{self.model}] failed after {budget} attempts: {last_exc}"
            ) from last_exc

        if print_results:
            result.images[0].show()
        if cache.cache_enabled:
            cache.store_response(
                provider="qwen-image",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response=_serialize_result(result),
            )
        return result

    @staticmethod
    def get_result_images(result):
        return list(result.images)
