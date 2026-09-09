# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

from simfoundry.models.qwen_vlm import QwenVL, default_vlm_endpoint
from simfoundry.models.vlm import create_vlm


class _Response:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = ""
        self.reason = "OK"

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def test_qwen_vl_uses_dashscope_compatible_payload(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    image = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), "white").save(image)
    session = _Session({
        "choices": [{"message": {"content": "ANSWER: 2"}}],
    })
    model = QwenVL(api_key="test-key", session=session)

    result = model(prompt="pick a frame", image_paths=str(image))

    assert result.text == "ANSWER: 2"
    url, request = session.calls[0]
    assert url.endswith("/compatible-mode/v1/chat/completions")
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "qwen3-vl-flash"
    assert request["json"]["enable_thinking"] is False
    content = request["json"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "pick a frame"}


def test_create_vlm_routes_qwen_and_rejects_unknown(monkeypatch):
    vlm = create_vlm("qwen3-vl-flash", api_key="test-key")
    assert isinstance(vlm, QwenVL)
    assert vlm.model == "qwen3-vl-flash"


def test_default_vlm_endpoint_follows_image_region(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_VLM_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "DASHSCOPE_IMAGE_ENDPOINT",
        "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    )
    assert "dashscope-intl" in default_vlm_endpoint()
