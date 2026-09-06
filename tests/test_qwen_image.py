# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

from PIL import Image

from simfoundry.models.qwen_image import MIN_OUTPUT_PIXELS, QwenImage, _valid_output_size


def _data_uri(color=(12, 34, 56)):
    image = Image.new("RGB", (32, 24), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = ""
        self.reason = "OK"

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, image_uri):
        self.image_uri = image_uri
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response({
            "output": {
                "choices": [{
                    "message": {"content": [{"image": self.image_uri}]},
                }],
            },
        })

    def get(self, *_args, **_kwargs):
        raise AssertionError("data URI responses must not be downloaded")


def test_qwen_image_uses_dashscope_payload(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 512), "white").save(source)
    session = _Session(_data_uri())
    model = QwenImage(api_key="test-key", session=session)

    result = model(prompt="remove the apple", image_path=source, size=(1024, 1024))

    assert result.images[0].size == (32, 24)
    _, request = session.calls[0]
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "qwen-image-3.0"
    assert request["json"]["parameters"]["size"] == "1024*1024"
    content = request["json"]["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[1] == {"text": "remove the apple"}


def test_qwen_image_scales_small_stage5_output_to_provider_minimum():
    width, height = _valid_output_size((672, 384))
    assert width * height >= MIN_OUTPUT_PIXELS
    assert abs(width / height - 672 / 384) < 0.02


def test_qwen_image_cache_replays_without_api_call(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 512), "white").save(source)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.delenv("TEST_MODE", raising=False)
    session = _Session(_data_uri((1, 2, 3)))
    QwenImage(api_key="test-key", session=session)(
        prompt="edit", image_path=source, size=(1024, 1024)
    )

    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    replay = QwenImage(api_key=None, session=_Session("unused"))(
        prompt="edit", image_path=source, size=(1024, 1024)
    )
    assert replay.images[0].getpixel((0, 0)) == (1, 2, 3)
