# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from simfoundry.pipeline.stage_utils import write_ffmpeg_concat_list


def test_concat_list_uses_absolute_paths(tmp_path):
    frames = tmp_path / "Data" / "bottle_box" / "s1_video" / "frames_subsampled_400"
    frames.mkdir(parents=True)
    a = frames / "frame_0001.png"
    b = frames / "frame_0016.png"
    a.write_bytes(b"png")
    b.write_bytes(b"png")
    list_path = tmp_path / "Data" / "bottle_box" / "s1_video" / "_splat_frame_list.txt"

    write_ffmpeg_concat_list(list_path, [a, b], duration=1.0 / 12)

    text = list_path.read_text()
    assert f"file '{a.resolve()}'" in text
    assert f"file '{b.resolve()}'" in text
    assert text.count(f"file '{b.resolve()}'") == 2
    assert "duration" in text
    assert ".." not in Path(text.split("'")[1]).parts
