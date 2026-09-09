from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from comfy_api.input_impl import VideoFromFile
import h3_audio_t8_pkg.video_outpaint_preview as module
from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import ASPECTS, build_outpaint_plan


def _plan(**kwargs):
    return build_outpaint_plan(**({"source_sha256": "a" * 64, "width": 736, "height": 416,
                                  "frame_count": 80, "cut_frames": (40,)} | kwargs))


@pytest.mark.parametrize("aspect", ASPECTS)
def test_geometry_uses_authoritative_plan_and_never_allocates_output_canvas(aspect):
    margins = {"left": 17, "top": 23, "right": 49, "bottom": 61} if aspect == "custom" else {}
    plan = _plan(aspect=aspect, **margins)
    original = deepcopy(plan)
    report = module.preview_geometry(plan, 40, 128)
    assert plan == original
    assert report["source_rect_output"] == plan["output"]["source_rect"]
    assert report["margins_left_top_right_bottom"] == plan["output"]["margins"]
    assert max(report["thumbnail_width"], report["thumbnail_height"]) <= 128
    x0, y0, x1, y1 = report["source_rect_thumbnail"]
    assert 0 <= x0 < x1 <= report["thumbnail_width"]
    assert 0 <= y0 < y1 <= report["thumbnail_height"]
    assert report["sampling_megapixels"] == plan["sampling"]["pixels"] / 1e6
    assert report["shot_index"] == 1 and report["time_seconds"] == "5/3"
    assert not report["model_called"] and not report["generated_reference_selected"]


def test_odd_unscaled_geometry_and_thumbnail_paste_are_exact_without_label_overlay():
    plan = _plan(width=73, height=41, aspect="custom", left=13, top=7, right=21, bottom=9)
    report = module.preview_geometry(plan)
    assert report["source_rect_thumbnail"] == [13, 7, 86, 48]
    rgb = bytes([180, 70, 29]) * (73 * 41)
    image = module._draw_preview(rgb, report)
    assert image.crop((13, 7, 86, 48)).tobytes() == rgb
    assert image.height == plan["output"]["height"] + 104
    assert image.getpixel((0, 0)) != (180, 70, 29)


def test_large_output_still_uses_only_a_bounded_thumbnail():
    plan = _plan(width=6000, height=4000, aspect="custom", left=500, right=500, top=1000, bottom=1000)
    report = module.preview_geometry(plan, 79, 128)
    assert report["output_width"] == 7000 and report["output_height"] == 6000
    rect = report["source_rect_thumbnail"]
    rgb = bytes((rect[2] - rect[0]) * (rect[3] - rect[1]) * 3)
    image = module._draw_preview(rgb, report)
    assert image.width <= 480 and image.height <= 128 + 104


@pytest.mark.parametrize("frame,edge", [(-1, 768), (80, 768), (True, 768), (0.5, 768), (0, True), (0, 63), (0, 1537)])
def test_invalid_preview_parameters_are_rejected(frame, edge):
    with pytest.raises(ValueError, match="preview"):
        module.preview_geometry(_plan(), frame, edge)


def test_tampered_plan_is_not_reinterpreted_as_different_preview():
    plan = _plan()
    plan["output"]["source_rect"][0] += 1
    with pytest.raises(ValueError):
        module.preview_geometry(plan)


def _source(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg required for the real CPU thumbnail check")
    path = tmp_path / "moving source.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-n", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24",
        "-frames:v", "39", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(path)],
        check=True, capture_output=True, timeout=30)
    inspection = inspect_outpaint_source(VideoFromFile(str(path)))
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=64, height=48,
        frame_count=39, aspect="custom", left=8, right=8)
    return path, inspection, plan


def test_real_ffmpeg_selects_exact_frame_without_model_or_source_mutation(tmp_path):
    path, inspection, plan = _source(tmp_path)
    original = path.read_bytes()
    image, report = module.render_source_geometry_preview(inspection, plan, 17)
    # A small test-only whole decode is independent of the production select
    # filter. The production route never materializes this 39-frame RGB batch.
    raw = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-an", "-threads", "1",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True, timeout=30).stdout
    stride = 64 * 48 * 3
    expected = raw[17 * stride:18 * stride]
    assert len(raw) == 39 * stride and expected != raw[:stride]
    assert image.crop(tuple(report["source_rect_thumbnail"])).tobytes() == expected
    assert path.read_bytes() == original
    assert not report["preview_is_pixel_preservation_evidence"]
    assert not report["model_called"] and report["frame_index"] == 17


def test_source_change_during_read_returns_no_preview(tmp_path, monkeypatch):
    path, inspection, plan = _source(tmp_path)
    original = module._decode_thumbnail
    def changed(*args, **kwargs):
        rgb = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed")
        return rgb
    monkeypatch.setattr(module, "_decode_thumbnail", changed)
    with pytest.raises(ValueError, match="source file bytes changed"):
        module.render_source_geometry_preview(inspection, plan)


def test_cancel_before_read_does_not_start_child(tmp_path, monkeypatch):
    _, inspection, plan = _source(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("decoder started despite cancellation")
    def cancel():
        raise InterruptedError("cancel")
    monkeypatch.setattr(module, "_decode_thumbnail", forbidden)
    with pytest.raises(InterruptedError):
        module.render_source_geometry_preview(inspection, plan, interrupt_check=cancel)


def test_owned_decoder_is_reaped_on_mid_read_cancel(monkeypatch):
    original = subprocess.Popen
    processes = []
    def delayed(_command, **kwargs):
        process = original([sys.executable, "-I", "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process
    def cancel():
        if processes:
            raise InterruptedError("cancel owned preview")
    monkeypatch.setattr(module.subprocess, "Popen", delayed)
    monkeypatch.setattr(module.shutil, "which", lambda _: sys.executable)
    with pytest.raises(InterruptedError):
        module._decode_thumbnail(Path("unused"), 0, 8, 8, interrupt_check=cancel)
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize("code,match", [
    ("import os; os._exit(55)", "decoder failed"),
    ("import sys; sys.stdout.buffer.write(bytes(191))", "exactly one"),
    ("import sys; sys.stdout.buffer.write(bytes(193))", "thumbnail|exactly one"),
    ("import sys; sys.stderr.write('decoder rejected input')", "decoder rejected input"),
])
def test_real_child_failure_or_partial_frame_never_returns_a_preview(monkeypatch, code, match):
    original = subprocess.Popen
    processes = []
    def faulty(_command, **kwargs):
        process = original([sys.executable, "-I", "-c", code], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(module.subprocess, "Popen", faulty)
    monkeypatch.setattr(module.shutil, "which", lambda _: sys.executable)
    with pytest.raises(RuntimeError, match=match):
        module._decode_thumbnail(Path("unused"), 0, 8, 8)
    assert len(processes) == 1 and processes[0].poll() is not None


def test_preview_timeout_reaps_its_child(monkeypatch):
    original = subprocess.Popen
    processes = []
    def delayed(_command, **kwargs):
        process = original([sys.executable, "-I", "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(module.subprocess, "Popen", delayed)
    monkeypatch.setattr(module.shutil, "which", lambda _: sys.executable)
    with pytest.raises(TimeoutError, match="preview timed out"):
        module._decode_thumbnail(Path("unused"), 0, 8, 8, timeout=0)
    assert len(processes) == 1 and processes[0].poll() is not None


def test_missing_ffmpeg_is_actionable_and_starts_nothing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no executable is available; no process should start")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="FFmpeg is needed"):
        module._decode_thumbnail(Path("unused"), 0, 8, 8)


def test_preview_node_is_model_free_and_uses_native_ui(monkeypatch):
    import h3_audio_t8_pkg.nodes_video_outpaint_preview as node_module
    from PIL import Image

    cls = node_module.MiniMaxH3VideoOutpaintGeometryPreviewT8
    schema = cls.define_schema()
    assert schema.is_output_node and schema.is_experimental
    assert [item.id for item in schema.inputs] == ["plan", "frame_index", "preview_max_edge"]
    handle = {"inspection": {}, "plan": _plan()}
    calls = []
    monkeypatch.setattr(node_module, "render_source_geometry_preview",
        lambda *a, **kw: (Image.new("RGB", (64, 48), (128, 64, 32)), {"model_called": False}))
    monkeypatch.setattr(node_module.ui, "PreviewImage", lambda tensor, **kw: calls.append((tensor, kw)) or {"images": []})
    result = cls.execute(handle, 0, 768)
    assert result.result[0] is handle
    assert result.result[1].shape == (1, 48, 64, 3) and result.result[1].device.type == "cpu"
    assert np.allclose(result.result[1][0, 0, 0].numpy(), [128 / 255, 64 / 255, 32 / 255])
    assert json.loads(result.result[2])["model_called"] is False
    assert calls[0][1]["cls"] is cls and result.ui == {"images": []}
