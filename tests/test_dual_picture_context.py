import json
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import long_video_dual_picture_context as picture
from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import (
    _write_effects_audit,
)
from test_long_video_dual_model_runner import rig  # noqa: F401


def test_reencode_is_video_only_and_keeps_the_reviewed_operation_order():
    old = torch.zeros(1, 24, 12, 2, 4)
    audio = torch.randn(1, 32, 2, 20)
    context = {
        "video_tail": old,
        "audio_tail": audio,
        "metadata": {"audio_overhang": 1 / 3},
    }
    frames = torch.zeros(39, 64, 128, 3)
    resized = torch.zeros(39, 32, 64, 3)
    calls = []

    def resize(value, width, height):
        assert value is frames and (width, height) == (64, 32)
        calls.append("resize")
        return resized

    def encode(value):
        assert value is resized and torch.is_inference_mode_enabled()
        calls.append("encode")
        return torch.ones_like(old, dtype=torch.float64)

    result, report = picture.reencode_tail(
        context, frames, SimpleNamespace(encode=encode), resize, 64, 32
    )
    assert calls == ["resize", "encode"]
    assert result["audio_tail"] is audio
    assert result["metadata"]["audio_overhang"] == 1 / 3
    assert result["video_tail"].dtype == old.dtype
    assert torch.all(result["video_tail"] == 1) and torch.count_nonzero(old) == 0
    assert "accepted_picture_context" not in context["metadata"]
    assert report["additional_sampling_nfe"] == 0 and report["audio_tensor_preserved"]


@pytest.mark.parametrize("fault", ["frame_count", "geometry", "nan"])
def test_bad_encode_rejected(fault):
    old = torch.zeros(1, 24, 12, 2, 4)
    encoded = old.clone() if fault != "geometry" else old[:, :, :7]
    if fault == "nan":
        encoded.fill_(float("nan"))
    with pytest.raises(ValueError):
        picture.reencode_tail(
            {"video_tail": old, "metadata": {}},
            torch.zeros(22 if fault == "frame_count" else 39, 32, 64, 3),
            SimpleNamespace(encode=lambda value: encoded),
            lambda value, *args: value,
            64,
            32,
        )


def source_fixture(tmp_path):
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"fixture-media")
    path = tmp_path / "candidates/segment_00000/parent/candidate.json"
    path.parent.mkdir(parents=True)
    info = {
        "candidate_id": "parent",
        "chain_id": "test",
        "index": 0,
        "video_path": "movie.mp4",
        "video_sha256": picture._sha256_file(media),
        "frame_count": 124,
    }
    path.write_text(json.dumps(info))
    return path, info, media


@pytest.mark.parametrize(
    "fault", ["none", "hash", "chain", "parent", "index", "path", "short"]
)
def test_predecessor_identity_is_verified(tmp_path, fault):
    path, info, media = source_fixture(tmp_path)
    if fault == "hash":
        info["video_sha256"] = "0" * 64
    if fault == "chain":
        info["chain_id"] = "wrong"
    if fault == "parent":
        info["candidate_id"] = "wrong"
    if fault == "index":
        info["index"] = 2
    if fault == "path":
        info["video_path"] = "../escape.mp4"
    if fault == "short":
        info["frame_count"] = 22
    path.write_text(json.dumps(info))
    if fault == "none":
        actual, report = picture.accepted_source(tmp_path, "parent", 1, "test")
        assert actual == media and report["source_frame_interval"] == [85, 124]
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            picture.accepted_source(tmp_path, "parent", 1, "test")


@pytest.mark.parametrize("resume", [False, True])
def test_runner_changes_only_later_low_and_cache_skips_encode(
    rig, tmp_path, monkeypatch, resume  # noqa: F811
):
    engine, run, calls, fail, first, second = rig
    engine.low_context_source = picture.NAME
    source_checks, prepares = [], []
    source = {"name": picture.NAME, "source_media_sha256": "a" * 64}

    def accepted(*args):
        source_checks.append(args)
        return tmp_path / "movie.mp4", dict(source)

    def prepare(context, *args):
        prepares.append(context)
        return {**context, "video_tail": context["video_tail"] + 7}, {
            "name": picture.NAME,
            "additional_sampling_nfe": 0,
            "audio_tensor_preserved": True,
        }

    monkeypatch.setattr(picture, "accepted_source", accepted)
    monkeypatch.setattr(picture, "prepare_context", prepare)
    first_result = run()
    assert not source_checks and not prepares
    _write_effects_audit(
        str(tmp_path / "candidates/segment_00000/candidate0/candidate.json"),
        {
            "contract_sha256": "job",
            "segment_index": 0,
            "candidate_id": "candidate0",
            "sampling_plan": first_result["sampling_report"],
        },
    )
    high = {"empty": False, "video_tail": torch.full((1, 24, 7, 4, 8), 19.0)}
    start = len(calls)
    if resume:
        fail["high"] = True
        with pytest.raises(RuntimeError, match="second pass failure"):
            run(1, "candidate1", "candidate0", high)
        fail["high"] = False
    result = run(1, "candidate1", "candidate0", high)
    conds = [entry for entry in calls[start:] if entry[0] == "condition"]
    low = [entry[3] for entry in conds if entry[1] is first]
    assert len(low) == len(prepares) == 1
    assert torch.all(low[0]["video_tail"] == 9)
    assert low[0]["audio_tail"] is prepares[0]["audio_tail"]
    assert all(entry[3] is high for entry in conds if entry[1] is second)
    assert (
        result["sampling_report"]["dual_model"]["first_pass"][
            "accepted_picture_context"
        ]["additional_sampling_nfe"]
        == 0
    )
    # No post-sampling bridge: fixture high sampler returns 2 + 2 unchanged.
    assert torch.all(result["sampled"]["samples"].unbind()[0] == 4)
    count = len(calls)
    cached = run(1, "candidate1", "candidate0", high)
    assert cached["sampling_report"]["dual_model"]["high_reused"]
    assert len(prepares) == 1 and len(source_checks) == (3 if resume else 2)
    assert not any(entry[0] == "sample" for entry in calls[count:])
    # A different selected movie cannot match either low/high stage receipt.
    source["source_media_sha256"] = "b" * 64
    changed = run(1, "candidate1", "candidate0", high)
    assert not changed["sampling_report"]["dual_model"]["low_reused"]
    assert not changed["sampling_report"]["dual_model"]["high_reused"]
    assert len(prepares) == 2


def test_schema_option_is_append_only_and_legacy_default():
    from h3_audio_t8_pkg.nodes_long_video_dual_model import (
        MiniMaxH3DualModelLongVideoEXPT8,
    )

    fields = MiniMaxH3DualModelLongVideoEXPT8.define_schema().inputs
    assert [f.id for f in fields[-2:]] == ["video_context_mode", "low_context_source"]
    assert fields[-1].default == picture.LEGACY


def test_unknown_source_rejected_before_model_work():
    with pytest.raises(ValueError, match="Unknown low video context source"):
        runner.DualModelSegmentRunner(
            None,
            None,
            contract={},
            low_width=64,
            low_height=32,
            upscaler_model="fixture",
            low_context_source="guess",
        )


def test_release_workflow_preserves_accepted_recipe_and_current_schema():
    from tools.build_dual_picture_workflow import ROOT, DEST, recipe
    from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
    from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
    from helpers import plugin_widget_map
    frozen = json.loads((ROOT / 'tests/fixtures/dual_picture_accepted_api.json').read_text(encoding='utf-8'))
    graph = recipe()
    permitted = {'chain_id', 'filename_prefix', 'low_context_source'}
    for key in graph:
        if key != '8':
            assert graph[key] == frozen[key]
        else:
            for name, value in graph[key]['inputs'].items():
                if name not in permitted:
                    assert value == frozen[key]['inputs'][name]
    workflow = json.loads(DEST.read_text(encoding='utf-8'))
    nodes = [n for n in workflow['nodes'] if n['type'] != 'MarkdownNote']
    ids = {key: index + 1 for index, key in enumerate(graph)}
    classes = {cls.define_schema().node_id: cls for cls in (
        MiniMaxH3DualModelLongVideoEXPT8, MiniMaxH3PromptRelayPlanT8Advanced,
        MiniMaxH3LoRACompatibilityLoaderT8Advanced)}
    edges = {e[0]: e for e in workflow['links']}
    consumed = set()
    for key, node in zip(graph, nodes):
        assert node['id'] == ids[key] and node['type'] == graph[key]['class_type'] and node['mode'] == 0
        values = plugin_widget_map(node, classes[node['type']]) if node['type'] in classes else None
        for name, value in graph[key]['inputs'].items():
            if isinstance(value, list):
                slot, pin = next((i, p) for i, p in enumerate(node['inputs']) if p['name'] == name)
                assert edges[pin['link']][1:5] == [ids[value[0]], value[1], ids[key], slot]
                consumed.add(pin['link'])
            elif values is not None:
                assert values[name] == value
    assert consumed == set(edges) and len(nodes) == 14
    assert workflow['extra']['accepted_video_sha256'] == '3ff583bc817dd845fa288aaf0b16c2d2be24a6ce282f50c1b37a823edc91ad73'


@pytest.mark.parametrize('fault', ['none', 'fps', 'count', 'changed'])
def test_cpu_decode_uses_exact_rgb24_tail_and_rechecks_source(tmp_path, fault):
    import av
    import numpy as np
    media = tmp_path / 'tiny.mp4'
    with av.open(str(media), 'w') as output:
        stream = output.add_stream('libx264', rate=25 if fault == 'fps' else 24)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for index in range(45):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 4, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    source = {'source_media_sha256': '0' * 64 if fault == 'changed' else picture._sha256_file(media),
              'source_frame_interval': [6, 44 if fault == 'count' else 45]}
    if fault != 'none':
        with pytest.raises(ValueError):
            picture.decode_tail(media, source)
    else:
        actual = picture.decode_tail(media, source)
        with av.open(str(media)) as container:
            expected = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in container.decode(video=0)]
        assert torch.equal(actual, torch.stack(expected[-39:]).float() / 255)
