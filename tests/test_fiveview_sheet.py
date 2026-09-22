from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.fiveview_sheet import (
    FIVE_VIEW_PROTOCOL,
    build_five_view_conditioning,
    decode_five_view_sheet,
)
from h3_audio_t8_pkg.nodes_fiveview_exp import FIVE_VIEW_EXP_NODE_CLASSES
from helpers import FakeClip, FakeVideoVAE


def _args():
    return {
        "clip": FakeClip(),
        "video_vae": FakeVideoVAE(),
        "prompt": "Five consistent camera views of <Picture 1>.",
        "ref_image": torch.zeros((1, 512, 640, 3)),
        "size": 512,
    }


def test_five_view_conditions_exact_author_slot_contract():
    args = _args()
    conditioning, latent, prompt, report_json = build_five_view_conditioning(**args)
    video, audio = latent["samples"].unbind()
    refs = conditioning[0][1]["minimax_refs"]

    assert latent["t8_five_view_protocol"] == FIVE_VIEW_PROTOCOL
    assert video.shape == (1, 24, 5, 32, 32)
    assert audio.shape == (1, 32, 2, 28)
    assert "noise_mask" not in latent
    assert len(refs) == 1
    assert refs[0]["kind"] == "image"
    assert refs[0]["latent"].shape == (1, 24, 1, 28, 36)
    assert prompt == args["prompt"]
    assert args["clip"].tokenize_calls[0][1]["minimax_ref_items"][0]["type"] == "image"
    report = json.loads(report_json)
    assert report["view_count"] == 5
    assert report["required_model"].startswith("MiniMax H3 Ref2VA pruned")


def test_five_view_rejects_missing_ref_tag_invalid_size_and_image_batch():
    args = _args()
    args["prompt"] = "Portrait of a woman."
    with pytest.raises(ValueError, match="<Picture 1>"):
        build_five_view_conditioning(**args)

    args = _args()
    args["size"] = 500
    with pytest.raises(ValueError, match="multiple of 32"):
        build_five_view_conditioning(**args)

    args = _args()
    args["ref_image"] = torch.zeros((2, 512, 512, 3))
    with pytest.raises(ValueError, match="exactly one RGB"):
        build_five_view_conditioning(**args)


class _SlotRecordingVAE(FakeVideoVAE):
    def __init__(self):
        super().__init__()
        self.decode_inputs = []

    def decode(self, latent):
        self.decode_inputs.append(latent.clone())
        value = float(latent[0, 0, 0, 0, 0])
        return torch.full((1, 5, 512, 512, 3), value)


def test_five_view_decodes_five_slots_independently_and_makes_strip():
    _, latent, _, _ = build_five_view_conditioning(**_args())
    video, _ = latent["samples"].unbind()
    for index in range(5):
        video[:, :, index] = index / 5

    vae = _SlotRecordingVAE()
    batch, strip, report_json = decode_five_view_sheet(latent, vae)
    assert batch.shape == (5, 512, 512, 3)
    assert strip.shape == (1, 512, 2560, 3)
    assert len(vae.decode_inputs) == 5
    for index, decode_input in enumerate(vae.decode_inputs):
        assert decode_input.shape == (1, 24, 2, 32, 32)
        assert torch.equal(decode_input[:, :, 0], decode_input[:, :, 1])
        assert batch[index, 0, 0, 0] == pytest.approx(index / 5)
    assert json.loads(report_json)["views"] == 5


def test_five_view_decode_rejects_ordinary_video_and_malformed_slots():
    _, latent, _, _ = build_five_view_conditioning(**_args())
    with pytest.raises(ValueError, match="not ordinary video"):
        decode_five_view_sheet({"samples": latent["samples"]}, FakeVideoVAE())

    latent["t8_five_view_protocol"] = FIVE_VIEW_PROTOCOL
    video, audio = latent["samples"].unbind()
    latent["samples"] = type(latent["samples"])((video[:, :, :4], audio))
    with pytest.raises(ValueError, match="exactly five"):
        decode_five_view_sheet(latent, FakeVideoVAE())


def test_five_view_registration_is_dedicated_exp_append_only_pair():
    assert [node.define_schema().node_id for node in FIVE_VIEW_EXP_NODE_CLASSES] == [
        "MiniMaxH3FiveViewConditioningEXPT8",
        "MiniMaxH3FiveViewDecodeEXPT8",
    ]
    assert all(node.define_schema().is_experimental for node in FIVE_VIEW_EXP_NODE_CLASSES)


def test_five_view_frontend_workflow_matches_executed_api_graph():
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "tests/fixtures/api/five_view_sheet_api.json").read_text(encoding="utf-8"))
    workflow = json.loads((
        root / "examples/workflows/03-image-video-edit/"
        "2026-09-21_H3_Five_View_Character_Sheet_EXP.json"
    ).read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert len(nodes) == len(api) == 13
    assert len(links) == workflow["last_link_id"] == 17
    assert {node["type"] for node in nodes.values()} == {
        item["class_type"] for item in api.values()
    }
    assert "MiniMaxH3StillDecodeT8" not in {node["type"] for node in nodes.values()}
    for node_id, item in api.items():
        frontend = nodes[node_id]
        assert frontend["type"] == item["class_type"]
        for name, value in item["inputs"].items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            front_input = next(inp for inp in frontend["inputs"] if inp["name"] == name)
            link = links[front_input["link"]]
            assert link[1] == int(value[0])
            assert link[2] == value[1]
            assert link[3] == int(node_id)

    assert nodes["3"]["widgets_values"] == [
        "minimax_h3_five_view_512_s1500.safetensors", 1.0
    ]
