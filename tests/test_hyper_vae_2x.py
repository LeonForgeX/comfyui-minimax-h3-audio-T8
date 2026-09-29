from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint, hyper_vae_2x
from h3_audio_t8_pkg.nodes_hyper_vae_2x import MiniMaxH3HyperVAE2xLoaderEXPT8


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "examples/workflows/58-hypervae-2x/2026-09-27_H3_HyperVAE_2x_Same_Latent_5s_Compare_EXP.json"


def test_header_requires_h3_packed_channels_and_2x_metadata(tmp_path, monkeypatch):
    class Slice:
        def __init__(self, shape):
            self.shape = shape

        def get_shape(self):
            return self.shape

    class Weights:
        def __init__(self, packed=12, factor=2):
            self.shapes = {
                "decoder.proj_out.weight": (packed * 1024, 2048),
                "decoder.proj_out.bias": (packed * 1024,),
                "latents_mean": (24,), "latents_std": (24,),
                "encoder.down.5.block.0.conv1.weight": (1,),
            }
            self.factor = factor

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return self.shapes.keys()

        def get_slice(self, key):
            return Slice(self.shapes[key])

        def metadata(self):
            return {
                "minimax_h3_x2_adapter": json.dumps({
                    "version": 1, "packed_output_channels": 12,
                    "pixel_shuffle_factor": self.factor, "encoder_unchanged": True,
                }),
                "minimax_h3_video_vae": json.dumps({
                    "packed_decoder_output_channels": 12,
                    "decoder_pixel_shuffle_factor": self.factor,
                }),
            }

    path = tmp_path / "hyper.safetensors"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(hyper_vae_2x, "safe_open", lambda *_args, **_kwargs: Weights())
    assert hyper_vae_2x.inspect_hyper_vae_2x(path)["decode_scale_from_latent"] == 32
    monkeypatch.setattr(hyper_vae_2x, "safe_open", lambda *_args, **_kwargs: Weights(packed=3))
    with pytest.raises(ValueError, match="形状不匹配"):
        hyper_vae_2x.inspect_hyper_vae_2x(path)
    monkeypatch.setattr(hyper_vae_2x, "safe_open", lambda *_args, **_kwargs: Weights(factor=1))
    with pytest.raises(ValueError, match="元数据"):
        hyper_vae_2x.inspect_hyper_vae_2x(path)


def test_pixel_shuffle_phase_order_and_geometry():
    packed = torch.arange(24, dtype=torch.float32).reshape(1, 2, 1, 1, 12)
    rgb = hyper_vae_2x.pixel_shuffle_h3_images(packed)
    assert rgb.shape == (1, 2, 2, 2, 3)
    assert torch.equal(rgb[0, 0, :, :, 0], torch.tensor([[0., 1.], [2., 3.]]))
    assert torch.equal(rgb[0, 0, :, :, 1], torch.tensor([[4., 5.], [6., 7.]]))
    assert torch.equal(rgb[0, 1, :, :, 2], torch.tensor([[20., 21.], [22., 23.]]))
    with pytest.raises(ValueError, match="12"):
        hyper_vae_2x.pixel_shuffle_h3_images(torch.empty(1, 1, 1, 1, 3))


def test_packed_decoder_stats_leave_native_encoder_stats_unchanged():
    class Inner:
        pixel_mean = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1, 1)
        pixel_std = torch.ones(1, 3, 1, 1, 1)

    inner = Inner()
    output = hyper_vae_2x._finalize_hyper_pixels(inner, torch.zeros(1, 12, 1, 1, 1))
    assert torch.allclose(output.flatten(), torch.tensor([0.1] * 4 + [0.2] * 4 + [0.3] * 4))
    assert inner.pixel_mean.shape == (1, 3, 1, 1, 1)


def test_node_schema_and_append_only_registration():
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    assert classes[-1] is MiniMaxH3HyperVAE2xLoaderEXPT8
    schema = MiniMaxH3HyperVAE2xLoaderEXPT8.define_schema().get_v1_info(MiniMaxH3HyperVAE2xLoaderEXPT8)
    assert schema.output == ["VAE", "STRING"]


def test_absolute_path_overrides_dropdown(monkeypatch, tmp_path):
    path = tmp_path / "hyper.safetensors"
    path.write_bytes(b"fixture")
    captured = []
    monkeypatch.setattr(hyper_vae_2x, "load_hyper_vae_2x", lambda value: (captured.append(value) or object(), {"status": "ok"}))
    result = MiniMaxH3HyperVAE2xLoaderEXPT8.execute("minimax_h3_video_vae_fp16.safetensors", str(path))
    assert captured == [path]
    assert json.loads(result.result[1])["status"] == "ok"
    with pytest.raises(ValueError, match="绝对路径"):
        MiniMaxH3HyperVAE2xLoaderEXPT8.execute("填写绝对路径", "relative.safetensors")


def test_vae_directory_dropdown_loads_selected_file(monkeypatch, tmp_path):
    import folder_paths

    selected = "hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors"
    path = tmp_path / "models" / "vae" / selected
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda category: [selected] if category == "vae" else [])
    resolved = []

    def get_full_path(category, name):
        resolved.append((category, name))
        return str(path)

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", get_full_path)
    loaded = []
    monkeypatch.setattr(hyper_vae_2x, "load_hyper_vae_2x", lambda value: (loaded.append(value) or object(), {"status": "ok"}))
    schema = MiniMaxH3HyperVAE2xLoaderEXPT8.define_schema().get_v1_info(MiniMaxH3HyperVAE2xLoaderEXPT8)
    assert selected in str(schema.input)
    result = MiniMaxH3HyperVAE2xLoaderEXPT8.execute(selected, "")
    assert resolved == [("vae", selected)]
    assert loaded == [path]
    assert json.loads(result.result[1])["status"] == "ok"


def test_saved_canvas_graph_is_single_sample_same_latent_five_seconds():
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    assert len(nodes) == 22
    assert sum(node["type"] == "SamplerCustomAdvanced" for node in nodes.values()) == 1
    assert nodes[17]["type"] == "MiniMaxH3HyperVAE2xLoaderEXPT8"
    assert nodes[17]["widgets_values"] == [
        "hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors", ""
    ]
    assert nodes[6]["widgets_values"][1] == 5.0
    assert links[14][1:5] == [6, 5, 11, 1]
    assert links[29][1:5] == [6, 5, 19, 1]
    assert nodes[4]["widgets_values"][1:3] == [512, 288]
    assert links[25][1:5] == [10, 2, 18, 0]
    assert links[26][1:5] == [17, 0, 18, 1]
    assert links[13][1:5] == [10, 1, 11, 2]
    assert links[28][1:5] == [10, 1, 19, 2]
    assert links[31][1:5] == [11, 1, 20, 1]
    assert nodes[12]["type"] == nodes[20]["type"] == "MiniMaxH3SafeAVSaveT8Advanced"
