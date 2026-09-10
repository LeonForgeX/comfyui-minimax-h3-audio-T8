from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.trt_vdn_probe import recipe, report_nodes, sample, verify_reports
from tools.run_progressive_pilot import RESEARCH, instrument_recipe
from tools.trt_latent_capture import capture_recipe


def fixed_graph():
    return instrument_recipe(json.loads((RESEARCH / "pilot-api-drafts/T2VA_native8.prompt.json").read_text()))


def test_vdn_recipe_preserves_original_and_captures_final_twopass_output():
    original = fixed_graph()
    before = deepcopy(original)
    graph = capture_recipe(recipe(original))
    assert original == before
    assert graph["6"]["inputs"]["width"] == 512 and graph["6"]["inputs"]["height"] == 256
    assert graph["60"]["inputs"]["scale_by"] == 2
    assert graph["63"]["inputs"]["refine_steps"] == 4
    assert graph["105"]["inputs"]["av_latent"] == ["67", 0]
    assert graph["67"]["inputs"]["second_pass_output"] == ["66", 0]
    assert graph["11"]["inputs"]["video_vae"] == ["1", 0]
    assert graph["5"]["inputs"]["verify_hashes"] is True
    assert not any("LoRA" in node["class_type"] for node in graph.values())
    assert report_nodes(graph)["composition"] == "103" and report_nodes(graph)["refine"] == "111"
    assert report_nodes(original)["lora"] == "103"
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


@pytest.mark.parametrize("steps,broken", [(4, False), (8, False), (4, True)])
def test_vdn_observer_counts_every_block_and_always_removes_hooks(monkeypatch, steps, broken):
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    from comfy.nested_tensor import NestedTensor
    blocks = [SimpleNamespace(softmax_gate=torch.nn.Identity()) for _ in range(50)]
    branches = {"branch": [SimpleNamespace(model=SimpleNamespace(blocks=blocks))]}
    clones = []
    class Model:
        additional_models = branches
        model_options = {}
        wrapper = None
        def get_attachment(self, key):
            return {"status": "configured", "stage": "stage_dmd_8nfe"}
        def clone(self):
            clone = Model()
            clones.append(clone)
            return clone
        def add_wrapper_with_key(self, kind, key, function):
            self.wrapper = function
        def remove_wrappers_with_key(self, kind, key):
            self.wrapper = None
    def run(noise, guider, sampler, sigmas, latent):
        def forward():
            for block in blocks[:-1] if broken else blocks:
                block.softmax_gate(torch.zeros(1))
        for _ in range(len(sigmas) - 1):
            guider.wrapper(forward)
        return SimpleNamespace(result=[latent])
    monkeypatch.setattr(BasicGuider, "execute", lambda model, positive: SimpleNamespace(result=[model]))
    monkeypatch.setattr(RandomNoise, "execute", lambda seed: SimpleNamespace(result=[seed]))
    monkeypatch.setattr(SamplerCustomAdvanced, "execute", run)
    vdn = SimpleNamespace(ATTACHMENT_KEY="attachment", ADDITIONAL_MODEL_KEY="branch", validate_vdn_runtime_options=lambda opts: None)
    model = Model()
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 9)))}
    if broken:
        with pytest.raises(ValueError, match="every actual VDN block"):
            sample(model, [], latent, object(), torch.arange(steps + 1), 42, vdn)
    else:
        result, report = sample(model, [], latent, object(), torch.arange(steps + 1), 42, vdn)
        assert result is latent and report["branch_gate_calls"] == [steps] * 50
    assert model.wrapper is None and clones[0].wrapper is None
    assert all(not block.softmax_gate._forward_hooks for block in blocks)


def reports_for(graph):
    return {"sampler": {"status": "actual_vdn_execution_pass_quality_unreviewed", "actual_network_forwards": 8,
                        "branch_gate_calls": [8] * 50, "cfg": 1, "seed": graph["10"]["inputs"]["seed"]},
            "refine": {"status": "actual_vdn_execution_pass_quality_unreviewed", "actual_network_forwards": 4,
                       "branch_gate_calls": [4] * 50, "cfg": 1, "seed": graph["66"]["inputs"]["seed"]},
            "composition": {"status": "configured", "stage": "stage_dmd_8nfe", "main_block_count": 50,
                "adapters": [{"name": name, "patch_targets": count, "applied_targets": count,
                              "shape_validation": {"checked_targets": count, "all_shapes_exact": True}}
                             for name, count in (("default", 104), ("turbo", 259))]},
            "refine_plan": {"first_pass_nfe": 8, "refine_nfe": 4, "additional_turbo_lora": False,
                            "audio_matches_first_pass": True, "high_video_shape": [1, 24, 22, 32, 64]},
            "audio_lock": {"status": "locked_audio_replaced_exact", "audio_relocked_exact": True,
                           "audio_within_tolerance": True, "video_finite": True, "audio_finite": True}}


@pytest.mark.parametrize("damage", [None, "branch", "adapter", "audio", "extra_lora"])
def test_vdn_reports_require_actual_calls_shapes_and_audio(damage):
    graph = recipe(fixed_graph())
    reports = reports_for(graph)
    if damage == "branch":
        reports["refine"]["branch_gate_calls"][-1] = 3
    elif damage == "adapter":
        reports["composition"]["adapters"][1]["applied_targets"] = 258
    elif damage == "audio":
        reports["audio_lock"]["audio_within_tolerance"] = False
    elif damage == "extra_lora":
        graph["123"] = {"class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced", "inputs": {}}
    if damage:
        with pytest.raises(ValueError):
            verify_reports(graph, reports)
    else:
        verify_reports(graph, reports)
