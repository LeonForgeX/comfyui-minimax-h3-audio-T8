import json
import struct
from types import SimpleNamespace

from h3_audio_t8_pkg.h3_weight_diagnostics import (
    describe_h3_payload, inspect_h3_weight_file, lora_mapping_diagnostics,
)


def test_special_runtime_is_structural_not_filename():
    hyperflow = describe_h3_payload(["transformer.endpoint_time_embedder.linear_1.lora_A.weight"])
    assert hyperflow["special_runtime_requirements"][0]["kind"] == "hyperflow"
    pdd = describe_h3_payload(["pdd.final_layer.audio_out.bias"])
    assert pdd["special_runtime_requirements"][0]["kind"] == "pdd"
    unknown = describe_h3_payload(["my_new_adapter.foo"])
    assert unknown["compatibility"] == "not_established_by_header"
    assert unknown["special_runtime_requirements"] == []


def test_merged_acceleration_requires_explicit_declaration():
    assert not describe_h3_payload([], {"modelspec.title": "Turbo merged"})["merged_acceleration"]["declared"]
    declared = describe_h3_payload([], {"merged_loras": '[{"name":"h3_turbo","strength":1}]'})
    assert declared["merged_acceleration"]["declared"]
    assert declared["merged_acceleration"]["verified"] is False


def test_header_read_is_bounded_and_does_not_load_body(tmp_path):
    header = {"__metadata__": {"hyperflow": "true"},
              "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "arbitrary.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"12345678")
    report = inspect_h3_weight_file(path)
    assert report["status"] == "inspected"
    assert report["bytes_read"] == len(raw) + 8
    assert report["tensor_values_loaded"] is False
    assert report["special_runtime_requirements"][0]["kind"] == "hyperflow"
    assert inspect_h3_weight_file(path, maximum_header_bytes=4)["status"] == "unknown"


def test_malformed_missing_and_non_safetensors_are_diagnostics(tmp_path):
    assert inspect_h3_weight_file(tmp_path / "missing")['status'] == "unknown"
    path = tmp_path / "turbo-merged.gguf"
    path.write_bytes(b"GGUF01234567")
    assert inspect_h3_weight_file(path)["status"] == "unknown"


def test_parser_mapping_is_not_claimed_as_execution():
    keys = {"a.lora_A.weight": None, "a.lora_B.weight": None, "a.alpha": None,
            "b.diff_b": None, "newformat.weight": None}
    patch = SimpleNamespace(loaded_keys={"a.lora_A.weight", "a.lora_B.weight"})
    report = lora_mapping_diagnostics(keys, {"a": "a.weight", "b": "b.weight"},
                                      {"a.weight": patch, "b.bias": ("diff", ())}, ["a.weight"])
    assert report["converted_parser_recognized_tensor_count"] == 4
    assert report["converted_unmapped_keys"] == ["newformat.weight"]
    assert report["registered_patch_targets"] == ["a.weight"]
    assert report["registration_is_execution"] is False


def test_tensor_offsets_outside_file_are_not_accepted(tmp_path):
    raw = json.dumps({"x": {"shape": [1], "data_offsets": [0, 80]}}).encode()
    path = tmp_path / "broken.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert inspect_h3_weight_file(path)["status"] == "unknown"


def test_real_native_parser_reports_lora_factors_and_unknown_key():
    import comfy.lora
    import torch

    state = {"a.lora_A.weight": torch.ones(1, 3), "a.lora_B.weight": torch.ones(2, 1),
             "unused.new_kind": torch.zeros(1)}
    key_map = {"a": "layer.weight"}
    patches = comfy.lora.load_lora(state, key_map, log_missing=False)
    report = lora_mapping_diagnostics(state, key_map, patches, patches)
    assert report["converted_parser_recognized_tensor_count"] == 2
    assert report["converted_unmapped_keys"] == ["unused.new_kind"]


def test_loader_report_preserves_unknown_data_as_warning_and_zero_strength(tmp_path, monkeypatch):
    import torch
    from h3_audio_t8_pkg import h3_lora_compat_advanced as compat

    class Model:
        def __init__(self):
            self.model = SimpleNamespace()
            self.attachments = {}

        def clone(self):
            return Model()

        def add_patches(self, patches, strength):
            return list(patches)

        def set_attachments(self, key, value):
            self.attachments[key] = value

    path = tmp_path / "custom.safetensors"
    path.write_bytes(b"test")
    state = {"a.lora_A.weight": torch.ones(1, 3), "a.lora_B.weight": torch.ones(2, 1),
             "custom.new_kind": torch.zeros(1)}
    monkeypatch.setattr(compat.comfy.utils, "load_torch_file", lambda *args, **kwargs: (state, {}))
    monkeypatch.setattr(compat, "build_minimax_h3_lora_key_map", lambda model: ({"a": "layer.weight"}, 0))
    output, raw = compat.load_minimax_h3_lora_model(Model(), path, 0)
    report = json.loads(raw)
    assert report["applied_patch_count"] == 1
    assert report["mapping_diagnostics"]["converted_unmapped_keys"] == ["custom.new_kind"]
    assert any("zero" in warning for warning in report["warnings"])
    assert output.attachments["t8_h3_lora_compat_report"]["full_algorithm_verified"] is False
