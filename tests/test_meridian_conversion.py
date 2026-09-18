"""Meridian conversion math: no model download and no GPU use."""

from dataclasses import replace

import pytest
import torch

from h3_audio_t8_pkg.meridian_conversion import (
    build_rules,
    convrot_group,
    derived_rope,
    expected_lora_shapes,
    encode_native_tensor,
    materialize_rule,
    validate_adapter,
    validate_index,
)


@pytest.fixture
def config():
    return dict(
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        time_embed_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=16,
        text_dim=12,
        in_channels=3,
        audio_in_channels=4,
        rope_freq_dim=2,
        rope_theta=10000.0,
        patch_size=[1, 2, 2],
    )


def test_exhaustive_full_size_plan():
    cfg = dict(
        hidden_size=5376,
        num_layers=50,
        num_refiner_layers=2,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_dim=14336,
        time_embed_dim=2688,
        freq_dim=256,
        time_embed_hidden_dim=5376,
        text_dim=5120,
        in_channels=24,
        audio_in_channels=32,
        rope_freq_dim=16,
        patch_size=[1, 2, 2],
    )
    rules = build_rules(cfg)
    assert len(rules) == 534
    names = [name for rule in rules for name in rule.sources]
    assert len(names) == len(set(names)) == 638
    assert len({rule.target for rule in rules}) == 534
    assert sum(rule.quantize for rule in rules) == 250
    assert len(expected_lora_shapes(cfg)) == 302
    for rule in rules:
        if rule.quantize:
            assert rule.target.startswith("blocks.")
            assert convrot_group(rule.shapes[0][1]) in (64, 256)


def test_index_requires_every_source_once(config):
    index = {
        name: "shard.safetensors"
        for rule in build_rules(config)
        for name in rule.sources
    }
    assert validate_index(config, index) == build_rules(config)
    with pytest.raises(ValueError, match="extra"):
        validate_index(config, dict(index, unknown="shard"))
    index.pop(next(iter(index)))
    with pytest.raises(ValueError, match="missing"):
        validate_index(config, index)


@pytest.mark.parametrize(
    "field,value",
    [("num_layers", True), ("hidden_size", -1), ("patch_size", [1, 0, 2])],
)
def test_bad_architecture(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        build_rules(config)


@pytest.fixture
def adapter(config):
    specs = {}
    for name, (rows, cols) in expected_lora_shapes(config).items():
        specs[name + ".lora_A.weight"] = dict(shape=[4, cols], dtype="F32")
        specs[name + ".lora_B.weight"] = dict(shape=[rows, 4], dtype="F32")
    return dict(peft_type="LORA", r=4, lora_alpha=8, bias="none"), specs


def test_adapter_scale_and_high_precision_targets(config, adapter):
    metadata, specs = adapter
    assert validate_adapter(config, metadata, specs) == 2
    assert "proj_in" in expected_lora_shapes(config)
    assert "proj_out" in expected_lora_shapes(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("use_dora", True),
        ("use_rslora", True),
        ("rank_pattern", {"x": 1}),
        ("fan_in_fan_out", True),
        ("lora_alpha", float("nan")),
        ("r", True),
        ("bias", "all"),
    ],
)
def test_adapter_other_mathematics_refused(config, adapter, key, value):
    metadata, specs = adapter
    metadata[key] = value
    with pytest.raises(ValueError):
        validate_adapter(config, metadata, specs)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype"])
def test_bad_adapter_tensor_contract(config, adapter, change):
    metadata, specs = adapter
    name = next(iter(specs))
    if change == "missing":
        del specs[name]
    elif change == "extra":
        specs["unknown"] = dict(shape=[1], dtype="F32")
    else:
        specs[name][change] = [1, 2] if change == "shape" else "BF16"
    with pytest.raises(ValueError):
        validate_adapter(config, metadata, specs)


@pytest.mark.parametrize(
    "target",
    [
        "video_patch_proj.weight",
        "final_layer.video_out.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.1.mlp.fc1.weight",
        "token_refiner.blocks.0.mlp.fc1.weight",
    ],
)
def test_real_dense_low_rank_math_before_reorder(config, target):
    rule = next(r for r in build_rules(config) if r.target == target)
    gen = torch.Generator().manual_seed(21)
    source = {
        name: torch.randn(shape, generator=gen)
        for name, shape in zip(rule.sources, rule.shapes)
    }
    original = {k: v.clone() for k, v in source.items()}
    targets = expected_lora_shapes(config)
    lora, expected = {}, []
    for name, shape in zip(rule.sources, rule.shapes):
        prefix = name[:-7]
        merged = source[name]
        if prefix in targets:
            a = torch.randn(3, shape[1], generator=gen)
            b = torch.randn(shape[0], 3, generator=gen)
            lora[prefix + ".lora_A.weight"] = a
            lora[prefix + ".lora_B.weight"] = b
            merged = merged + 0.5 * (b @ a)
        expected.append(merged)
    actual, dtype = materialize_rule(
        rule, source.__getitem__, lora.__getitem__, targets, 0.5
    )
    assert dtype == torch.float32
    if rule.operation == "concat_qkv":
        wanted = torch.cat(expected)
    elif rule.operation == "swap_swiglu":
        left, right = expected[0].chunk(2)
        wanted = torch.cat((right, left))
        x = torch.randn(5, wanted.shape[1], generator=gen)
        pre = x @ expected[0].T
        post = x @ actual.T
        value, gate = pre.chunk(2, dim=-1)
        postgate, postvalue = post.chunk(2, dim=-1)
        torch.testing.assert_close(
            value * torch.nn.functional.silu(gate),
            torch.nn.functional.silu(postgate) * postvalue,
        )
    else:
        wanted = expected[0]
    torch.testing.assert_close(actual, wanted)
    for key in source:
        assert torch.equal(original[key], source[key])
    assert not torch.cuda.is_initialized()


def test_bf16_source_merge_stays_float32(config):
    rule = next(r for r in build_rules(config) if r.target == "blocks.0.mlp.fc2.weight")
    source = torch.ones(rule.shapes[0], dtype=torch.bfloat16)
    prefix = rule.sources[0][:-7]
    adapter = {
        prefix + ".lora_A.weight": torch.full((1, source.shape[1]), 0.001),
        prefix + ".lora_B.weight": torch.ones(source.shape[0], 1),
    }
    actual, dtype = materialize_rule(
        rule, lambda _: source, adapter.__getitem__, {prefix}, 1.0
    )
    assert dtype == torch.bfloat16 and actual.dtype == torch.float32
    assert actual[0, 0] > 1


def test_nonfinite_and_bad_operation_rejected(config):
    rule = next(r for r in build_rules(config) if r.target == "video_patch_proj.weight")
    tensor = torch.zeros(rule.shapes[0])
    tensor[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        materialize_rule(rule, lambda _: tensor, None, set(), 1.0)
    tensor.zero_()
    with pytest.raises(ValueError, match="operation"):
        materialize_rule(
            replace(rule, operation="unknown"), lambda _: tensor, None, set(), 1.0
        )


@pytest.mark.parametrize(
    "width,wanted", [(5376, 256), (7168, 256), (2688, 64), (14336, 256), (16, 16)]
)
def test_per_matrix_group(width, wanted):
    assert convrot_group(width) == wanted


@pytest.mark.parametrize("width", [0, True, 15, -64])
def test_invalid_group(width):
    with pytest.raises(ValueError):
        convrot_group(width)


def test_rope_is_derived_float32_not_permuted(config):
    actual = derived_rope(config)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.01]))


@pytest.mark.parametrize("chunk", [1, 7, 256])
def test_native_ck_quantization_chunk_equivalence_and_file_reload(
    config, tmp_path, chunk
):
    import json
    from safetensors.torch import save_file, load_file
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    rule = next(r for r in build_rules(config) if r.target == "blocks.0.mlp.fc2.weight")
    value = torch.randn(rule.shapes[0], generator=torch.Generator().manual_seed(52))
    stored, report = encode_native_tensor(rule, value, torch.bfloat16, row_chunk=chunk)
    reference, params = TensorWiseINT8Layout.quantize(
        value,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=convrot_group(value.shape[1]),
    )
    assert torch.equal(reference, stored[rule.target])
    # CPU BLAS may select a different reduction for1/7/all rows. Verify both
    # scales against an independent FP64 Hadamard oracle with a derived dot-
    # product rounding bound, not an arbitrary broad quantization tolerance.
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    group = convrot_group(value.shape[1])
    hadamard = _build_hadamard(group, device=value.device, dtype=torch.float64)
    blocks = value.double().reshape(value.shape[0], -1, group)
    rotated = blocks @ hadamard
    oracle = rotated.abs().reshape(value.shape[0], -1).amax(dim=1, keepdim=True) / 127
    unit = torch.finfo(torch.float32).eps / 2
    gamma = (group * unit) / (1 - group * unit)
    bound = (
        (blocks.abs() @ hadamard.abs())
        .reshape(value.shape[0], -1)
        .amax(dim=1, keepdim=True)
        * gamma
        / 127
    )
    bound += oracle * 2 * unit
    for scales in (params.scale, stored[rule.target + "_scale"]):
        assert ((scales.double() - oracle).abs() <= bound).all()
    assert report["relative_l2"] < 0.025
    path = tmp_path / "small-convrot.safetensors"
    save_file(stored, path)
    recovered = load_file(path)
    assert all(torch.equal(t, recovered[k]) for k, t in stored.items())
    configuration = json.loads(
        bytes(recovered["blocks.0.mlp.fc2.comfy_quant"].tolist())
    )
    assert configuration == dict(
        format="int8_tensorwise", convrot=True, convrot_groupsize=16
    )
    assert not torch.cuda.is_initialized()


def test_high_precision_projection_keeps_merged_dmd(config):
    rule = next(
        r for r in build_rules(config) if r.target == "final_layer.video_out.weight"
    )
    value = torch.full(rule.shapes[0], 1.001, dtype=torch.float32)
    stored, report = encode_native_tensor(rule, value, torch.float32)
    assert not report["quantized"] and list(stored) == [rule.target]
    assert torch.equal(stored[rule.target], value)


def test_quantization_cancellation_returns_no_partial_result(config):
    rule = next(r for r in build_rules(config) if r.target == "blocks.0.mlp.fc2.weight")
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) == 3

    with pytest.raises(InterruptedError):
        encode_native_tensor(
            rule,
            torch.ones(rule.shapes[0]),
            torch.bfloat16,
            row_chunk=1,
            cancelled=cancelled,
        )
    assert len(calls) == 3
