"""Inspect the pinned Meridian teacher/DMD and native H3 mapping, CPU only.

Produces a local preparation receipt. Missing shards remain explicitly missing;
this is not a checkpoint conversion or an inference/quality certificate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def inspect(source, core):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "h3_t8"), str(core)]
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import torch
    from safetensors import safe_open
    from meridian_conversion import (
        convrot_group,
        validate_index,
        validate_adapter,
    )

    torch.set_num_threads(2)
    cfg_path = source / "transformer/config.json"
    index_path = source / "transformer/diffusion_pytorch_model.safetensors.index.json"
    adapter_path = source / "lora/pytorch_lora_weights.safetensors"
    config = json.loads(cfg_path.read_text(encoding="utf8"))
    expected = dict(
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
        rope_theta=10000.0,
        patch_size=[1, 2, 2],
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Architecture differs from the audited Meridian release")
    index = json.loads(index_path.read_text(encoding="utf8"))
    if index["metadata"]["total_size"] != 66280430080:
        raise ValueError("Teacher tensor byte count changed")
    rules = validate_index(config, index["weight_map"])
    shards = sorted(set(index["weight_map"].values()))
    if shards != [
        f"diffusion_pytorch_model-{i:05d}-of-00014.safetensors" for i in range(1, 15)
    ]:
        raise ValueError("Unexpected teacher shard inventory")
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["lora_adapter_metadata"])
        specs = {
            key: dict(
                shape=handle.get_slice(key).get_shape(),
                dtype=handle.get_slice(key).get_dtype(),
            )
            for key in handle.keys()
        }
    scale = validate_adapter(config, metadata, specs)
    if metadata["r"] != 128 or scale != 1.0:
        raise ValueError("Released DMD rank/scale changed")
    adapter_sha = digest(adapter_path)
    if (
        adapter_sha
        != "8c3c331b4ead47a3e8b3b133dd61c288325686d94e2f4f50e949f5fad9b09f0a"
    ):
        raise ValueError("Pinned DMD full SHA mismatch")

    # Instantiate only meta tensors: native constructor/state shape proof, no
    # 66GB original model allocation or GPU inference.
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    model = MiniMaxH3Model(
        device="meta", dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init
    )
    native = model.state_dict()
    targets = {rule.target for rule in rules} | {"rope.inv_freq"}
    if set(native) != targets:
        raise ValueError(f"Native Core keys differ: {sorted(set(native) ^ targets)}")
    for rule in rules:
        shape = list(rule.shapes[0])
        if rule.operation == "concat_qkv":
            shape[0] *= 3
        if tuple(native[rule.target].shape) != tuple(shape):
            raise ValueError(f"Native shape mismatch: {rule.target}")
    # Core creates its16-value derived RoPE placeholder on CPU regardless of
    # constructor device. No trained parameter may leave meta during inspection.
    nonmeta = {
        name: list(t.shape) for name, t in native.items() if t.device.type != "meta"
    }
    if (
        nonmeta != {"rope.inv_freq": [16]}
        or native["rope.inv_freq"].device.type != "cpu"
    ):
        raise RuntimeError("Native architecture inspection allocated real weights")
    missing = [name for name in shards if not (source / "transformer" / name).is_file()]
    source_specs = {
        name: shape for r in rules for name, shape in zip(r.sources, r.shapes)
    }
    checked = []
    for filename in shards:
        path = source / "transformer" / filename
        if filename in missing:
            continue
        wanted = {
            key for key, shard in index["weight_map"].items() if shard == filename
        }
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != wanted:
                raise ValueError(f"Shard index/header keys differ: {filename}")
            for key in wanted:
                header = handle.get_slice(key)
                if tuple(header.get_shape()) != source_specs[
                    key
                ] or header.get_dtype() not in ("F32", "BF16"):
                    raise ValueError(f"Teacher shape/dtype changed: {key}")
        checked.append(filename)
    if torch.cuda.is_initialized():
        raise RuntimeError("Inspection initialized CUDA")
    return dict(
        status="CPU_mapping_and_DMD_contract_pass_not_converted",
        upstream="Viggle/Meridian",
        revision="2083d059d8544ff7eaaf86966b83e4964a904737",
        config_sha256=digest(cfg_path),
        index_sha256=digest(index_path),
        dmd_sha256=adapter_sha,
        dmd_rank=128,
        dmd_scale=scale,
        dmd_pairs=len(specs) // 2,
        source_tensor_count=len(index["weight_map"]),
        native_tensor_count=len(native),
        quantized_layers=sum(r.quantize for r in rules),
        quant_groups=sorted(
            {convrot_group(r.shapes[0][1]) for r in rules if r.quantize}
        ),
        cpu_meta_native_shapes_verified=True,
        cuda_initialized=False,
        teacher_shards_header_checked=checked,
        missing_teacher_shards=missing,
        teacher_shards_fully_hashed=False,
        conversion_complete=False,
        model_quality_verified=False,
        high_precision_video_projection_DMD_included=True,
        core_model_sha256=digest(core / "comfy/ldm/minimax/model.py"),
        implementation_sha256=digest(
            Path(__file__).resolve().parents[1] / "h3_t8/meridian_conversion.py"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Inspection receipt already exists; choose a new path")
    result = inspect(args.source.resolve(strict=True), args.core.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
