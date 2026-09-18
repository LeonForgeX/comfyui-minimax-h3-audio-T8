"""Real tiny CPU conversions, interruption and immutable-source recovery."""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg import meridian_convert_job as job_module
from h3_audio_t8_pkg.meridian_checkpoint_io import file_sha, part_header
from h3_audio_t8_pkg.meridian_conversion import build_rules, expected_lora_shapes


@pytest.fixture
def conversion(tmp_path):
    source = tmp_path / "source"
    (source / "transformer").mkdir(parents=True)
    (source / "lora").mkdir()
    cfg = dict(
        hidden_size=16,
        num_layers=1,
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
    generator = torch.Generator().manual_seed(56)
    teacher = {
        name: torch.randn(shape, generator=generator).to(torch.bfloat16)
        for rule in build_rules(cfg)
        for name, shape in zip(rule.sources, rule.shapes)
    }
    teacher["proj_in.weight"] = teacher["proj_in.weight"].float()
    teacher["proj_out.weight"] = teacher["proj_out.weight"].float()
    shard = "teacher.safetensors"
    save_file(teacher, source / "transformer" / shard)
    adapter = {}
    for prefix, (rows, cols) in expected_lora_shapes(cfg).items():
        adapter[prefix + ".lora_A.weight"] = (
            torch.randn(2, cols, generator=generator) * 0.02
        )
        adapter[prefix + ".lora_B.weight"] = (
            torch.randn(rows, 2, generator=generator) * 0.02
        )
    save_file(
        adapter,
        source / "lora/pytorch_lora_weights.safetensors",
        metadata={
            "lora_adapter_metadata": json.dumps(
                dict(peft_type="LORA", r=2, lora_alpha=2, bias="none")
            )
        },
    )
    (source / "transformer/config.json").write_text(json.dumps(cfg), encoding="utf8")
    (source / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in teacher}}), encoding="utf8"
    )
    manifest = dict(
        schema="t8-meridian-source-v1",
        repo_id="Viggle/Meridian",
        revision="a" * 40,
        files=[
            dict(
                path=p.relative_to(source).as_posix(),
                bytes=p.stat().st_size,
                sha256=file_sha(p),
            )
            for p in sorted(source.rglob("*"))
            if p.is_file()
        ],
    )
    return dict(
        source=source,
        job=tmp_path / "job",
        destination=tmp_path / "out/model.safetensors",
        manifest=manifest,
    )


def test_actual_conversion_and_idempotent_verified_resume(conversion):
    result = job_module.run_conversion(**conversion, row_chunk=7)
    assert result["quantized_layers"] == 5
    assert not result["cuda_initialized"] and not result["full_model_GPU_ran"]
    assert not result["human_qualified"]
    output = load_file(conversion["destination"])
    assert output["blocks.0.attn.qkv_proj.weight"].dtype == torch.int8
    assert output["video_patch_proj.weight"].dtype == torch.float32
    assert output["token_refiner.blocks.0.attn.qkv_proj.weight"].dtype == torch.bfloat16
    assert part_header(conversion["destination"])["metadata"]["dmd_merged"] == "true"
    # First projection must include the DMD despite remaining high precision.
    teacher = load_file(conversion["source"] / "transformer/teacher.safetensors")
    adapter = load_file(conversion["source"] / "lora/pytorch_lora_weights.safetensors")
    expected = (
        teacher["proj_in.weight"]
        + adapter["proj_in.lora_B.weight"] @ adapter["proj_in.lora_A.weight"]
    )
    torch.testing.assert_close(output["video_patch_proj.weight"], expected)
    stat = conversion["destination"].stat()
    stages = []
    assert (
        job_module.run_conversion(**conversion, row_chunk=7, progress=stages.append)
        == result
    )
    assert conversion["destination"].stat().st_mtime_ns == stat.st_mtime_ns
    assert (
        sum(s["stage"] == "reused_verified_layer" for s in stages)
        == result["completed_layers"]
    )
    assert any(
        s["stage"] == "verifying_source" and s["file"].endswith("teacher.safetensors")
        for s in stages
    )


def interrupted(conversion):
    stop = [False]

    def progress(event):
        if event["stage"] == "converting_layer" and event["index"] == 4:
            stop[0] = True

    with pytest.raises(InterruptedError):
        job_module.run_conversion(
            **conversion, progress=progress, cancelled=lambda: stop[0]
        )
    assert not conversion["destination"].exists()
    return json.loads((conversion["job"] / "state.json").read_text())


def test_cancel_resume_preserves_prior_completed_files(conversion):
    state = interrupted(conversion)
    assert len(state["completed"]) == 4
    parts = {
        p: ((conversion["job"] / p).stat().st_mtime_ns, file_sha(conversion["job"] / p))
        for p in (item["file"] for item in state["completed"].values())
    }
    result = job_module.run_conversion(**conversion)
    assert result["completed_layers"] > 4
    for p, expected in parts.items():
        path = conversion["job"] / p
        assert (path.stat().st_mtime_ns, file_sha(path)) == expected


@pytest.mark.parametrize("point", ["before_link", "after_link", "result"])
def test_crash_at_publication_recovers_without_reconversion(
    conversion, monkeypatch, point
):
    original_json = job_module.atomic_json
    original_link = job_module.os.link

    def crash_json(path, value, **kwargs):
        if Path(path).name == "result.json":
            raise OSError("simulated result crash")
        return original_json(path, value, **kwargs)

    def crash_link(source, destination):
        if Path(destination) == conversion["destination"]:
            if point == "after_link":
                original_link(source, destination)
            raise OSError("simulated publication crash")
        return original_link(source, destination)

    with monkeypatch.context() as patch:
        if point == "result":
            patch.setattr(job_module, "atomic_json", crash_json)
        else:
            patch.setattr(job_module.os, "link", crash_link)
        with pytest.raises(OSError, match="simulated"):
            job_module.run_conversion(**conversion)
    assert (conversion["job"] / "publication.json").is_file()
    stages = []
    result = job_module.run_conversion(**conversion, progress=stages.append)
    assert (
        sum(e["stage"] == "reused_verified_layer" for e in stages)
        == result["completed_layers"]
    )
    assert not any(e["stage"] == "converting_layer" for e in stages)


@pytest.mark.parametrize("kind", ["bytes", "inventory", "metadata"])
def test_invalid_resumed_part_refused(conversion, kind):
    state = interrupted(conversion)
    name, item = next(iter(state["completed"].items()))
    path = conversion["job"] / item["file"]
    if kind == "bytes":
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            b = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([b[0] ^ 1]))
    else:
        tensors = load_file(path)
        metadata = part_header(path)["metadata"]
        if kind == "inventory":
            tensors = {"impostor": next(iter(tensors.values()))}
        else:
            metadata["target"] = "another-target"
        save_file(tensors, path, metadata=metadata)
        item["sha256"] = file_sha(path)
        item["keys"] = list(tensors)
        (conversion["job"] / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Saved part"):
        job_module.run_conversion(**conversion)


def test_changed_recipe_refuses_resume(conversion):
    interrupted(conversion)
    with pytest.raises(ValueError, match="identity changed"):
        job_module.run_conversion(**conversion, row_chunk=17)


def test_complete_output_does_not_skip_source_revalidation(conversion):
    job_module.run_conversion(**conversion)
    source = conversion["source"] / "transformer/teacher.safetensors"
    with source.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x91")
    with pytest.raises(ValueError, match="Source SHA"):
        job_module.run_conversion(**conversion)


def test_missing_source_retains_resume_not_success(conversion):
    state = interrupted(conversion)
    path = conversion["source"] / "transformer/teacher.safetensors"
    moved = path.with_suffix(".incomplete")
    path.rename(moved)
    with pytest.raises(job_module.SourcePending):
        job_module.run_conversion(**conversion)
    assert not conversion["destination"].exists()
    assert (
        json.loads((conversion["job"] / "state.json").read_text())["completed"]
        == state["completed"]
    )
    moved.rename(path)
    job_module.run_conversion(**conversion)


def test_unowned_existing_output_untouched(conversion):
    conversion["destination"].parent.mkdir()
    conversion["destination"].write_bytes(b"unrelated user model")
    with pytest.raises(FileExistsError, match="publication receipt"):
        job_module.run_conversion(**conversion)
    assert conversion["destination"].read_bytes() == b"unrelated user model"


@pytest.mark.parametrize("problem", ["extra", "shape", "dtype"])
def test_even_hashed_teacher_must_match_every_indexed_tensor(conversion, problem):
    path = conversion["source"] / "transformer/teacher.safetensors"
    tensors = load_file(path)
    if problem == "extra":
        tensors["unconsumed"] = torch.ones(1)
    elif problem == "shape":
        tensors["proj_in.weight"] = torch.ones(1)
    else:
        tensors["proj_in.weight"] = tensors["proj_in.weight"].to(torch.float16)
    save_file(tensors, path)
    item = next(
        f
        for f in conversion["manifest"]["files"]
        if f["path"].endswith("teacher.safetensors")
    )
    item.update(bytes=path.stat().st_size, sha256=file_sha(path))
    with pytest.raises(ValueError, match="teacher shard|Teacher header"):
        job_module.run_conversion(**conversion)
    assert not conversion["destination"].exists()


@pytest.mark.parametrize("path", ["../x", "/root", "C:/x", "a\\b", "a//b", "."])
def test_manifest_path_rejected(conversion, path):
    manifest = copy.deepcopy(conversion["manifest"])
    manifest["files"][0]["path"] = path
    with pytest.raises(ValueError, match="relative paths"):
        job_module.validate_manifest(manifest)


def test_conversion_requires_cpu_visibility(conversion, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CPU-only"):
        job_module.run_conversion(**conversion)
