"""Pinned Meridian CPU conversion. No package registration or GPU inference.

First use --prepare-manifest to obtain immutable Hub identities (no weights are
downloaded by this command). Then use --convert with that manifest. The existing
hf download command owns weight downloads and resumable .incomplete files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types


REPO = "Viggle/Meridian"
REVISION = "2083d059d8544ff7eaaf86966b83e4964a904737"
DMD_SHA = "8c3c331b4ead47a3e8b3b133dd61c288325686d94e2f4f50e949f5fad9b09f0a"
CONFIG = "transformer/config.json"
INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
DMD = "lora/pytorch_lora_weights.safetensors"


def modules():
    # Load only these CPU utilities, never the custom-node __init__/registry.
    name = "_t8_meridian_cpu_conversion"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "h3_t8")]
        sys.modules[name] = package
    from _t8_meridian_cpu_conversion import meridian_convert_job
    from _t8_meridian_cpu_conversion import meridian_checkpoint_io

    return meridian_convert_job, meridian_checkpoint_io


def build_manifest(source, info):
    """Bind LFS SHA256 and actual small Git blobs from a pinned Hub response."""
    if info.sha != REVISION or info.id != REPO:
        raise ValueError("Hub resolved a different source identity")
    siblings = {item.rfilename: item for item in info.siblings}
    if len(siblings) != len(info.siblings):
        raise ValueError("Duplicate Hub source filenames")
    entries = {}
    for name in (CONFIG, INDEX):
        item = siblings[name]
        path = source / name
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if item.size != len(raw) or item.blob_id != blob or item.lfs is not None:
            raise ValueError(f"Local config/index differs from pinned Git blob: {name}")
        entries[name] = dict(
            path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
        )
    cfg = json.loads((source / CONFIG).read_text(encoding="utf8"))
    index = json.loads((source / INDEX).read_text(encoding="utf8"))
    if index.get("metadata", {}).get("total_size") != 66280430080:
        raise ValueError("Pinned Meridian teacher size differs")
    architecture = dict(
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
    if any(cfg.get(k) != v for k, v in architecture.items()):
        raise ValueError("Pinned Meridian architecture differs")
    shards = sorted(set(index["weight_map"].values()))
    if shards != [
        f"diffusion_pytorch_model-{i:05d}-of-00014.safetensors" for i in range(1, 15)
    ]:
        raise ValueError("Expected all14 pinned teacher shards")
    for name in [DMD] + ["transformer/" + s for s in shards]:
        item = siblings[name]
        if item.lfs is None or item.lfs.size != item.size:
            raise ValueError(f"Hub is missing immutable LFS identity: {name}")
        entries[name] = dict(path=name, bytes=item.size, sha256=item.lfs.sha256)
    if entries[DMD]["sha256"] != DMD_SHA:
        raise ValueError("Pinned DMD differs from audited full SHA")
    manifest = dict(
        schema="t8-meridian-source-v1",
        repo_id=REPO,
        revision=REVISION,
        files=[entries[name] for name in sorted(entries)],
    )
    module, _ = modules()
    module.validate_manifest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-manifest", action="store_true")
    mode.add_argument("--convert", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--row-chunk", type=int, default=256)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    module, io = modules()
    if args.prepare_manifest:
        if args.manifest.exists():
            raise FileExistsError(
                "Choose a new manifest path; previous identities are immutable"
            )
        # Installed hf predates `hf models info`; use its own official SDK for
        # read-only metadata, without upgrading the user's runtime or credentials.
        from huggingface_hub import HfApi

        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
        manifest = build_manifest(source, info)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        io.atomic_json(args.manifest, manifest)
        print(
            json.dumps(
                dict(
                    status="pinned_manifest_ready_not_full_local_SHA_verified",
                    path=str(args.manifest.resolve()),
                    files=len(manifest["files"]),
                    total_bytes=sum(f["bytes"] for f in manifest["files"]),
                )
            )
        )
        return
    if args.job is None or args.output is None or not 1 <= args.threads <= 64:
        parser.error("Conversion needs --job, --output and1..64 CPU threads")
    manifest = module.read_json(args.manifest)
    identities = module.validate_manifest(manifest)
    if manifest["revision"] != REVISION or identities[DMD]["sha256"] != DMD_SHA:
        raise ValueError("Production conversion requires the audited pinned source")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    # This is informational; actual allocation failures remain explicit. It is
    # not a GPU-headroom gate and does not stop other user processes.
    for path in (args.job, args.output.parent):
        ancestor = path.resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        print(
            json.dumps(
                dict(
                    stage="storage_inventory",
                    path=str(ancestor),
                    free_bytes=shutil.disk_usage(ancestor).free,
                )
            ),
            flush=True,
        )

    def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    result = module.run_conversion(
        source,
        args.job,
        args.output,
        manifest,
        row_chunk=args.row_chunk,
        progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
