"""Inspect or losslessly wrap an H3 Semantic Bridge; never change tensor values."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from semantic_bridge import KNOWN_MODELS, SCHEMA, _tensor_sha, file_sha, read_weights  # noqa: E402


def convert(source, output=None):
    from safetensors import safe_open
    from safetensors.torch import save_file

    weights, source_sha = read_weights(source)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    report = {
        "schema": SCHEMA, "operation": "lossless_metadata_wrap", "source_sha256": source_sha,
        "source": KNOWN_MODELS.get(source_sha, {"name": "unrecognized_compatible_weights"}),
        "compute_profile": "fp32", "activation": "silu", "rms_epsilon": 1e-6,
        "magnitude_epsilon": 1e-8,
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": _tensor_sha(value)} for key, value in weights.items()},
    }
    if output is None:
        return report
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing output: {output}")
    if output.suffix != ".safetensors":
        raise ValueError("Output must end in .safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata["t8_semantic_bridge_conversion"] = json.dumps(report, sort_keys=True)
    fd, temp_name = tempfile.mkstemp(prefix=".bridge-", suffix=".partial", dir=output.parent)
    os.close(fd)
    try:
        save_file(weights, temp_name, metadata=metadata)
        # Safetensors direct decode, since temporary file is deliberately not loadable by nodes.
        from safetensors.torch import load_file
        check = load_file(temp_name, device="cpu")
        for key, tensor in weights.items():
            if _tensor_sha(check[key]) != _tensor_sha(tensor):
                raise RuntimeError(f"Conversion changed tensor {key}")
        # Exclusive atomic publication on both Windows and POSIX. rename()
        # overwrites an existing destination on POSIX after a racing creator.
        # Unsupported filesystems fail clearly, never fall back to overwrite.
        os.link(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    report["output_sha256"] = file_sha(output)
    report["tensor_identity_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="Omit for inspect/dry-run")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
