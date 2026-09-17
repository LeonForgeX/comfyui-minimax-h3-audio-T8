"""CPU-only identity and same-profile numerical checks for installed bridge pairs."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from semantic_bridge import BridgeConfig, KNOWN_MODELS, apply_bridge, read_weights  # noqa: E402


def verify_pair(source, converted):
    weights, source_sha = read_weights(source)
    other, converted_sha = read_weights(converted)
    if source_sha not in KNOWN_MODELS:
        raise ValueError("Source SHA is not one of the pinned author artifacts")
    if not all(weights[key].dtype == other[key].dtype and torch.equal(weights[key], other[key]) for key in weights):
        raise ValueError("Converted tensors differ")
    h = torch.randn(2, 11, 5120, generator=torch.Generator().manual_seed(917))
    checks = []
    for mode in ("per_token", "global", "none"):
        result = []
        for path, sha in ((source, source_sha), (converted, converted_sha)):
            result.append(apply_bridge([[h, {}]], BridgeConfig(str(path), sha, magnitude_match=mode,
                                                              device="cpu", chunk_tokens=4), cancel=lambda: None)[0][0][0])
        if not torch.equal(*result):
            raise ValueError(f"Same-profile parity failed: {mode}")
        checks.append({"mode": mode, "bit_equal": True, "finite": bool(torch.isfinite(result[0]).all()),
                       "delta_rms": float((result[0] - h).square().mean().sqrt())})
    return {"source_name": Path(source).name, "source_sha256": source_sha,
            "converted_name": Path(converted).name, "converted_sha256": converted_sha,
            "tensor_values_and_dtype_equal": True, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.model_root
    pairs = [
        (root / "original/MiniMaxH3_SemanticBridge_v1.safetensors", root / "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors"),
        (root / "bunny/BUNNY_H3_ActionLogic_Bridge_V1.safetensors", root / "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors"),
    ]
    payload = {"stage": "SB-P0", "status": "passed", "device": "cpu", "compute_profile": "fp32",
               "pairs": [verify_pair(*pair) for pair in pairs], "cuda_initialized": torch.cuda.is_initialized(),
               "scope": "Conversion parity only; not generation or quality acceptance"}
    if payload["cuda_initialized"]:
        raise RuntimeError("CPU verifier unexpectedly initialized CUDA")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
