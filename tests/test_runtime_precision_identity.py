from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from h3_audio_t8_pkg.runtime_precision_identity import matmul_precision_identity


class _NewBackend:
    fp32_precision = "tf32"

    @property
    def allow_tf32(self):
        raise AssertionError("legacy TF32 getter must not be touched")


class _LegacyBackend:
    allow_tf32 = True


def test_backend_specific_precision_never_touches_legacy_getters():
    def legacy_global_getter():
        raise AssertionError("legacy global precision getter must not be touched")

    assert matmul_precision_identity(_NewBackend(), legacy_global_getter) == {
        "api": "backend_specific_fp32_precision",
        "cuda_matmul_fp32_precision": "tf32",
    }


def test_older_torch_falls_back_to_legacy_precision_pair():
    assert matmul_precision_identity(_LegacyBackend(), lambda: "high") == {
        "api": "legacy_allow_tf32",
        "global_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
    }


def test_real_torch_210_backend_specific_state_does_not_raise_mixed_api_error():
    module_path = Path(__file__).resolve().parents[1] / "h3_t8" / "runtime_precision_identity.py"
    script = f"""
import importlib.util
import json
import torch
spec = importlib.util.spec_from_file_location('runtime_precision_probe', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
print(json.dumps(module.matmul_precision_identity(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == {
        "api": "backend_specific_fp32_precision",
        "cuda_matmul_fp32_precision": "tf32",
    }
