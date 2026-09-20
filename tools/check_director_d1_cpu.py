"""Explicit CPU-only scoped checks; does not start or submit to a ComfyUI server."""

import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from comfy.cli_args import args  # noqa: E402 -- configure Core CPU before Torch imports

args.cpu = True
sys.path.insert(0, str(ROOT))
import pytest  # noqa: E402
import torch  # noqa: E402

if __name__ == "__main__":
    result = pytest.main(
        sys.argv[1:] or [str(ROOT / "tests" / "test_director_d1.py"), "-q"]
    )
    assert not torch.cuda.is_initialized(), "CPU check initialized CUDA"
    raise SystemExit(result)
