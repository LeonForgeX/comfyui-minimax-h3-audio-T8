"""Optional exact local Bridge-weight CPU chain; no downloads, GPU or queue."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from comfy.cli_args import args as core_args  # noqa: E402

core_args.cpu = True
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "_h18_cpu_namespace", ROOT / "tests/conftest.py"
)
namespace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(namespace)
import pytest  # noqa: E402
import torch  # noqa: E402
from test_h18_prompt_chain import _exercise_chain  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not options.model.is_file():
        raise FileNotFoundError(options.model)
    options.output.mkdir(parents=True, exist_ok=False)
    with pytest.MonkeyPatch.context() as patch:
        _exercise_chain(options.model, options.output, patch, alphas=(0.1, 0.1))
    assert not torch.cuda.is_initialized()
    print(
        "Local converted Bridge exact .10/.10 CPU chain passed; CUDA not initialized. "
        + str(options.output / "h18-chain-trace.json")
    )


if __name__ == "__main__":
    main()
