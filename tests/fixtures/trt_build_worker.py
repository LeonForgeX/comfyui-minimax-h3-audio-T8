"""CPU fake compiler exercising real process isolation/publication; no GPU imports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trt_vae_build import OUTPUT_SHAPE, digest_file, validate_request, write_new_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text())
staging = Path(request["staging_path"])
mode = request.get("test_mode", "complete")
if mode in ("hang", "error"):
    (staging / "model.engine.partial").write_bytes(b"incomplete synthetic engine")
    if mode == "error":
        raise RuntimeError("synthetic compiler failure")
    child = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import time; time.sleep(120)"],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    print(f"CHILD={child.pid}", flush=True)
    time.sleep(120)
else:
    (staging / "model.engine").write_bytes(b"synthetic-engine-not-tensorrt")
    manifest = {"status": "compiled_not_execution_qualified", "request_sha256": validate_request(request),
                "engine_sha256": digest_file(staging / "model.engine"), "input_shape": request["input_shape"],
                "output_shape": list(OUTPUT_SHAPE), "gpu_uuid": request["gpu_uuid"],
                "runtime_version": request["runtime_version"]}
    write_new_json(staging / "manifest.json", manifest)
