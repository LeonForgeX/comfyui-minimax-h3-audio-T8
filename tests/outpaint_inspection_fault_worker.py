"""Owned subprocess fault fixture. No Comfy, Torch, GPU, or user-file modification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

blob = Path(sys.argv[1]).read_bytes()
request = json.loads(blob)
params = request["parameters"]
mode = params.get("mode", "ok")
if mode == "exit":
    os._exit(55)
if mode == "sleep":
    time.sleep(30)
if mode == "orphan":
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    Path(params["child_pid_file"]).write_text(str(child.pid))
    time.sleep(0.7)  # Parent can observe the owned descendant before abrupt exit.
    os._exit(55)
if mode == "invalid":
    print("not JSON")
    raise SystemExit(0)
if mode == "large_output":
    print("x" * 2048)
    raise SystemExit(0)
if mode == "large_log":
    print("x" * 2048, file=sys.stderr)
    raise SystemExit(0)
response = {"schema": request["schema"], "request_sha256": hashlib.sha256(blob).hexdigest(),
            "ok": True, "result": {"fixture": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
if mode == "identity":
    response["request_sha256"] = "0" * 64
if mode == "value_error":
    response.update(ok=False, error_type="ValueError", error="fixture invalid media")
print(json.dumps(response))
