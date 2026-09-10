"""Bind a recorded Windows NGX LUID to CUDA/NVML UUID without creating a CUDA context.

Uses NVIDIA's documented cuDeviceGetLuid device-management API, not a device-name
guess: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
Enumeration alone is not DLSS frame-generation qualification.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_pilot import RESEARCH, write_json  # noqa: E402
from progressive_probe_control import SerialProbeLease, file_identity  # noqa: E402
from dlss_fi_transport import runtime_identity  # noqa: E402


def cuda_device_inventory():
    if os.name != "nt":
        raise RuntimeError("This graphics LUID route is Windows-only")
    driver_path = Path(os.environ["SystemRoot"]) / "System32/nvcuda.dll"
    driver = ctypes.WinDLL(str(driver_path.resolve(strict=True)))
    signatures = {"cuInit": [ctypes.c_uint], "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetLuid": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]}
    def call(name, *args):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = signatures[name], ctypes.c_int
        result = fn(*args)
        if result != 0:
            raise RuntimeError(f"CUDA device enumeration {name} returned {result}")
    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if not 1 <= count.value <= 32:
        raise RuntimeError("Unexpected CUDA device count")
    rows = []
    for ordinal in range(count.value):
        device, mask = ctypes.c_int(), ctypes.c_uint()
        luid, unique, name = ctypes.create_string_buffer(8), ctypes.create_string_buffer(16), ctypes.create_string_buffer(256)
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        call("cuDeviceGetLuid", luid, ctypes.byref(mask), device.value)
        call("cuDeviceGetUuid", unique, device.value)
        call("cuDeviceGetName", name, 256, device.value)
        rows.append({"ordinal": ordinal, "luid": hex(int.from_bytes(luid.raw, "little")), "node_mask": mask.value,
            "uuid": "GPU-" + str(uuid.UUID(bytes=unique.raw)), "name": name.value.decode("utf-8", "strict")})
    return {"devices": rows, "driver": file_identity(driver_path), "cuda_context_created": False,
        "api_scope": "cuInit and device-management queries only; no allocation/kernel/primary context"}


def bind_probe(probe, inventory, expected_uuid):
    if not probe.get("supports_native_2x_by_report"):
        raise ValueError("Recorded feature report does not support native 2x")
    matches = re.findall(r"SetGPUArch:: Gpu count = (\d+), luid: (0x[0-9a-fA-F]+)", probe["stderr_tail"])
    identities = {(int(count), int(luid, 16)) for count, luid in matches}
    if len(identities) != 1:
        raise ValueError("Missing or ambiguous actual NGX GPU LUID")
    count, luid = identities.pop()
    if count != 1:
        raise ValueError("Only the observed single-device NGX route is qualified")
    candidates = [row for row in inventory["devices"] if int(row["luid"], 16) == luid]
    if len(candidates) != 1 or candidates[0]["node_mask"] != 1 or candidates[0]["uuid"] != expected_uuid:
        raise ValueError("NGX LUID is not uniquely bound to the expected physical CUDA/NVML UUID")
    return {"status": "recorded_NGX_LUID_matches_current_CUDA_and_NVML_UUID",
        "device": candidates[0], "generation_qualified": False, "audio_qualified": False, "quality_qualified": False}


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.probe_root.resolve(strict=True), args.output.resolve()
    if not root.is_relative_to(RESEARCH) or not output.is_relative_to(RESEARCH) or output.exists():
        raise ValueError("Use existing research evidence and a new research output file")
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    terminal, probe, installed = read("terminal.json"), read("probe-result.json"), read("runtime-identity.json")
    if (terminal["status"] != "feature_probe_completed_device_and_generation_unqualified" or
            terminal["server_stop"]["exit_code"] != 0 or terminal["server_stop"]["owned_children_remaining"] or
            terminal["resources"]["status"] != "observations_within_policy"):
        raise ValueError("Not a completed successful feature probe")
    # LUID is locally scoped. Never bind a pre-reboot recording to a new session.
    boot_time = psutil.boot_time()
    if not boot_time < (root / "probe-result.json").stat().st_mtime <= time.time():
        raise ValueError("Recorded feature probe predates current Windows boot; fresh execution evidence needed")
    runtime = Path(installed["dlssg-worker.exe"]["path"]).parent
    if runtime_identity(runtime) != installed:
        raise ValueError("Recorded runtime identity changed")
    observed = {Path(row["path"]).name: row for row in probe["observed_module_identities"]}
    for name, identity in installed.items():
        if name not in observed or any(observed[name][key] != identity[key] for key in ("path", "sha256", "bytes")):
            raise ValueError("Actual mapped modules do not match the pinned runtime")
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        inventory = cuda_device_inventory()
        result = bind_probe(probe, inventory, terminal["resources"]["gpu_uuid"])
        result.update(inventory=inventory, boot_time=boot_time, evidence={name: file_identity(root / name) for name in
            ("terminal.json", "probe-result.json", "probe-output.json", "runtime-identity.json")},
            scope="Supplemental same-Windows-boot LUID binding; not a new feature probe or generated-frame test")
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
