"""Owned, serial ComfyUI pilot. Default is read-only preflight, never inference.

CPU smoke runs only live environment/primitive text nodes. GPU mode requires a
named fixed recipe and frozen asset manifest; no retries, installs or fallbacks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_probe_control import (  # noqa: E402
    NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity, summarize_execution_events,
)
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402
from progressive_qualification import QUALIFICATION_CASES, qualification_recipe, requested_frames  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
# A development checkout can live outside ComfyUI, including directly on a
# Windows drive. Importing this helper must not index nonexistent ancestors.
# Explicit controllers bind CORE after parsing their --core argument.
CORE = next((root for root in PROJECT.parents if (root / 'comfy').is_dir()
             and (root / 'main.py').is_file()), None)
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"
CASES = tuple(f"{task}_{route}" for task in ("T2VA", "I2VA") for route in ("native8", "progressive6plus2"))
EXPLORATION_CASES = tuple(f"{content}_{seed}" for content, seeds in (
    ("portrait", (2609032102, 2609032103)), ("game", (2609032101, 2609032102, 2609032103))) for seed in seeds)
GAME_PROMPT = (
    "A single continuous third-person cinematic shot from a high-quality fantasy video game. "
    "One armored knight with engraved silver plates and a dark blue cloth cape walks forward across a stone bridge "
    "toward a towering castle gate. The camera tracks smoothly beside the knight, showing the full body, moving legs, "
    "layered armor edges, stone paving lines and detailed background towers. A light breeze moves the cape. "
    "Consistent character proportions and coherent parallax, sharp stable textures, natural daylight. "
    "Clean footsteps and quiet wind ambience, no speech or music, no subtitles, no cuts, no glitches or distortion."
)


def exploration_recipe(recipe, case, exploration):
    """Only the five predeclared T2VA cases; never mutate the four baseline recipes."""
    if exploration not in EXPLORATION_CASES or case not in ("T2VA_native8", "T2VA_progressive6plus2"):
        raise ValueError("Exploration requires a predeclared T2VA case")
    graph = deepcopy(recipe)
    content, seed = exploration.split("_")
    if case.endswith("native8"):
        graph["8"]["inputs"]["noise_seed"] = int(seed)
    else:
        graph["10"]["inputs"]["seed"] = int(seed)
    if content == "game":
        graph["90"]["inputs"]["value"] = GAME_PROMPT
    graph["18"]["inputs"]["filename_prefix"] = f"MiniMaxH3/ProgressiveExploration/{exploration}/{case}"
    return graph


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_snapshot():
    paths = [*PROJECT.glob("*.py"), *(PROJECT / "h3_t8").rglob("*.py"), Path(__file__), PROJECT / "tools/progressive_probe_control.py",
             PROJECT / "tools/vdn_probe_environment.py", PROJECT / "tools/progressive_probe_extension/__init__.py",
             PROJECT / "tools/progressive_pilot_analysis.py", PROJECT / "tools/progressive_memory_metrics.py",
             PROJECT / "tools/run_progressive_exploration.py", PROJECT / "tools/progressive_qualification.py",
             PROJECT / "tools/run_progressive_qualifications.py", PROJECT / "tools/trt_latent_capture.py",
             PROJECT / "tools/trt_encoder_condition_probe.py", PROJECT / "tools/trt_vdn_probe.py"]
    return {str(path.relative_to(PROJECT)): file_identity(path)["sha256"] for path in sorted(set(paths))}


def instrument_recipe(recipe):
    graph = deepcopy(recipe)
    if graph["10"]["class_type"] == "SamplerCustomAdvanced":
        seed = graph.pop("8")["inputs"]["noise_seed"]
        graph.pop("9")
        graph["10"] = {"class_type": "T8ProgressiveNativeBaseline", "inputs": {
            "model": ["7", 0], "positive": ["100", 0], "av_latent": ["100", 1],
            "sampler": ["7", 1], "sigmas": ["7", 2], "seed": seed}}
    elif graph["10"]["class_type"] == "MiniMaxH3ProgressiveSamplerEXPT8":
        graph["10"]["inputs"].update(positive=["100", 0], av_latent=["100", 1])
        graph["13"]["inputs"]["conditioning"] = ["100", 0]
    else:
        raise ValueError("Only the fixed native/progressive routes may be instrumented")
    graph["100"] = {"class_type": "T8ProgressiveConditionAudit", "inputs": {
        "positive": ["6", 0], "av_latent": ["6", 1]}}
    graph["7"]["inputs"]["av_latent"] = ["100", 1]
    graph["104"] = {"class_type": "T8ProgressiveModelAudit", "inputs": {
        "model": ["5", 0], "report_json": ["5", 1], "expected_patch_count": 259}}
    graph["7"]["inputs"]["model"] = ["104", 0]
    graph["11"]["class_type"] = "T8ProgressiveTimedDecode"
    for node, source in {"21": ["10", 1], "101": ["100", 2], "102": ["11", 2], "103": ["104", 1]}.items():
        graph[node] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    return graph


def server_command(run_root, port, cpu, headroom_gib=0):
    if CORE is None:
        raise ValueError('An isolated checkout requires an explicitly selected ComfyUI Core')
    if type(headroom_gib) is not int or headroom_gib not in (0,2):
        raise ValueError('Only the declared 0/2GiB DynamicVRAM headroom is allowed')
    cmd = [sys.executable, "-X", "utf8", str(CORE / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
           "--disable-auto-launch", "--preview-method", "none", "--cache-none", "--reserve-vram", "5",
           "--disable-all-custom-nodes", "--whitelist-custom-nodes", PROJECT.name, "progressive_probe_extension",
           "--extra-model-paths-config", str(run_root / "paths.json"), "--input-directory", str(CORE / "input"),
           "--output-directory", str(run_root / "output"), "--temp-directory", str(run_root / "temp"),
           "--user-directory", str(run_root / "user"), "--database-url", "sqlite:///:memory:"]
    if cpu:
        cmd.append("--cpu")
    if headroom_gib:
        cmd += ['--vram-headroom',str(headroom_gib)]
    return cmd


class OwnedServer:
    """Only this Popen handle and verified descendants are ever terminated."""
    def __init__(self, root, port, cpu, headroom_gib=0):
        self.root, self.port, self.cpu = root, port, cpu
        self.headroom_gib = headroom_gib
        self.process = None
        self.logs = []
        self.stop_lock = threading.Lock()
        self.stop_receipt = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        for name in ("output", "temp", "user"):
            (self.root / name).mkdir()
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", PYTHONPATH=str(CORE))
        if self.cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        elif env.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
            raise RuntimeError("Unexpected CUDA device mask; refusing an ambiguous GPU route")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.logs = [(self.root / f"server.{name}.log").open("x", encoding="utf-8") for name in ("stdout", "stderr")]
        command = server_command(self.root, self.port, self.cpu, self.headroom_gib)
        write_json(self.root / "server-command.json", command)
        self.process = subprocess.Popen(command, cwd=CORE, env=env, stdout=self.logs[0], stderr=self.logs[1], creationflags=flags)
        print(f"Owned isolated server PID {self.process.pid}, mode {'CPU smoke' if self.cpu else 'GPU'}", flush=True)

    def assert_port_owner(self):
        import psutil
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Owned server is not live")
        rows = [row for row in psutil.net_connections(kind="tcp")
                if row.laddr and row.laddr.port == self.port and row.status == psutil.CONN_LISTEN]
        if not rows or any(row.pid != self.process.pid for row in rows):
            raise RuntimeError("Port is not exclusively owned by this server process")

    def request(self, method, endpoint, **kwargs):
        import requests
        self.assert_port_owner()
        with requests.Session() as session:
            session.trust_env = False
            response = session.request(method, self.url + endpoint, timeout=(1, 2), **kwargs)
            response.raise_for_status()
            return response.json()

    def stop(self):
        import psutil
        with self.stop_lock:
            if self.stop_receipt is not None:
                return
            children = []
            process = self.process
            try:
                if process is not None and process.poll() is None:
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        pass
                    process.terminate()  # Popen owns its Windows process handle, not a reused PID.
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for child in reversed(children):
                    try:
                        child.terminate()  # psutil checks PID reuse on this stored Process object.
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(children, timeout=3)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=2)
                self.stop_receipt = {"pid": process.pid if process else None, "shutdown": "controller_owned_termination",
                    "exit_code": process.poll() if process else None, "owned_children_remaining": [p.pid for p in alive]}
                if alive:
                    raise RuntimeError("An owned child did not exit; no further generation is allowed")
            finally:
                for handle in self.logs:
                    handle.close()


class ContinuousGuard:
    """Independent observation thread; fails closed and stops only the owned server."""
    def __init__(self, reader, guard, path, server):
        self.reader, self.guard, self.path, self.server = reader, guard, path, server
        self.done = threading.Event()
        self.failure = None
        self.thread = None

    def start(self):
        def watch():
            try:
                with self.path.open("x", encoding="utf-8") as stream:
                    while not self.done.is_set():
                        row = self.reader.sample()
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        reason = self.guard.observe(row)
                        if reason:
                            raise RuntimeError(reason)
                        self.done.wait(0.5)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                try:
                    self.server.stop()
                except BaseException as stop_error:
                    self.failure += f"; owned cleanup failed: {stop_error}"
        self.thread = threading.Thread(target=watch, name="t8-owned-resource-guard", daemon=True)
        self.thread.start()

    def check(self):
        if self.failure:
            raise RuntimeError("Resource guard stopped the owned run: " + self.failure)

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=25)
            if self.thread.is_alive():
                raise RuntimeError("Resource observation thread has not exited")


def wait_ready(server, check, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check()
        if server.process.poll() is not None:
            raise RuntimeError("Owned server exited during startup; see server stderr")
        try:
            with socket.create_connection(("127.0.0.1", server.port), timeout=0.5):
                server.assert_port_owner()
                return
        except (OSError, RuntimeError):
            time.sleep(0.5)
    raise TimeoutError("Owned server startup timed out")


def execute_graph(server, graph, folder, check, timeout=1800):
    import websocket
    folder.mkdir()
    write_json(folder / "prompt.json", graph)
    check()
    server.assert_port_owner()
    client = str(uuid.uuid4())
    ws = websocket.create_connection(server.url.replace("http:", "ws:") + "/ws?clientId=" + client,
                                     timeout=0.5, http_no_proxy=["127.0.0.1", "localhost"])
    events, history, prompt_id, terminal = [], None, None, None
    started = time.perf_counter()
    try:
        queued = server.request("POST", "/prompt", json={"prompt": graph, "client_id": client})
        write_json(folder / "submission.json", queued)
        if queued.get("node_errors") or not queued.get("prompt_id"):
            raise RuntimeError("Core rejected the instrumented graph")
        prompt_id = queued["prompt_id"]
        with (folder / "events.jsonl").open("x", encoding="utf-8") as stream:
            while time.perf_counter() - started < timeout:
                check()
                if server.process.poll() is not None:
                    raise RuntimeError("Owned server exited while executing")
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not isinstance(raw, str):
                    continue
                packet = json.loads(raw)
                event = {"elapsed_seconds": time.perf_counter() - started, **packet}
                events.append(event)
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                data = packet.get("data", {})
                if data.get("prompt_id") == prompt_id and packet["type"] in {
                        "execution_success", "execution_error", "execution_interrupted"}:
                    terminal = packet["type"]
                    break
            if terminal is None:
                raise TimeoutError("No terminal execution event before the fixed timeout")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            check()
            history = server.request("GET", "/history/" + prompt_id).get(prompt_id)
            if history:
                break
            time.sleep(0.1)
        if not history:
            raise RuntimeError("Terminal event did not produce authoritative history")
        write_json(folder / "history.json", history)
        summary = summarize_execution_events(events, prompt_id, graph, time.perf_counter() - started)
        write_json(folder / "timing.json", summary)
        if not summary["complete_uncached_graph"] or not history.get("status", {}).get("completed"):
            raise RuntimeError("Graph failed, was cached, or did not execute all instrumented nodes")
        return history, summary
    finally:
        ws.close()


def preview_report(history, node):
    texts = history["outputs"][node]["text"]
    if not isinstance(texts, list) or len(texts) != 1:
        raise RuntimeError("Unexpected preview report structure")
    return json.loads(texts[0])


def allocator_action(server, root, action, device_type, check):
    graph = {"1": {"class_type": "T8ProgressiveAllocatorAudit", "inputs": {
        "run_id": root.name, "action": action, "device_type": device_type,
        "expected_pid": server.process.pid}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}
    history, _ = execute_graph(server, graph, root / ("allocator-" + action), check, timeout=60)
    return preview_report(history, "2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "cpu-smoke", "gpu"), default="preflight")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--exploration-case", choices=EXPLORATION_CASES)
    parser.add_argument("--qualification-case", choices=QUALIFICATION_CASES)
    parser.add_argument("--capture-trt-latent", action="store_true",
                        help="Save exact sampler AV latent before native decode, only in the isolated output directory")
    parser.add_argument("--encoder-evidence", type=Path)
    parser.add_argument("--encoder-backend", choices=("native", "trt"))
    parser.add_argument("--encoder-reference-kind", choices=("image", "video"), default="image")
    parser.add_argument("--vdn-two-pass", action="store_true", help="Fixed512x256 VDN8 then learned2x and VDN4; no extra EMA")
    parser.add_argument('--headroom-gib',type=int,choices=(0,2),default=0)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--port", type=int, default=8197)
    args = parser.parse_args()
    if args.vdn_two_pass and (args.case != "T2VA_native8" or args.encoder_evidence or args.exploration_case or args.qualification_case):
        raise ValueError("VDN probe requires only the fixed short T2VA native8 case")
    if bool(args.encoder_evidence) != bool(args.encoder_backend):
        raise ValueError("Encoder evidence and backend must be supplied together")
    if args.encoder_reference_kind == "video" and not args.encoder_evidence:
        raise ValueError("Video encoder replay requires audited encoder evidence")
    if args.encoder_evidence and (args.case != "I2VA_native8" or args.exploration_case or args.qualification_case):
        raise ValueError("Encoder comparison is limited to the fixed short I2VA native8 case")
    if args.exploration_case and args.case not in ("T2VA_native8", "T2VA_progressive6plus2"):
        raise ValueError("Exploration requires an explicit T2VA route")
    if args.qualification_case and (args.exploration_case or args.case not in ('T2VA_native8','T2VA_progressive6plus2')):
        raise ValueError('Qualification cannot be mixed with exploration or another task')
    if args.headroom_gib and args.qualification_case != 'long32':
        raise ValueError('Additional headroom is limited to the diagnosed long32 qualification')
    root = args.run_root.resolve()
    if root == RESEARCH or not root.is_relative_to(RESEARCH) or root.exists():
        raise ValueError("A new dedicated run directory within acceleration research is required")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid isolated port")
    if args.mode == "gpu" and (not args.case or not args.identities or not args.ffmpeg or not args.ffprobe):
        raise ValueError("GPU mode needs a fixed case, frozen identity manifest and explicit FFmpeg/FFprobe binaries")
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        root.mkdir(parents=True)
        receipt = {"mode": args.mode, "status": "incomplete", "case": args.case,
                   "quality_qualified": False, "performance_qualified": False}
        if args.exploration_case:
            receipt["exploration_case"] = args.exploration_case
        if args.qualification_case:
            receipt['qualification_case'] = args.qualification_case
        server = monitor = None
        reader = None
        allocator_open = False
        try:
            expected = {"core": verify_core_source(CORE), "sources": source_snapshot(), "mode": args.mode,
                        "pilot_graphs": {case: instrument_recipe(json.loads(
                            (RESEARCH / "pilot-api-drafts" / (case + ".prompt.json")).read_text(encoding="utf-8")))
                            for case in CASES}}
            if args.exploration_case:
                base = json.loads((RESEARCH / "pilot-api-drafts" / (args.case + ".prompt.json")).read_text(encoding="utf-8"))
                expected["pilot_graphs"][args.case] = instrument_recipe(exploration_recipe(base, args.case, args.exploration_case))
                expected["exploration_case"] = args.exploration_case
            if args.qualification_case:
                expected['pilot_graphs'][args.case] = qualification_recipe(expected['pilot_graphs'][args.case],args.case,args.qualification_case)
                expected['qualification_case'] = args.qualification_case
            if args.vdn_two_pass:
                from trt_vdn_probe import recipe as vdn_recipe
                expected["pilot_graphs"][args.case] = vdn_recipe(expected["pilot_graphs"][args.case])
                receipt["vdn_two_pass"] = "fixed_8_plus_4_no_extra_ema"
            if args.encoder_evidence:
                from trt_encoder_condition_probe import condition_recipe, evidence_identity
                encoder_identity = evidence_identity(args.encoder_evidence)
                expected["encoder_evidence"] = encoder_identity
                expected["pilot_graphs"][args.case] = condition_recipe(expected["pilot_graphs"][args.case],
                    encoder_identity["root"], args.encoder_backend, encoder_identity["audit_sha256"], args.encoder_reference_kind)
                receipt["encoder_backend"] = args.encoder_backend
                receipt["encoder_reference_kind"] = args.encoder_reference_kind
                receipt["encoder_scope"] = "saved_actual_reference_encoding_comparison_not_runtime_benchmark"
            if args.capture_trt_latent:
                from trt_latent_capture import capture_recipe
                expected["pilot_graphs"] = {name: capture_recipe(value) for name, value in expected["pilot_graphs"].items()}
                receipt["trt_latent_capture"] = "enabled_non_benchmark_capture_overhead"
            if args.headroom_gib:
                expected['runtime_options'] = {'reserve_vram_gib':5,'headroom_gib':args.headroom_gib}
            graph = None
            if args.mode == "cpu-smoke" and args.identities:
                manifest = json.loads(args.identities.read_text(encoding="utf-8"))
                if manifest["core"] != expected["core"]:
                    raise RuntimeError("Frozen asset evidence belongs to another Core")
                expected["assets"] = manifest["installed_assets"]
                receipt["asset_identity_scope"] = "previous_full_hash_scan_plus_live_path_size_mtime_check; not_fresh_rehash_or_weight_load"
            if args.mode == "gpu":
                manifest = json.loads(args.identities.read_text(encoding="utf-8"))
                if manifest["core"] != expected["core"]:
                    raise RuntimeError("Frozen model evidence belongs to a different Core")
                for asset in manifest["installed_assets"]:
                    print("Verifying installed asset " + Path(asset["path"]).name, flush=True)
                    if file_identity(asset["path"]) != asset:
                        raise RuntimeError("Installed asset identity changed")
                recipe = RESEARCH / "pilot-api-drafts" / (args.case + ".prompt.json")
                if file_identity(recipe) != manifest["recipes"][recipe.name]:
                    raise RuntimeError("Fixed pilot recipe changed")
                write_json(root / "assets-verified.json", manifest["installed_assets"])
                expected["assets"] = manifest["installed_assets"]
                graph = expected["pilot_graphs"][args.case]
            write_json(root / "environment-expected.json", expected)
            if args.mode != "cpu-smoke":
                reader = NvmlResourceReader().__enter__()
                guard = ResourceGuard()
                row = reader.sample()
                guard.observe(row, startup=True)
                write_json(root / "startup-resources.json", {"sample": row, "guard": guard.report()})
                if guard.reason:
                    raise RuntimeError(guard.reason)
                if args.mode == "preflight":
                    receipt["status"] = "preflight_only_pass_no_server_or_inference"
                    return
            write_json(root / "paths.json", probe_resource_config(CORE, PROJECT))
            server = OwnedServer(root, args.port, args.mode == "cpu-smoke", args.headroom_gib)
            server.start()
            if reader:
                monitor = ContinuousGuard(reader, guard, root / "resources.jsonl", server)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            wait_ready(server, check)
            environment_graph = {
                "1": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}
            history, _ = execute_graph(server, environment_graph, root / "environment", check, timeout=60)
            live = preview_report(history, "2")
            if live["pid"] != server.process.pid or live["status"] != "pass":
                raise RuntimeError("Environment report does not bind the owned process")
            write_json(root / "live-environment.json", live)
            allocator_device = "cpu" if args.mode == "cpu-smoke" else "cuda"
            allocator_action(server, root, "begin", allocator_device, check)
            allocator_open = True
            if graph is not None:
                history, timing = execute_graph(server, graph, root / "generation", check)
                allocator = allocator_action(server, root, "finish", allocator_device, check)
                allocator_open = False
                write_json(root / "allocator-memory.json", allocator)
                from progressive_memory_metrics import validate_completed_interval
                validate_completed_interval(allocator, run_id=root.name, pid=server.process.pid,
                    device_type=allocator_device, gpu_uuid=guard.report()["gpu_uuid"])
                from trt_vdn_probe import report_nodes
                reports = {name: preview_report(history, node) for name, node in report_nodes(graph).items()}
                write_json(root / "reports.json", reports)
                if args.encoder_evidence:
                    encoder_report = preview_report(history, "109")
                    if (encoder_report.get("status") != "audited_reference_encoding_consumed"
                            or encoder_report.get("actual_encode_calls") != 1
                            or encoder_report.get("backend") != args.encoder_backend
                            or encoder_report.get("identity") != encoder_identity
                            or evidence_identity(args.encoder_evidence) != encoder_identity):
                        raise RuntimeError("Saved reference encoding was not consumed as audited")
                    write_json(root / "encoder-reference-consumed.json", encoder_report)
                if args.capture_trt_latent:
                    capture = preview_report(history, "106")
                    if capture.get("status") != "actual_sampler_output_captured_bit_exact":
                        raise RuntimeError("Actual sampled latent was not captured")
                    for item in capture["files"].values():
                        path = Path(item["path"]).resolve(strict=True)
                        if not path.is_relative_to(root / "output") or file_identity(path)["sha256"] != item["sha256"]:
                            raise RuntimeError("Saved latent leaves this run or changed after capture")
                    write_json(root / "trt-latent-capture.json", capture)
                from progressive_pilot_analysis import audit_media, verify_execution_report
                verify_execution_report(args.case, graph, reports)
                saved = reports["save"]
                output = Path(saved["output"]).resolve(strict=True)
                if saved.get("status") != "pass" or not output.is_relative_to(root / "output"):
                    raise RuntimeError("Safe output does not belong to the owned run")
                media = audit_media(output, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                    width=1024, height=512, count=requested_frames(args.qualification_case))
                if media["file"]["sha256"] != saved["output_sha256"]:
                    raise RuntimeError("Saved media hash changed before postflight")
                write_json(root / "media-audit.json", media)
                receipt.update(status="generation_media_validated_unreviewed", timing=timing,
                               thermal_scope="model_process_cold_after_lightweight_environment_probe; disk_cache_not_cold")
            else:
                # Actual second queue submission verifies cache-none, not just a synthetic event parser.
                execute_graph(server, environment_graph, root / "environment-repeat", check, timeout=60)
                allocator = allocator_action(server, root, "finish", allocator_device, check)
                allocator_open = False
                write_json(root / "allocator-memory.json", allocator)
                from progressive_memory_metrics import validate_completed_interval
                validate_completed_interval(allocator, run_id=root.name, pid=server.process.pid, device_type="cpu")
                receipt["status"] = "cpu_live_transport_pass_no_models_or_gpu"
            receipt["allocator_interval"] = {"file": "allocator-memory.json",
                "status": allocator["status"], "counter_scope": allocator["counter_scope"]}
            check()
            if verify_core_source(CORE) != expected["core"] or source_snapshot() != expected["sources"]:
                raise RuntimeError("Core/project changed while the owned server was executing")
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if allocator_open and server and server.process and server.process.poll() is None:
                try:
                    queue = server.request("GET", "/queue")
                    if queue.get("queue_running") or queue.get("queue_pending"):
                        receipt["allocator_abort_error"] = "Task not idle; stopping owned server instead of queuing a diagnostic"
                    else:
                        failed_allocator = allocator_action(server, root, "abort", allocator_device, lambda: None)
                        write_json(root / "allocator-aborted.json", failed_allocator)
                except BaseException as allocator_error:
                    receipt["allocator_abort_error"] = f"{type(allocator_error).__name__}: {allocator_error}"
            raise
        finally:
            try:
                if server:
                    server.stop()
                    receipt["server_stop"] = server.stop_receipt
                if monitor:
                    monitor.close()
                    receipt["resource_guard"] = monitor.guard.report()
                    receipt["resource_monitor_failure"] = monitor.failure
                    if monitor.failure:
                        receipt["status"] = "failed"
            except BaseException as cleanup_error:
                receipt.update(status="failed", cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}")
                raise
            finally:
                if reader:
                    reader.__exit__()
                write_json(root / "terminal.json", receipt)
                print(json.dumps({"status": receipt["status"], "run_root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
