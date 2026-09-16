"""Verify a completed dual-model chain resumes in a new process without resampling."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
CORE = next(
    (
        root
        for root in PROJECT.parents
        if (root / "comfy").is_dir() and (root / "main.py").is_file()
    ),
    None,
)
TOOLS = PROJECT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import progressive_probe_control as control  # noqa: E402
import run_progressive_pilot as transport  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inside_artifacts(path: Path, *, must_exist: bool) -> Path:
    artifacts = (PROJECT / "artifacts").resolve(strict=True)
    path = path.resolve(strict=must_exist)
    if not path.is_relative_to(artifacts) or path == artifacts:
        raise ValueError("Resume probe paths must stay inside this project's artifacts directory")
    return path


def _snapshot_output(output: Path) -> dict[str, dict[str, int | str]]:
    rows: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        rows[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if not rows:
        raise ValueError("Completed source output is empty")
    return rows


def prepare_resume(source_root: Path, evidence_root: Path) -> tuple[dict, dict]:
    source_root = _inside_artifacts(source_root, must_exist=True)
    evidence_root = _inside_artifacts(evidence_root, must_exist=False)
    if evidence_root.exists():
        raise ValueError("Resume evidence directory must be new")
    terminal = _json(source_root / "terminal.json")
    if terminal.get("status") != "generation_completed_independent_media_and_human_review_pending":
        raise ValueError("Source artifact is not a completed reviewed-scope GPU generation")
    if terminal.get("video_context_mode") != "high_native_mask_ramp_exp":
        raise ValueError("Source artifact is not the HIGH latent-ramp candidate")
    if terminal.get("low_context_source") != "accepted_picture_low_context_v1":
        raise ValueError("Source artifact does not use accepted-picture LOW context")
    graph = _json(source_root / "generation" / "prompt.json")
    runner = graph.get("8", {}).get("inputs", {})
    if runner.get("resume_existing") is not False:
        raise ValueError("Source graph was not the original fresh generation")
    if runner.get("video_context_mode") != "high_native_mask_ramp_exp":
        raise ValueError("Source graph and terminal disagree on the ramp mode")
    if runner.get("low_context_source") != "accepted_picture_low_context_v1":
        raise ValueError("Source graph and terminal disagree on LOW context")
    expected = _json(source_root / "expected.json")
    expected_sources = {
        key.replace("\\", "/"): value
        for key, value in expected.get("sources", {}).items()
    }
    runtime_sources = sorted((PROJECT / "h3_t8").glob("*.py"))
    for path in runtime_sources:
        key = path.relative_to(PROJECT).as_posix()
        if expected_sources.get(key) != _sha256(path):
            raise ValueError(f"Runtime source changed since the bound GPU generation: {key}")
    report = _json(next((source_root / "output").rglob("last_execution_report.json")))
    compose = report.get("compose", {})
    final_path = Path(compose.get("output_path", "")).resolve(strict=True)
    if not final_path.is_relative_to((source_root / "output").resolve(strict=True)):
        raise ValueError("Bound final video escaped the source output directory")
    if _sha256(final_path) != compose.get("output_sha256"):
        raise ValueError("Bound final video checksum changed before resume")
    resumed = deepcopy(graph)
    resumed["8"]["inputs"]["resume_existing"] = True
    if "90" in resumed:
        raise ValueError("Reserved resume report node already exists")
    resumed["90"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["8", 5]},
    }
    receipt = {
        "source_root": str(source_root),
        "evidence_root": str(evidence_root),
        "chain_id": runner.get("chain_id"),
        "source_terminal_sha256": _sha256(source_root / "terminal.json"),
        "source_graph_sha256": _sha256(source_root / "generation" / "prompt.json"),
        "final_path": str(final_path),
        "final_sha256": compose["output_sha256"],
        "manifest_revision": compose.get("manifest_revision"),
        "segment_count": compose.get("segment_count"),
        "runtime_source_count": len(runtime_sources),
        "output_before": _snapshot_output(source_root / "output"),
    }
    return resumed, receipt


class ResumeOwnedServer(transport.OwnedServer):
    """Owned isolated server whose output is the existing bounded chain directory."""

    def __init__(self, root: Path, output: Path, port: int, headroom_gib: int):
        super().__init__(root, port, False, headroom_gib)
        self.output = output.resolve(strict=True)

    def start(self):
        if CORE is None:
            raise RuntimeError("ComfyUI core is required to start the isolated resume server")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        for name in ("temp", "user"):
            (self.root / name).mkdir()
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", PYTHONPATH=str(CORE))
        if env.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
            raise RuntimeError("Unexpected CUDA device mask; refusing an ambiguous GPU route")
        flags = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt"
            else 0
        )
        self.logs = [
            (self.root / f"server.{name}.log").open("x", encoding="utf-8")
            for name in ("stdout", "stderr")
        ]
        command = transport.server_command(
            self.root, self.port, False, self.headroom_gib
        )
        command[command.index("--output-directory") + 1] = str(self.output)
        transport.write_json(self.root / "server-command.json", command)
        self.process = subprocess.Popen(
            command,
            cwd=CORE,
            env=env,
            stdout=self.logs[0],
            stderr=self.logs[1],
            creationflags=flags,
        )
        print(f"Owned resume server PID {self.process.pid}", flush=True)


def _preview_report(history: dict) -> dict:
    text = history.get("outputs", {}).get("90", {}).get("text", [])
    if len(text) != 1:
        raise ValueError("Resume report PreviewAny output is missing")
    report = json.loads(text[0])
    if report.get("resume_action") != "returned_verified_existing_final":
        raise ValueError("Completed chain did not take the verified cache-only resume path")
    return report


def validate_resume(receipt: dict, report: dict, output_after: dict) -> dict:
    if output_after != receipt["output_before"]:
        raise ValueError("Resume changed an existing output, stage cache, manifest, or final file")
    if report.get("final_video_sha256") != receipt["final_sha256"]:
        raise ValueError("Resume returned a different final video checksum")
    audits = report.get("segment_audits")
    if not isinstance(audits, list) or len(audits) != receipt["segment_count"]:
        raise ValueError("Resume did not revalidate every accepted segment audit")
    return {
        "status": "cache_only_resume_pass",
        "resume_action": report["resume_action"],
        "chain_id": receipt["chain_id"],
        "manifest_revision": receipt["manifest_revision"],
        "segment_count": receipt["segment_count"],
        "final_sha256": receipt["final_sha256"],
        "runtime_source_count": receipt["runtime_source_count"],
        "output_file_count": len(output_after),
        "all_output_hashes_unchanged": True,
        "sampling_reused": False,
        "scope": "new-process resume of the exact bound two-segment eight-second chain",
    }


def main() -> None:
    global CORE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8209)
    parser.add_argument("--headroom-gib", type=int, choices=(0, 2), default=2)
    parser.add_argument("--core", type=Path, default=CORE)
    args = parser.parse_args()
    if args.core is None:
        raise ValueError("--core is required when the project is outside a ComfyUI checkout")
    CORE = args.core.resolve(strict=True)
    if not (CORE / "main.py").is_file() or not (CORE / "comfy").is_dir():
        raise ValueError("--core must point to a ComfyUI root")
    transport.CORE = CORE
    graph, receipt = prepare_resume(args.source_root, args.evidence_root)
    evidence_root = Path(receipt["evidence_root"])
    evidence_root.mkdir(parents=True)
    source_root = Path(receipt["source_root"])
    transport.write_json(evidence_root / "paths.json", _json(source_root / "paths.json"))
    transport.write_json(evidence_root / "source-receipt.json", receipt)
    server = ResumeOwnedServer(
        evidence_root, source_root / "output", args.port, args.headroom_gib
    )
    monitor = None
    result = {"status": "failed_before_resume_completion"}
    started = time.perf_counter()
    lease_path = PROJECT / "artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with control.SerialProbeLease(lease_path), control.NvmlResourceReader() as reader:
            guard = control.ResourceGuard()
            first = reader.sample()
            if guard.observe(first, startup=True):
                raise RuntimeError(guard.reason)
            server.start()
            monitor = transport.ContinuousGuard(
                reader, guard, evidence_root / "resources.jsonl", server
            )
            monitor.start()
            transport.wait_ready(server, monitor.check, seconds=180)
            server.request("GET", "/object_info")
            history, timing = transport.execute_graph(
                server,
                graph,
                evidence_root / "resume",
                monitor.check,
                timeout=900,
            )
            report = _preview_report(history)
            output_after = _snapshot_output(source_root / "output")
            result = validate_resume(receipt, report, output_after)
            result.update(
                wall_seconds=time.perf_counter() - started,
                timing=timing,
                resources=guard.report(),
            )
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        if monitor is not None:
            try:
                monitor.close()
            except BaseException as error:
                cleanup_errors.append(f"monitor: {type(error).__name__}: {error}")
        try:
            server.stop()
        except BaseException as error:
            cleanup_errors.append(f"server: {type(error).__name__}: {error}")
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        result["server_stop"] = server.stop_receipt
        transport.write_json(evidence_root / "terminal.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
