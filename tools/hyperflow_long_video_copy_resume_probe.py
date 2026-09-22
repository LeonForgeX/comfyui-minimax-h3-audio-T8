"""Prove a completed P7 chain resumes from an isolated copy, without sampling.

The source P7 evidence is read-only. A second throwaway copy tests only the
immutable AVStageCache corruption guard; it is not a completed-chain Core test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
DEVELOPMENT = ROOT / "artifacts" / "development"
STATE = "in_node_loop_effects_state.json"
LOCKS = {"in_node_loop.lock", "manifest.lock.v2"}
sys.path.insert(0, str(ROOT / "tools"))
import run_progressive_pilot as pilot  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in LOCKS and path.name != STATE
    }


def source_contract(source_evidence: Path) -> tuple[str, Path, dict, dict]:
    source_evidence = source_evidence.resolve(strict=True)
    if source_evidence.parent != DEVELOPMENT.resolve() or not source_evidence.name.startswith("hyperflow-long-video-p7-"):
        raise ValueError("Source must be one completed P7 evidence directory in this project's development artifacts")
    run = json.loads((source_evidence / "run.json").read_text(encoding="utf-8"))
    workflow = json.loads((source_evidence / "workflow.json").read_text(encoding="utf-8"))
    chain_id = run["chain_id"]
    if workflow["8"]["class_type"] != "MiniMaxH3HyperFlowLongVideoEXPT8":
        raise ValueError("Source workflow is not the isolated HyperFlow long-video node")
    if workflow["8"]["inputs"].get("chain_id") != chain_id or workflow["8"]["inputs"].get("resume_existing") is not True:
        raise ValueError("Source workflow chain ID or resume setting changed")
    source_chain = source_evidence / "output" / "minimax_h3_t8_long_video" / chain_id
    state = json.loads((source_chain / STATE).read_text(encoding="utf-8"))
    manifest = json.loads((source_chain / "manifest.json").read_text(encoding="utf-8"))
    if (state.get("status") != "complete" or state.get("accepted_count") != 2
            or manifest.get("revision") != 2 or len(manifest.get("segments", [])) != 2):
        raise ValueError("Source is not the completed two-segment P7 qualification chain")
    final = Path(state["final_video_path"]).resolve(strict=True)
    if not final.is_relative_to(source_chain.resolve()) or sha256(final) != state["final_video_sha256"]:
        raise ValueError("Source final video escaped its chain or failed SHA-256")
    return chain_id, source_chain, state, workflow


def copy_chain(source_chain: Path, dest_chain: Path, state: dict) -> dict[str, str]:
    dest_chain.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_chain, dest_chain, ignore=lambda _folder, names: LOCKS.intersection(names))
    source_hashes = snapshot(source_chain)
    if snapshot(dest_chain) != source_hashes:
        raise RuntimeError("Copied P7 chain differs from source before state path relocation")
    source_final = Path(state["final_video_path"]).resolve(strict=True)
    relative_final = source_final.relative_to(source_chain.resolve())
    dest_final = (dest_chain / relative_final).resolve(strict=True)
    copied_state = json.loads((dest_chain / STATE).read_text(encoding="utf-8"))
    copied_state["final_video_path"] = str(dest_final)
    if sha256(dest_final) != copied_state["final_video_sha256"]:
        raise RuntimeError("Relocated final video failed the frozen source SHA-256")
    (dest_chain / STATE).write_text(
        json.dumps(copied_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if snapshot(dest_chain) != source_hashes:
        raise RuntimeError("Relocation changed a file other than the copied state JSON")
    return source_hashes


def corrupt_cache_copy(dest_chain: Path, evidence: Path) -> dict:
    receipts = list((dest_chain / "hyperflow_stages" / "segment_00001").rglob("low_x0-*.json"))
    if len(receipts) != 1:
        raise RuntimeError("Expected exactly one second-segment LOW receipt")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if receipt.get("stage") != "low_x0":
        raise RuntimeError("Selected cache receipt is not a LOW x0 stage")
    corrupt_dir = evidence / "corrupt-cache-copy"
    corrupt_dir.mkdir()
    shutil.copy2(receipts[0].parent / receipt["tensor_file"], corrupt_dir / receipt["tensor_file"])
    bad = dict(receipt)
    bad["tensor_sha256"] = "0" * 64
    pilot.write_json(corrupt_dir / receipts[0].name, bad)

    sys.path.insert(0, str(CORE))
    import comfy.cli_args
    comfy.cli_args.args.disable_comfy_compiler = True
    package_name = "h3_audio_t8_copy_resume_probe"
    spec = importlib.util.spec_from_file_location(
        package_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    assert spec.loader is not None
    spec.loader.exec_module(package)
    module = importlib.import_module(package_name + ".h3_t8.long_video_dual_stage_cache")
    try:
        module.AVStageCache(corrupt_dir).load("low_x0", receipt["contract"])
    except ValueError as error:
        if "integrity failed" not in str(error):
            raise
        return {"status": "rejected_before_sampling", "error": str(error),
                "scope": "direct AVStageCache.load on throwaway copied receipt; completed-chain Core fast path does not scan stages"}
    raise RuntimeError("Corrupt copied LOW receipt was incorrectly accepted")


def request_json(server, method: str, path: str, **kwargs):
    server.assert_port_owner()
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, server.url + path, timeout=(4, 90), **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code} {response.text[:4000]}")
        return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8869)
    parser.add_argument("--timeout", type=int, default=900)
    options = parser.parse_args()
    chain_id, source_chain, source_state, workflow = source_contract(options.source_evidence)
    source_state_sha256 = sha256(source_chain / STATE)
    evidence = DEVELOPMENT / ("hyperflow-long-video-copy-resume-" + uuid.uuid4().hex[:12])
    evidence.mkdir(parents=True, exist_ok=False)
    config = probe_resource_config(CORE, ROOT)
    config["t8_runtime_models"]["hyperflow"] = str(CORE / "models" / "hyperflow" / "loras")
    pilot.write_json(evidence / "paths.json", config)
    pilot.CORE = CORE
    server = pilot.OwnedServer(evidence, options.port, False)
    base_command = pilot.server_command

    def command_with_video_helper(*args, **kwargs):
        command = base_command(*args, **kwargs)
        index = command.index("--extra-model-paths-config")
        command[index:index] = ["ComfyUI-VideoHelperSuite"]
        command.append("--disable-comfy-compiler")
        return command

    pilot.server_command = command_with_video_helper
    try:
        server.start()
        pilot.wait_ready(server, lambda: None, seconds=240)
        dest_chain = evidence / "output" / "minimax_h3_t8_long_video" / chain_id
        source_hashes = copy_chain(source_chain, dest_chain, source_state)
        pilot.write_json(evidence / "source-copy-sha256.json", source_hashes)
        negative = corrupt_cache_copy(dest_chain, evidence)
        pilot.write_json(evidence / "corrupt-cache-result.json", negative)
        info = request_json(server, "GET", "/object_info/MiniMaxH3HyperFlowLongVideoEXPT8")
        if "MiniMaxH3HyperFlowLongVideoEXPT8" not in info:
            raise RuntimeError("HyperFlow long-video node unavailable in the owned Core")
        pilot.write_json(evidence / "workflow.json", workflow)
        submitted = request_json(server, "POST", "/prompt", json={
            "prompt": workflow, "client_id": "hyperflow-long-video-copy-resume-owned"})
        pilot.write_json(evidence / "submitted.json", submitted)
        if submitted.get("node_errors"):
            raise RuntimeError("Copied P7 graph validation failed: " + json.dumps(submitted["node_errors"]))
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + options.timeout
        while time.monotonic() < deadline:
            history = request_json(server, "GET", "/history/" + prompt_id)
            if prompt_id in history:
                pilot.write_json(evidence / "history.json", history[prompt_id])
                if history[prompt_id]["status"].get("status_str") != "success":
                    raise RuntimeError("Copied P7 resume did not succeed; see owned Core history/log")
                previews = history[prompt_id].get("outputs", {}).get("8", {}).get("images", [])
                expected_folder = f"minimax_h3_t8_long_video/{chain_id}/assembled"
                if len(previews) != 1 or previews[0].get("subfolder") != expected_folder:
                    raise RuntimeError("Core did not return the copied chain's assembled preview")
                break
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during copy-resume proof")
            time.sleep(3)
        else:
            raise TimeoutError("Copied P7 resume did not complete before deadline")
        if snapshot(dest_chain) != source_hashes:
            raise RuntimeError("Resume changed immutable manifest/candidates/stages/final-media bytes")
        if snapshot(source_chain) != source_hashes or sha256(source_chain / STATE) != source_state_sha256:
            raise RuntimeError("Source P7 evidence changed during copied-chain resume")
        copied_state = json.loads((dest_chain / STATE).read_text(encoding="utf-8"))
        final = Path(copied_state["final_video_path"]).resolve(strict=True)
        if not final.is_relative_to(dest_chain.resolve()) or sha256(final) != source_state["final_video_sha256"]:
            raise RuntimeError("Resume returned a final movie outside the copy or with changed SHA-256")
        log = (evidence / "server.stderr.log").read_text(encoding="utf-8", errors="replace")
        if "0%|" in log or "100%|" in log:
            raise RuntimeError("Sampling progress appeared during a completed-chain copy resume")
        result = {"status": "copy_resume_without_resampling", "chain_id": chain_id,
                  "final_video_path": str(final), "final_video_sha256": sha256(final),
                  "immutable_file_count": len(source_hashes), "corrupt_stage_copy": negative,
                  "resume_existing": True,
                  "qualification_boundary": "No fresh visual-quality claim; one completed-chain Core return and a separate direct cache-corruption guard"}
        pilot.write_json(evidence / "copy-resume-result.json", result)
        print(json.dumps({"evidence": str(evidence), **result}, ensure_ascii=False), flush=True)
        return 0
    finally:
        server.stop()
        pilot.write_json(evidence / "owned-stop.json", server.stop_receipt)
        print("Evidence:", evidence, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
