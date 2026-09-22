"""Owned-Core P7 copy probe: accepted segment 0, resume only segment 1 HIGH.

No production module or completed P7 evidence is changed. The copied rev1
manifest and LOW/HIGH-input receipts model an interruption before HIGH output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import uuid

import requests

if __package__:
    from .hyperflow_long_video_copy_resume_probe import (
        CORE, DEVELOPMENT, LOCKS, ROOT, STATE, sha256, source_contract,
    )
    from . import run_progressive_pilot as pilot
    from .vdn_probe_environment import probe_resource_config
else:
    from hyperflow_long_video_copy_resume_probe import (
        CORE, DEVELOPMENT, LOCKS, ROOT, STATE, sha256, source_contract,
    )
    import run_progressive_pilot as pilot
    from vdn_probe_environment import probe_resource_config


RECIPE = "hyperflow_long_video_partial4plus4_exp_v1"
SAMPLE_START = re.compile(r"0%\|[^\r\n]*0/(\d+) \[00:00<\?, \?it/s\]")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stage_receipt(stage_dir: Path, name: str) -> tuple[Path, dict]:
    matches = list(stage_dir.glob(name + "-*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} receipt, got {len(matches)}")
    path = matches[0]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    contract = receipt.get("contract")
    if receipt.get("stage") != name or not isinstance(contract, dict):
        raise ValueError(f"{name} receipt stage/contract is invalid")
    key = name + "-" + hashlib.sha256(canonical(contract).encode()).hexdigest()
    tensor_name = receipt.get("tensor_file")
    if path.name != key + ".json" or not isinstance(tensor_name, str) or not tensor_name.startswith(key + "-"):
        raise ValueError(f"{name} receipt does not match its immutable stage key")
    tensor = (stage_dir / tensor_name).resolve(strict=True)
    if tensor.parent != stage_dir.resolve() or sha256(tensor) != receipt.get("tensor_sha256"):
        raise ValueError(f"{name} tensor is absent or failed SHA-256")
    return path, receipt


def preflight(source_chain: Path, state: dict) -> dict:
    full = json.loads((source_chain / "manifest.json").read_text(encoding="utf-8"))
    rev1 = json.loads((source_chain / "manifest.json.bak").read_text(encoding="utf-8"))
    if (full.get("revision") != 2 or len(full.get("segments", [])) != 2
            or rev1.get("revision") != 1 or len(rev1.get("segments", [])) != 1
            or full["segments"][0] != rev1["segments"][0]):
        raise ValueError("P7 backup is not the exact accepted-first-segment rev1 manifest")
    parent = rev1["segments"][0]
    accepted = (source_chain / parent["video_path"]).resolve(strict=True)
    if not accepted.is_relative_to(source_chain.resolve()) or sha256(accepted) != parent["video_sha256"]:
        raise ValueError("Accepted segment 0 video is not source-bound")
    audit = json.loads((source_chain / "candidates" / "segment_00000" / parent["candidate_id"] / "effects_audit.json").read_text(encoding="utf-8"))
    parent_low_sha = audit["sampling_plan"]["dual_model"]["low_context"]["sha256"]
    stage_parent = source_chain / "hyperflow_stages" / "segment_00001"
    stage_dirs = [item for item in stage_parent.iterdir() if item.is_dir()]
    if len(stage_dirs) != 1:
        raise ValueError("Expected exactly one second-segment HyperFlow stage namespace")
    stage_dir = stage_dirs[0]
    low_path, low = stage_receipt(stage_dir, "low_x0")
    high_input_path, high_input = stage_receipt(stage_dir, "high_input")
    high_output_path, _ = stage_receipt(stage_dir, "high_output")
    contract = low["contract"]
    if (contract.get("schema") != RECIPE or contract.get("job") != state["contract_sha256"]
            or contract.get("segment") != 1 or contract.get("parent_high") != parent["candidate_id"]
            or contract.get("parent_revision") != 1 or contract.get("parent_low_sha256") != parent_low_sha
            or contract.get("seed_low") != int(parent["seed"]) + 1
            or contract.get("seed_high") != int(parent["seed"]) + 2
            or contract.get("intervals") != [[0, 4], [4, 8]]):
        raise ValueError("Second-segment stage contract differs from frozen P7 parent/recipe")
    picture = contract.get("low_picture_context", {})
    if (picture.get("name") != "accepted_picture_low_context_v1"
            or picture.get("source_media_sha256") != parent["video_sha256"]
            or picture.get("source_frame_interval") != [85, 124]
            or picture.get("source_segment_index") != 0):
        raise ValueError("Second-segment LOW picture source differs from accepted first movie")
    high_contract = dict(high_input["contract"])
    if high_contract.pop("low_tensor_sha256", None) != low["tensor_sha256"] or high_contract != contract:
        raise ValueError("Cached HIGH input is not tied to this LOW x0 receipt")
    return {
        "rev1_manifest": rev1, "parent": parent,
        "stage_dir_relative": stage_dir.relative_to(source_chain).as_posix(),
        "low_receipt_relative": low_path.relative_to(source_chain).as_posix(),
        "high_input_receipt_relative": high_input_path.relative_to(source_chain).as_posix(),
        "omitted_high_output_receipt": high_output_path.relative_to(source_chain).as_posix(),
        "job_sha256": state["contract_sha256"],
        "accepted_first_sha256": parent["video_sha256"],
        "low_tensor_sha256": low["tensor_sha256"],
        "high_input_tensor_sha256": high_input["tensor_sha256"],
        "stage_contract_key_verified": True,
    }


def omit_relative(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if relative.name in LOCKS or relative.name == "last_execution_report.json":
        return True
    if parts[0] == "assembled":
        return True
    if len(parts) >= 2 and parts[:2] == ("candidates", "segment_00001"):
        return True
    if parts[0] == "accepted" and relative.name.startswith("segment_00001"):
        return True
    if len(parts) >= 3 and parts[:2] == ("hyperflow_stages", "segment_00001") and relative.name.startswith("high_output-"):
        return True
    return False


def immutable_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {STATE, "manifest.json", "manifest.json.bak"}
        and not omit_relative(path.relative_to(root))
    }


def prepare_copy(source_chain: Path, dest_chain: Path, state: dict, gate: dict) -> dict[str, str]:
    def ignore(folder, names):
        relative = Path(folder).resolve().relative_to(source_chain.resolve())
        return [name for name in names if omit_relative(relative / name)]

    dest_chain.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_chain, dest_chain, ignore=ignore)
    expected = immutable_snapshot(source_chain)
    if immutable_snapshot(dest_chain) != expected:
        raise RuntimeError("Interrupted-chain copy differs from selected source bytes")
    (dest_chain / "manifest.json").write_text(
        json.dumps(gate["rev1_manifest"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    copied = json.loads((dest_chain / STATE).read_text(encoding="utf-8"))
    copied.update(status="running", accepted_count=1, manifest_revision=1,
                  current_segment_index=1, final_video_path="", final_video_sha256="")
    if copied["contract_sha256"] != state["contract_sha256"]:
        raise RuntimeError("Copied loop state changed the frozen P7 job contract")
    (dest_chain / STATE).write_text(json.dumps(copied, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if immutable_snapshot(dest_chain) != expected:
        raise RuntimeError("Interrupted-chain staging changed retained source bytes")
    manifest = json.loads((dest_chain / "manifest.json").read_text(encoding="utf-8"))
    if manifest != gate["rev1_manifest"] or (dest_chain / "candidates" / "segment_00001").exists():
        raise RuntimeError("Copy did not preserve exactly one accepted segment")
    stage_dir = dest_chain / gate["stage_dir_relative"]
    stage_receipt(stage_dir, "low_x0")
    stage_receipt(stage_dir, "high_input")
    if list(stage_dir.glob("high_output-*")):
        raise RuntimeError("Completed second-segment HIGH stage survived into interrupted copy")
    return expected


def request_json(server, method: str, path: str, **kwargs):
    server.assert_port_owner()
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, server.url + path, timeout=(4, 90), **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code} {response.text[:4000]}")
        return response.json()


def source_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file() and path.name not in LOCKS}


def implementation_hashes() -> dict[str, str]:
    folder = ROOT / "h3_t8" / "hyperflow_long_video_exp"
    return {path.name: sha256(path) for path in folder.glob("*.py")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--copy-proof", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--timeout", type=int, default=1800)
    options = parser.parse_args()
    chain_id, source_chain, state, workflow = source_contract(options.source_evidence)
    copy_proof_path = options.copy_proof.resolve(strict=True)
    if copy_proof_path.parent != DEVELOPMENT.resolve() or not copy_proof_path.name.startswith("hyperflow-long-video-copy-resume-"):
        raise ValueError("copy-proof must be this project's independent completed-copy evidence")
    proof = json.loads((copy_proof_path / "copy-resume-result.json").read_text(encoding="utf-8"))
    if (proof.get("status") != "copy_resume_without_resampling" or proof.get("chain_id") != chain_id
            or proof.get("final_video_sha256") != state["final_video_sha256"]):
        raise ValueError("Cross-Core complete-copy identity proof does not match this source P7 chain")
    gate = preflight(source_chain, state)
    frozen_source = source_hashes(source_chain)
    frozen_implementation = implementation_hashes()
    evidence = DEVELOPMENT / ("hyperflow-long-video-interrupted-resume-" + uuid.uuid4().hex[:12])
    evidence.mkdir(parents=True, exist_ok=False)
    pilot.write_json(evidence / "preflight.json", {k: v for k, v in gate.items() if k != "rev1_manifest"})
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
        retained = prepare_copy(source_chain, dest_chain, state, gate)
        pilot.write_json(evidence / "retained-sha256.json", retained)
        info = request_json(server, "GET", "/object_info/MiniMaxH3HyperFlowLongVideoEXPT8")
        if "MiniMaxH3HyperFlowLongVideoEXPT8" not in info:
            raise RuntimeError("HyperFlow P7 node unavailable in owned Core")
        pilot.write_json(evidence / "workflow.json", workflow)
        submitted = request_json(server, "POST", "/prompt", json={
            "prompt": workflow, "client_id": "hyperflow-long-video-interrupted-resume-owned"})
        pilot.write_json(evidence / "submitted.json", submitted)
        if submitted.get("node_errors"):
            raise RuntimeError("Interrupted P7 graph validation failed: " + json.dumps(submitted["node_errors"]))
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + options.timeout
        starts = []
        while time.monotonic() < deadline:
            log = (evidence / "server.stderr.log").read_text(encoding="utf-8", errors="replace")
            starts = SAMPLE_START.findall(log)
            if len(starts) > 1 or any(value != "4" for value in starts):
                server.stop()
                raise RuntimeError(f"Only one HIGH 4-NFE sampling stage is allowed; observed starts={starts}")
            history = request_json(server, "GET", "/history/" + prompt_id)
            if prompt_id in history:
                pilot.write_json(evidence / "history.json", history[prompt_id])
                if history[prompt_id]["status"].get("status_str") != "success":
                    raise RuntimeError("Interrupted copied-chain resume failed; see owned Core history/log")
                break
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during interrupted resume")
            time.sleep(1)
        else:
            raise TimeoutError("Interrupted P7 resume did not finish by deadline")
        starts = SAMPLE_START.findall((evidence / "server.stderr.log").read_text(
            encoding="utf-8", errors="replace"))
        if starts != ["4"]:
            raise RuntimeError(f"Expected exactly one HIGH 4-NFE sampling pass, observed {starts}")
        if immutable_snapshot(dest_chain) != retained:
            raise RuntimeError("Resume changed a retained parent or LOW/HIGH-input byte")
        if source_hashes(source_chain) != frozen_source or implementation_hashes() != frozen_implementation:
            raise RuntimeError("Source P7 evidence or HyperFlow implementation changed during resume")
        manifest = json.loads((dest_chain / "manifest.json").read_text(encoding="utf-8"))
        final_state = json.loads((dest_chain / STATE).read_text(encoding="utf-8"))
        if (manifest.get("revision") != 2 or len(manifest.get("segments", [])) != 2
                or manifest["segments"][0] != gate["parent"] or final_state.get("status") != "complete"
                or final_state.get("accepted_count") != 2):
            raise RuntimeError("Resume did not preserve parent and complete one new child segment")
        stage_dir = dest_chain / gate["stage_dir_relative"]
        _, high = stage_receipt(stage_dir, "high_output")
        _, prepared = stage_receipt(stage_dir, "high_input")
        if high["contract"] != prepared["contract"] or high["report"].get("nfe") != 4:
            raise RuntimeError("New second-segment HIGH did not report four NFE")
        child = manifest["segments"][1]
        if (sum(int(row["frame_count"]) for row in manifest["segments"]) != 192
                or int(child["audio_end_sample"]) != 256000
                or any(row.get("strict_decode_validated") is not True for row in manifest["segments"])):
            raise RuntimeError("Resumed two-segment AV delivery failed frame/audio/strict-decode checks")
        audit_path = dest_chain / "candidates" / "segment_00001" / child["candidate_id"] / "effects_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        report = audit["sampling_plan"]["dual_model"]
        if report.get("low_reused") is not True or report.get("high_reused") is not False:
            raise RuntimeError("Second-segment audit did not prove LOW reuse and HIGH regeneration")
        if (dest_chain / "dual_stages").exists():
            raise RuntimeError("Interrupted HyperFlow route wrote the legacy stage namespace")
        final = Path(final_state["final_video_path"]).resolve(strict=True)
        if not final.is_relative_to(dest_chain.resolve()) or sha256(final) != final_state["final_video_sha256"]:
            raise RuntimeError("Composed resume output escaped copy or failed SHA-256")
        result = {"status": "interrupted_copy_resumed_high_only", "chain_id": chain_id,
                  "sample_starts": starts, "low_reused": True, "high_reused": False,
                  "retained_file_count": len(retained), "accepted_first_sha256": gate["accepted_first_sha256"],
                  "new_high_tensor_sha256": high["tensor_sha256"],
                  "final_video_path": str(final), "final_video_sha256": sha256(final),
                  "quality_gate": "new film still requires manual picture/audio/seam review"}
        pilot.write_json(evidence / "interrupted-resume-result.json", result)
        print(json.dumps({"evidence": str(evidence), **result}, ensure_ascii=False), flush=True)
        return 0
    finally:
        server.stop()
        pilot.write_json(evidence / "owned-stop.json", server.stop_receipt)
        print("Evidence:", evidence, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
