"""Read final D1/H18 CPU receipts and freeze a local requirement/evidence audit.

No Torch, model/queue operations, Git mutations or publication. This verifies
specific supplied receipts, never invents cross-version/GPU qualification.
"""

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FORMAL = "examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json"
PROTECTED = {
    ".git/index": "a23cd02b9680601d008f7a4227ffa7cc87cffb2ead563f61db4d74841904c183",
    "sol_attn_minimax_v2.py": "931c3602d7433a1dad313aebbbbe13067fdf6c1c607cdcb1b52ac173ae21e5e6",
    "artifacts/director-plan-20260919/interactive-v4-provisional-baseline.html": "d7103c2a5fae78e02cfc8d4679457f5554fe43a9cb29b9d9f74d3480ded05a50",
    FORMAL: "76301a5a887ad3a5874ddf0984aa1b8f405a8a4434b6f8c2a0b11fae14fff73f",
}
SOURCES = [
    "__init__.py",
    "h3_t8/nodes.py",
    "h3_t8/director_project.py",
    "h3_t8/director_routes.py",
    "h3_t8/nodes_director.py",
    "web/director.js",
    "web/director/index.html",
    "web/director/session.mjs",
    "h3_t8/prompt_relay_advanced.py",
    "h3_t8/prompt_budget_advanced.py",
    "h3_t8/prompt_relay_long_video_advanced.py",
    "h3_t8/semantic_bridge.py",
    "h3_t8/long_video_dual_model_stages.py",
    "tests/test_director_d1.py",
    "tests/test_director_d1_integration.py",
    "tests/test_h18_prompt_chain.py",
    "tests/test_preflight_and_registration.py",
    "tools/check_director_d1_cpu.py",
    "tools/check_director_d1_browser.py",
    "tools/check_director_d1_edges.py",
    "tools/check_director_d1_node_audio.py",
    "tools/check_h18_live_chain_cpu.py",
    "tools/audit_director_d1_completion.py",
    "docs/DIRECTOR_D1.md",
    "docs/H18_PROMPT_CHAIN_AUDIT.md",
    "meta.json",
    "features.json",
    "pyproject.toml",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-xml", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--node-audio", type=Path, required=True)
    parser.add_argument("--bridge-trace", type=Path, required=True)
    parser.add_argument("--bridge-model", type=Path, required=True)
    parser.add_argument("--retired-core-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (
        ROOT
        == Path(
            "F:/AI-T8-video-onekey/ComfyUI/custom_nodes/minimax-h3-audio-T8"
        ).resolve()
    )
    assert not args.output.exists(), "Use a new audit filename; never replace evidence"
    requirements = []

    def proven(name, evidence):
        requirements.append(
            {"requirement": name, "status": "proven", "evidence": evidence}
        )

    receipts, hashes = {}, {}
    for name in ("browser", "edges", "node_audio"):
        path = getattr(args, name)
        receipt = read_json(path)
        assert receipt["status"] == "passed", (name, receipt)
        assert not receipt["page_errors"] and not receipt["queue_posts"]
        assert receipt["before_queue"] == receipt["after_queue"]
        receipts[name] = receipt
        hashes[str(path)] = digest(path)
        for check in receipt["checks"]:
            proven(check, [str(path)])
    assert tuple(
        len(receipts[n]["checks"]) for n in ("browser", "edges", "node_audio")
    ) == (7, 9, 3)

    suite = ET.parse(args.cpu_xml).getroot().find("testsuite")
    assert suite is not None
    counts = {
        k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")
    }
    assert counts == {"tests": 329, "failures": 0, "errors": 0, "skipped": 0}, counts
    cases = {c.attrib["name"] for c in suite.findall("testcase")}
    needed = {
        "test_actual_same_name_assets_persist_restart_and_cas",
        "test_alias_manifest_prevents_retargeting_and_missing_is_explicit",
        "test_media_order_first_last_video_soundtrack_independent_audio",
        "test_same_size_derived_preview_tamper_is_rejected",
        "test_active_draft_only_global_once_and_event_grid",
        "test_limits_apply_to_reference_slots_not_generation_frames",
        "test_traversal_and_asset_overwrite_are_refused",
        "test_old_354_registry_class_order_and_schemas_are_unchanged",
        "test_real_native_Core_validates_export_and_real_D1_node_executes_without_queue",
        "test_H18_formal_ordinary_prompt_plan_and_token_inputs_are_exact_before_after",
        "test_budget_to_plan_preserves_global_and_event_payload_whitespace",
        "test_formal_dual_recipe_full_cpu_chain_and_stage_pairing",
    }
    assert needed <= cases, needed - cases
    for case in sorted(needed):
        proven(case, [str(args.cpu_xml)])
    classes = {c.attrib["classname"] for c in suite.findall("testcase")}
    for name in (
        "test_dual_picture_context",
        "test_dual_high_video_prefix",
        "test_long_video_dual_model_runner",
        "test_long_video_dual_stage_cache",
        "test_nodes_long_video_dual_model",
    ):
        assert "tests." + name in classes, name
    proven(
        "Mandatory five double-stage/seam CPU regressions included", [str(args.cpu_xml)]
    )
    hashes[str(args.cpu_xml)] = digest(args.cpu_xml)

    # Entire Relay AST must match the original after reversing only the two
    # declared whitespace-preservation edits. This guards all sampling math.
    before = ast.parse(
        (
            ROOT / "artifacts/director-d1-20260919/backups/prompt_relay_advanced.py"
        ).read_text(encoding="utf-8")
    )
    current = ast.parse(
        (ROOT / "h3_t8/prompt_relay_advanced.py").read_text(encoding="utf-8")
    )
    old_functions = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    for node in current.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_local_prompt_lines":
            expected_local = deepcopy(old_functions[node.name])
            loop = next(s for s in expected_local.body if isinstance(s, ast.For))
            loop.body = ast.parse("if raw.strip():\n    lines.append(raw)").body
            assert ast.dump(node) == ast.dump(expected_local), (
                "Unexpected local-line logic change"
            )
            node.body = deepcopy(old_functions[node.name].body)
        if isinstance(node, ast.FunctionDef) and node.name == "build_prompt_relay_plan":
            assert ast.unparse(node.body[0]) == "global_prompt = str(global_prompt)"
            assert ast.unparse(node.body[1].test) == "not global_prompt.strip()"
            node.body[:2] = deepcopy(old_functions[node.name].body[:2])
    assert ast.dump(current) == ast.dump(before), "Unexpected Relay code change"
    proven(
        "Relay AST change limited to Global/Local whitespace preservation",
        [
            "h3_t8/prompt_relay_advanced.py",
            "artifacts/director-d1-20260919/backups/prompt_relay_advanced.py",
        ],
    )

    original_nodes = ast.parse(
        (ROOT / "artifacts/director-d1-20260919/backups/nodes.py").read_text(
            encoding="utf-8"
        )
    )
    new_nodes = ast.parse((ROOT / "h3_t8/nodes.py").read_text(encoding="utf-8"))
    new_nodes.body = [
        n
        for n in new_nodes.body
        if not (
            isinstance(n, ast.ImportFrom)
            and n.module in {"nodes_director", "director_routes"}
        )
    ]
    for node in ast.walk(new_nodes):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_node_list":
            node.body = [
                s
                for s in node.body
                if not (
                    isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Call)
                    and isinstance(s.value.func, ast.Name)
                    and s.value.func.id == "register_director_routes"
                )
            ]
            returned = next(s for s in node.body if isinstance(s, ast.Return))
            assert (
                isinstance(returned.value.elts[-1], ast.Name)
                and returned.value.elts[-1].id == "MiniMaxH3DirectorProjectT8"
            )
            returned.value.elts.pop()
    assert ast.dump(new_nodes) == ast.dump(original_nodes), (
        "Non-additive node registry edit"
    )
    proven(
        "Formal registry source is append-only, old node bodies untouched",
        ["h3_t8/nodes.py", "artifacts/director-d1-20260919/backups/nodes.py"],
    )

    trace = read_json(args.bridge_trace)
    assert trace["bridge_alphas"] == [0.1, 0.1]
    assert trace["formal_sha256"] == PROTECTED[FORMAL]
    assert trace["bridge_weight_sha256"] == digest(args.bridge_model)
    observed = trace["observations"]
    assert [(o["segment"], o["stage"]) for o in observed] == [
        (0, "LOW"),
        (0, "HIGH"),
        (1, "LOW"),
        (1, "HIGH"),
    ]
    assert len({json.dumps(o["token_ids"]) for o in observed}) == 1
    assert len({o["binding"]["binding_hash"] for o in observed}) == 4
    raw = bytes.fromhex(trace["original_global_utf8_hex"])
    assert all(
        x in raw for x in (b"\t", b"\r\n", "🙂".encode(), " ".encode(), "𠀀".encode())
    )
    for o in observed:
        assert bytes.fromhex(o["global_utf8_hex"]).startswith(raw + b"\n\n")
        assert o["installed_owner_execution_counts"] == {
            "completed_forwards": 4,
            "routed_attention_calls": 4,
        }
        assert o["stage_report"]["status"] == "applied_exp"
        assert o["stage_report"]["attention_patch_installed"]
        assert (
            bytes.fromhex(o["final_utf8_hex"]).count(b"Reference role bindings:") == 1
        )
        assert o["projected_events"][0]["global_end_frame_exclusive"] == 72
    proven(
        "H18 real converted Bridge .10/.10 and all four installed Relay owners",
        [str(args.bridge_trace), str(args.bridge_model)],
    )
    hashes[str(args.bridge_trace)] = digest(args.bridge_trace)

    for relative, expected in PROTECTED.items():
        assert digest(ROOT / relative) == expected, relative
    proven(
        "Frozen v4/formal workflow/user index/original Sol bytes unchanged",
        list(PROTECTED),
    )
    assert not list(
        (ROOT / "artifacts/director-d1-20260919/core-input").rglob(
            "*.qa-missing-backup"
        )
    )
    proven(
        "Missing-media fixture originals restored",
        ["artifacts/director-d1-20260919/core-input"],
    )
    assert args.retired_core_pid > 0
    alive = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-Process -Id {args.retired_core_pid} -ErrorAction SilentlyContinue).Id",
        ],
        text=True,
    ).strip()
    assert not alive, (
        "Owned CPU Core PID still alive (or reused); inspect, never kill blindly"
    )
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 8852)) != 0, "CPU QA port still listening"
    proven(
        "Owned CPU Core stopped after QA, no leftover QA listener",
        [f"OS PID {args.retired_core_pid}", "127.0.0.1:8852"],
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == "1.85.0"
    for doc in ("docs/DIRECTOR_D1.md", "docs/H18_PROMPT_CHAIN_AUDIT.md"):
        assert doc in project["tool"]["comfy"]["includes"] and (ROOT / doc).is_file()
    for metadata in ("meta.json", "features.json"):
        data = read_json(ROOT / metadata)
        assert data["director_D1_local"]["registered_nodes"] == 355
        assert (
            data["director_D1_local"]["status"]
            == "local_D1_CPU_and_browser_verified_not_published"
        )
        assert data["H18_1_local"]["documentation"] == "docs/H18_PROMPT_CHAIN_AUDIT.md"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert head == "91c1b4e9b680d07a6eacee6a3aa6b449a4697554"
    proven(
        "Formal source/docs/metadata present; published version and local Git HEAD unchanged",
        [
            "pyproject.toml",
            "meta.json",
            "features.json",
            "docs/DIRECTOR_D1.md",
            "docs/H18_PROMPT_CHAIN_AUDIT.md",
        ],
    )
    report = {
        "schema": "t8.director_D1_H18_1.local_completion.v1",
        "status": "D1_and_H18_1_verified_CPU_and_real_browser_only_not_published",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "formal_directory": str(ROOT),
        "version": "1.85.0",
        "local_nodes": 355,
        "cpu": counts,
        "browser_check_groups": 19,
        "requirements": requirements,
        "protected_sha256": PROTECTED,
        "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
        "evidence_sha256": hashes,
        "boundaries": [
            "D1 CPU preflight is not a GPU generation workflow; D2 has not started.",
            "No GPU, new generated video, external publication or user queue mutation.",
            "H18 Qwen/VAE/H3 projection doubles are explicit; real tokenizer/Bridge/Relay owner/router only.",
            "Legacy preserve_all is temporal weighting, not hard window text omission or speech timing certification.",
            "Cross-Core/frontend, Nodes2, multi-user and novice human efficiency remain D4.",
            "These 329 tests are the scoped combination regression, not all project tests.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "status": report["status"],
                "requirements": len(requirements),
                "cpu": counts,
                "browser_groups": 19,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
