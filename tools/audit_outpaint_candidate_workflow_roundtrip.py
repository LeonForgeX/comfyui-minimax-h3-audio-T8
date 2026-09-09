"""Read native browser-saved workflows and compare every execution input/edge.

No browser automation, generation or workflow writes. Writes only a new evidence
report after every generated graph passes. Named widget values require the tested native
frontend; absence is an error rather than guessing positional widget order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .build_video_outpaint_candidate_workflows import ROUTES, CANDIDATE
except ImportError:
    from build_video_outpaint_candidate_workflows import ROUTES, CANDIDATE


def audit_roundtrip(prompt, saved):
    def same_widget(actual, expected):
        if isinstance(expected, bool):
            return isinstance(actual, bool) and actual is expected
        if isinstance(expected, int):
            return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        if isinstance(expected, float):
            return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    and float(actual) == expected)
        return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)

    real = {node["id"]: node for node in saved["nodes"] if node["type"] != "MarkdownNote"}
    ids = {key: index + 1 for index, key in enumerate(prompt)}
    if set(real) != set(ids.values()):
        raise ValueError("browser changed the execution node set")
    links = {item[0]: item for item in saved["links"]}
    checked_edges, checked_values, owned = 0, 0, set()
    for key, source in prompt.items():
        node = real[ids[key]]
        if node["type"] != source["class_type"] or node["mode"] != 0:
            raise ValueError("browser changed a node type or execution mode")
        named = node.get("widgets_values_named", {})
        for name, value in source["inputs"].items():
            matches = [(i, pin) for i, pin in enumerate(node["inputs"]) if pin["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous input {key}.{name}")
            slot, pin = matches[0]
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in ids and isinstance(value[1], int):
                expected = [ids[str(value[0])], value[1], ids[key], slot]
                link = links.get(pin["link"])
                if link is None or link[1:5] != expected:
                    raise ValueError(f"browser changed edge {key}.{name}")
                origin = real[expected[0]]["outputs"][expected[1]]
                if pin["link"] not in (origin.get("links") or []):
                    raise ValueError("edge lost origin ownership")
                owned.add(pin["link"])
                checked_edges += 1
            else:
                # bool is an int subclass; JSON equality keeps their types distinct.
                if pin.get("link") is not None or name not in named or not same_widget(named[name], value):
                    raise ValueError(f"browser changed widget {key}.{name}: {named.get(name)!r} != {value!r}")
                checked_values += 1
        if node["type"] == CANDIDATE and named.get("control_after_generate") != "fixed":
            raise ValueError("candidate seed no longer stays fixed")
    if owned != set(links):
        raise ValueError("browser added unaccounted execution links")
    return {"execution_nodes": len(real), "edges_checked": checked_edges,
            "widget_values_checked": checked_values, "all_execution_inputs_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--saved-dir", type=Path, required=True)
    parser.add_argument("--saved-prefix", default="V2_")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for route in ROUTES:
        base = "H3_Outpaint_" + route
        api = args.generated_dir / (base + "_api.json")
        saved = args.saved_dir / (args.saved_prefix + base + "_DRAFT.json")
        checked = audit_roundtrip(json.loads(api.read_bytes()), json.loads(saved.read_bytes()))
        reports[route] = {**checked, "api_sha256": hashlib.sha256(api.read_bytes()).hexdigest(),
            "browser_saved_sha256": hashlib.sha256(saved.read_bytes()).hexdigest()}
    record = {"status": "nine_workflow_native_save_roundtrip_pass_not_generation_acceptance",
        "workflows": reports, "queued_inference": False,
        "separately_evidenced": [
            "copying live candidate and selection IDs in the isolated CPU archive workflow",
            "live archive image reload after a fresh ComfyUI process",
            "learned GPU candidate generation and explicit-selection continuation to all windows",
            "no-diffusion completed-selection recompose with strict media validation",
        ],
        "not_covered_by_this_roundtrip": [
            "queueing any of these nine draft graphs",
            "human full-clip visual acceptance",
            "real regional-guidance generation and its perceptual effect",
            "DLSS processing of a completed outpaint output",
            "multi-material release gates",
        ]}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
