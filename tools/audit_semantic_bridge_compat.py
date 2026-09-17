"""Compare the entire published registry/schema and workflow bytes on CPU."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def registry(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    classes = asyncio.run(module.comfy_entrypoint().get_node_list())
    return [(cls.define_schema().node_id, cls.GET_NODE_INFO_V1()) for cls in classes]


def strip_additions(value):
    value = deepcopy(value)
    optional = value.get("input", {}).get("optional", {})
    for key in ("semantic_bridge", "semantic_bridge_pass1", "semantic_bridge_pass2"):
        optional.pop(key, None)
    order = value.get("input_order", {}).get("optional", [])
    value.get("input_order", {})["optional"] = [name for name in order if not name.startswith("semantic_bridge")]
    # Internal Python package location differs intentionally between isolated checkouts.
    value.pop("python_module", None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Isolate this CPU-only subprocess from optional Sage import-time GPU probes.
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0")
    sys.path.insert(0, str(args.core))
    sys.argv = [sys.argv[0]]
    from comfy.cli_args import args as core_args
    core_args.cpu = True
    import torch
    torch.set_num_threads(2)
    def refuse_cuda():
        raise RuntimeError("CPU schema audit attempted CUDA initialization")
    torch.cuda._lazy_init = refuse_cuda
    old, new = registry(args.baseline, "bridge_baseline_check"), registry(root, "bridge_candidate_check")
    old_ids, new_ids = [row[0] for row in old], [row[0] for row in new]
    assert len(old_ids) == 340 and new_ids[:340] == old_ids
    assert new_ids[340:] == ["MiniMaxH3SemanticBridgeConfigT8", "MiniMaxH3SemanticBridgeApplyT8"]
    changed = []
    for (node_id, before), (_, after) in zip(old, new):
        if strip_additions(before) != strip_additions(after):
            changed.append(node_id)
    if changed:
        raise RuntimeError("Unexpected legacy schema changes: " + str(changed))
    workflows = {}
    for source in sorted((args.baseline / "examples/workflows").rglob("*.json")):
        relative = source.relative_to(args.baseline)
        current = root / relative
        if current.read_bytes() != source.read_bytes():
            raise RuntimeError("Legacy workflow changed: " + str(relative))
        workflows[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {"status": "legacy_registry_schema_workflow_pass", "legacy_nodes": 340,
              "current_nodes": len(new_ids), "legacy_workflows": workflows,
              "allowed_schema_changes": "append-only optional Bridge sockets only",
              "cuda_initialized": torch.cuda.is_initialized(), "human_quality": "not tested"}
    if result["cuda_initialized"]:
        raise RuntimeError("CPU audit initialized CUDA")
    with args.report.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "legacy_workflows"}))


if __name__ == "__main__":
    main()
