"""Validate all candidate execution graphs with actual Core; no queue or UI."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import execution
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("semantic_workflow_pkg", project / "__init__.py",
                                                submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    async def validate():
        for name in ("nodes_custom_sampler.py", "nodes_video.py", "nodes_preview_any.py"):
            if not await nodes.load_custom_node(str(args.core / "comfy_extras" / name), module_parent="comfy_extras"):
                raise RuntimeError("Required native nodes failed import: " + name)
        for cls in await package.comfy_entrypoint().get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
        results = {}
        for path in sorted(args.workflows.glob("*.api.json")):
            graph = json.loads(path.read_text(encoding="utf8"))
            result = await execution.validate_prompt("bridge-" + path.stem, graph, None)
            results[path.name] = result
        return results
    results = asyncio.run(validate())
    passed = len(results) == 5 and all(value[0] and not value[3] for value in results.values())
    report = {"status": "pass" if passed else "failed", "cases": results,
              "cuda_initialized": torch.cuda.is_initialized(), "queued": False, "browser_used": False}
    if report["cuda_initialized"]:
        raise RuntimeError("CPU workflow validation initialized CUDA")
    with args.report.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
