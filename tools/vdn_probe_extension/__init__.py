"""Probe-only native second-pass ownership evidence; never registered by the pack."""

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import ComfyExtension, io

from .media import save_raw_av


class CoreEnvironmentProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8VDNCoreEnvironmentProbe", category="T8/Probe only",
                         inputs=[io.String.Input("expected_json")], outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, expected_json):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "vdn_probe_environment.py"
        spec = importlib.util.spec_from_file_location("_t8_vdn_probe_environment", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        return io.NodeOutput(json.dumps(helper.verify_live_core(json.loads(expected_json)), indent=2))


def check_native(model, options):
    if getattr(model, "get_attachment", lambda _: None)("t8_minimax_h3_openvdn_contract_v2") is not None:
        raise RuntimeError("native refine contains a VDN attachment")
    if any("vdn" in str(key).lower() for key in options):
        raise RuntimeError("native refine contains VDN transformer state")
    if options.get("patches_replace", {}).get("dit"):
        raise RuntimeError("native refine has DiT replacement hooks")
    additional = getattr(model, "additional_models", {})
    if any("vdn" in str(key).lower() for key in additional):
        raise RuntimeError("native refine retains a VDN additional-model branch")
    if len(model.patches) != 259:
        raise RuntimeError("native EMA B refine must have exactly 259 patch targets")


class NativeBranchProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8VDNNativeBranchProbe", category="T8/Probe only",
                         inputs=[io.Model.Input("model"), io.Model.Input("vdn_model")],
                         outputs=[io.Model.Output("model"), io.Custom("T8_NATIVE_BRANCH_PROBE").Output("receipt")])

    @classmethod
    def execute(cls, model, vdn_model):
        if model is vdn_model:
            raise RuntimeError("native and VDN MODEL patchers must be distinct")
        check_native(model, model.model_options.get("transformer_options", {}))
        patched = model.clone()
        state = {"forwards": 0, "checks": [], "patch_targets": len(model.patches),
                 "different_patcher": model is not vdn_model,
                 "shared_base_object": model.model is vdn_model.model,
                 "scope": "runtime MODEL options/additional-model ownership and native LoRA patch inventory; not a weight-value equivalence claim"}

        def audit(executor, *args, **kwargs):
            options = kwargs.get("transformer_options")
            if options is None and len(args) >= 4:
                options = args[3]
            if not isinstance(options, dict):
                raise RuntimeError("native branch runtime options missing")
            check_native(patched, options)
            state["forwards"] += 1
            state["checks"].append({"no_vdn_state": True, "no_dit_replacement": True,
                                    "no_vdn_additional_model": True, "patch_targets": len(patched.patches)})
            return executor(*args, **kwargs)

        patched.add_wrapper_with_key("diffusion_model", "t8_native_branch_probe", audit)
        return io.NodeOutput(patched, state)


class NativeBranchResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8VDNNativeBranchResult", category="T8/Probe only",
                         inputs=[io.Latent.Input("av_latent"), io.Custom("T8_NATIVE_BRANCH_PROBE").Input("receipt"),
                                 io.Int.Input("expected_forwards", default=4)],
                         outputs=[io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, receipt, expected_forwards):
        if receipt["forwards"] != expected_forwards or len(receipt["checks"]) != expected_forwards:
            raise RuntimeError("native refinement ownership was not checked on every expected forward")
        return io.NodeOutput(av_latent, json.dumps({**receipt, "status": "pass"}, indent=2))


class SafeProbeSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8VDNSafeProbeSave", category="T8/Probe only", is_output_node=True,
                         inputs=[io.Image.Input("images"), io.Audio.Input("audio"),
                                 io.String.Input("filename_prefix")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, images, audio, filename_prefix):
        root = Path(folder_paths.get_output_directory()).resolve()
        output = (root / (filename_prefix + ".mp4")).resolve()
        if not output.is_relative_to(root):
            raise ValueError("probe output must stay inside output directory")
        return io.NodeOutput(json.dumps(save_raw_av(images, audio, output), indent=2))


class ProbeExtension(ComfyExtension):
    async def get_node_list(self):
        return [NativeBranchProbe, NativeBranchResult, SafeProbeSave, CoreEnvironmentProbe]


def comfy_entrypoint():
    return ProbeExtension()
