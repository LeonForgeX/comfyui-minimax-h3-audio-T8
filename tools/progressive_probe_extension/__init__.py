"""Isolated benchmark instrumentation, not included in the pack's node registry."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import time
from contextlib import nullcontext

from comfy_api.latest import ComfyExtension, io
import torch


_allocator_session = None


def observe_allocator(boundary):
    # Local to this isolated probe package, never an installed global node hook.
    if _allocator_session is not None and _allocator_session.state == "active":
        _allocator_session.observe(boundary)


def project_module(name):
    import nodes
    expected = Path(__file__).resolve().parents[2] / "__init__.py"
    cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ProgressiveSamplerEXPT8")
    if cls is None:
        raise RuntimeError("The selected T8 progressive node has not been registered")
    package_name = cls.__module__.rsplit(".", 1)[0]
    package = importlib.import_module(package_name)
    if Path(package.__file__).resolve() != expected:
        raise RuntimeError("Registered progressive node belongs to a different project checkout")
    return importlib.import_module(package_name + "." + name)


def value_identity(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        raw = data.view(torch.uint8).numpy().tobytes()
        return {"tensor_shape": list(data.shape), "dtype": str(data.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(key): value_identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [value_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported conditioning identity type: {type(value).__name__}")


class ConditionAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveConditionAudit", category="T8/Probe only",
            inputs=[io.Conditioning.Input("positive"), io.Latent.Input("av_latent")],
            outputs=[io.Conditioning.Output("positive"), io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, positive, av_latent):
        parts = project_module("core").nested_av_parts(av_latent)
        identity = value_identity(positive)
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        report = {"conditioning_sha256": hashlib.sha256(encoded).hexdigest(), "conditioning": identity,
                  "initial_av": [value_identity(part) for part in parts], "scope": "actual_encoded_tensors_before_sampling"}
        observe_allocator("encoded_condition_identity_complete")
        return io.NodeOutput(positive, av_latent, json.dumps(report, allow_nan=False))


class ModelAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveModelAudit", category="T8/Probe only",
            inputs=[io.Model.Input("model"), io.String.Input("report_json"), io.Int.Input("expected_patch_count", default=259)],
            outputs=[io.Model.Output("model"), io.String.Output("report")])

    @classmethod
    def execute(cls, model, report_json, expected_patch_count):
        report = json.loads(report_json)
        if (report.get("status") != "applied" or report.get("selected_strength") != 1.0 or
                report.get("missed_patch_target_count") != 0 or
                report.get("patch_target_count") != expected_patch_count or
                report.get("applied_patch_count") != expected_patch_count or
                len(model.patches) != expected_patch_count):
            raise RuntimeError("Pilot EMA B did not bind every expected patch target; sampling is blocked")
        observe_allocator("lora_registration_audited_not_proof_of_gpu_residency")
        return io.NodeOutput(model, json.dumps({**report, "observed_model_patch_targets": len(model.patches),
                            "scope": "actual_loader_result_and_model_patch_registration_before_sampling"}))


class NativeBaseline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveNativeBaseline", category="T8/Probe only",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Latent.Input("av_latent"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Int.Input("seed", default=1, min=0, max=2**64-1)],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, model, positive, av_latent, sampler, sigmas, seed):
        from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
        runtime = project_module("progressive_sampling_runtime")
        runtime.validate_native_model(model, sampler)
        clone = model.clone()
        counts = {"actual_network_forwards": 0, "first_shapes": None}
        started = time.perf_counter()

        def measure(executor, x, *args, **kwargs):
            counts["actual_network_forwards"] += 1
            if counts["first_shapes"] is None:
                counts["first_shapes"] = [list(part.shape) for part in x]
            return executor(x, *args, **kwargs)

        clone.add_wrapper_with_key("diffusion_model", "t8_native_baseline_observer", measure)
        try:
            noise = RandomNoise.execute(seed).result[0]
            guider = BasicGuider.execute(clone, positive).result[0]
            compat = project_module('h3_core_compat')
            backend = compat.plain_attention_backend(model.model_options.get('transformer_options',{}).get('optimized_attention_override'))
            helper = project_module('tools.progressive_qualification')
            observer = helper.sage_audit() if backend == 'sage' else None
            with observer if observer is not None else nullcontext():
                output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av_latent).result[0]
            if counts["actual_network_forwards"] != len(sigmas)-1:
                raise RuntimeError("Native CFG1 baseline did not execute every planned network forward")
            report = {**counts, "status": "native_execution_count_pass_quality_unverified", "cfg": 1.,
                      "sampler_seconds": time.perf_counter()-started, "seed": seed,
                      "lora_patch_targets": len(model.patches),
                      "scope": "actual_Core_BasicGuider_RandomNoise_SamplerCustomAdvanced_with_branch_local_counter"}
            if observer is not None:
                report['attention_audit'] = helper.require_sage(observer.counts)
            return io.NodeOutput(output, json.dumps(report))
        finally:
            clone.remove_wrappers_with_key("diffusion_model", "t8_native_baseline_observer")


class TRTVDNSamplerProbe(NativeBaseline):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TRTVDNSamplerProbe", category="T8/Probe only",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Latent.Input("av_latent"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Int.Input("seed", default=1, min=0, max=2**64-1)],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, model, positive, av_latent, sampler, sigmas, seed):
        result, report = project_module("tools.trt_vdn_probe").sample(
            model, positive, av_latent, sampler, sigmas, seed, project_module("vdn_h3_advanced"))
        return io.NodeOutput(result, json.dumps(report))


class TimedVaeDelegate:
    def __init__(self, vae, label, timings):
        self.vae, self.label, self.timings = vae, label, timings

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def decode(self, *args, **kwargs):
        observe_allocator("before_" + self.label)
        started = time.perf_counter()
        try:
            return self.vae.decode(*args, **kwargs)
        finally:
            self.timings.append({"stage": self.label, "seconds": time.perf_counter()-started})


class TRTLatentCapture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TRTLatentCapture", category="T8/Probe only",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent):
        import folder_paths
        video, audio = project_module("core").nested_av_parts(av_latent)
        report = project_module("tools.trt_latent_capture").capture_parts(video, audio, folder_paths.get_output_directory())
        return io.NodeOutput(av_latent, json.dumps(report))


class TRTSavedReferenceEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TRTSavedReferenceEncoder", category="T8/Probe only",
            inputs=[io.Vae.Input("video_vae"), io.String.Input("evidence_root"),
                    io.Combo.Input("backend", options=["native", "trt"]), io.String.Input("audit_sha256"),
                    io.Combo.Input("reference_kind", options=["image", "video"], default="image", optional=True)],
            outputs=[io.Vae.Output("video_vae"), io.Image.Output("image")])

    @classmethod
    def execute(cls, video_vae, evidence_root, backend, audit_sha256, reference_kind="image"):
        delegate, image = project_module("tools.trt_encoder_condition_probe").load_reference(
            video_vae, evidence_root, backend, audit_sha256, reference_kind=reference_kind)
        return io.NodeOutput(delegate, image)


class TRTReferenceConsumedAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TRTReferenceConsumedAudit", category="T8/Probe only",
            inputs=[io.Vae.Input("video_vae"), io.Conditioning.Input("positive")],
            outputs=[io.Conditioning.Output("positive"), io.String.Output("report")])

    @classmethod
    def execute(cls, video_vae, positive):
        helper = project_module("tools.trt_encoder_condition_probe")
        if not isinstance(video_vae, helper.SavedReferenceEncoder):
            raise ValueError("Expected the research saved-reference delegate")
        return io.NodeOutput(positive, json.dumps(video_vae.report()))


class TimedDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveTimedDecode", category="T8/Probe only",
            inputs=[io.Latent.Input("av_latent"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae")],
            outputs=[io.Image.Output("frames"), io.Audio.Output("audio"), io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, video_vae, audio_vae):
        observe_allocator("sampling_output_ready_before_decode")
        timings = []
        frames, audio, *_ = project_module("audio_ops").decode_av_latent(av_latent,
            TimedVaeDelegate(video_vae, "video_vae_decode", timings), TimedVaeDelegate(audio_vae, "audio_vae_decode", timings))
        if [row["stage"] for row in timings] != ["video_vae_decode", "audio_vae_decode"]:
            raise RuntimeError("VAE decode timing did not cover both real decoder calls")
        observe_allocator("both_decoders_complete")
        return io.NodeOutput(frames, audio, json.dumps({"timings": timings,
            "scope": "unchanged_T8_decode_helpers_and_VAE_calls; wall_including_load_offload; no_forced_cuda_sync"}))


class SageBackend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='T8ProgressiveSageBackend',category='T8/Probe only',
                         inputs=[io.Model.Input('model')],outputs=[io.Model.Output('model')])

    @classmethod
    def execute(cls,model):
        from comfy.ldm.modules import attention
        if getattr(attention,'sageattn',None) is None:
            raise RuntimeError('Sage is not installed; no fallback/install')
        options = model.model_options.get('transformer_options',{})
        if options.get('optimized_attention_override') is not None:
            raise RuntimeError('Probe will not overwrite another attention owner')
        clone = model.clone()
        project_module('h3_core_compat').set_h3_attention_backend(clone,attention.attention_sage)
        return io.NodeOutput(clone)


class QualifiedProgressive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from copy import deepcopy
        schema = deepcopy(project_module('nodes_progressive_sampling').MiniMaxH3ProgressiveSamplerEXPT8.define_schema())
        schema.node_id = 'T8ProgressiveQualifiedSampler'
        return schema

    @classmethod
    def execute(cls,**kwargs):
        model = kwargs['model']
        override = model.model_options.get('transformer_options',{}).get('optimized_attention_override')
        if project_module('h3_core_compat').plain_attention_backend(override) != 'sage':
            raise ValueError('Sage qualification requires the actual plain Sage selector')
        helper = project_module('tools.progressive_qualification')
        with helper.sage_audit() as observer:
            result = project_module('nodes_progressive_sampling').MiniMaxH3ProgressiveSamplerEXPT8.execute(**kwargs).result
        report = json.loads(result[1])
        report['attention_audit'] = helper.require_sage(observer.counts)
        return io.NodeOutput(result[0],json.dumps(report))


class ProgressiveProbeExtension(ComfyExtension):
    async def get_node_list(self):
        return [ConditionAudit, ModelAudit, NativeBaseline, TimedDecode, EnvironmentAudit, AllocatorAudit, SageBackend, QualifiedProgressive, TRTLatentCapture, TRTSavedReferenceEncoder, TRTReferenceConsumedAudit, TRTVDNSamplerProbe]


class AllocatorAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveAllocatorAudit", category="T8/Probe only",
            inputs=[io.String.Input("run_id"), io.String.Input("action"),
                    io.String.Input("device_type"), io.Int.Input("expected_pid", min=1)],
            outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, run_id, action, device_type, expected_pid):
        import os
        import comfy.model_management as mm
        global _allocator_session
        if expected_pid != os.getpid():
            raise RuntimeError("Allocator action belongs to another process")
        if action not in {"begin", "finish", "abort"}:
            raise ValueError("Unknown allocator action")
        device = mm.get_torch_device()
        if device.type != device_type or device_type not in {"cpu", "cuda"}:
            raise RuntimeError("Allocator mode differs from the actual Comfy device")
        if action == "begin":
            if _allocator_session is not None and _allocator_session.state in {"active", "unavailable"}:
                raise RuntimeError("Another allocator interval is still open")
            # Import through the verified project package, rather than another tools directory.
            module = project_module("tools.progressive_memory_metrics")
            _allocator_session = module.AllocatorMemorySession(run_id=run_id,
                device_type=device_type, device_index=device.index if device_type == "cuda" else None)
            report = _allocator_session.begin()
        else:
            if _allocator_session is None or _allocator_session.run_id != run_id:
                raise RuntimeError("Allocator interval identity does not match")
            if action == "abort" and _allocator_session.state == "failed":
                report = _allocator_session.report()
            else:
                report = _allocator_session.finish(outcome="failed" if action == "abort" else "complete")
        return io.NodeOutput(json.dumps(report, allow_nan=False))


class EnvironmentAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8ProgressiveEnvironmentAudit", category="T8/Probe only",
            inputs=[io.String.Input("expected_json")], outputs=[io.String.Output("report")])

    @classmethod
    async def execute(cls, expected_json):
        import importlib.util
        import os
        import comfy.model_management as mm
        import execution
        import folder_paths
        expected = json.loads(expected_json)
        root = Path(__file__).resolve().parents[2]
        # Establish the actual registered project, not an arbitrary module alias.
        project_module("progressive_sampling_runtime")
        helper = root / "tools/vdn_probe_environment.py"
        spec = importlib.util.spec_from_file_location("t8_progressive_live_environment", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        core = module.verify_live_core(expected["core"])
        for relative, digest in expected["sources"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Probe source identity changed: " + relative)
        device = str(mm.get_torch_device())
        if (device == "cpu") != (expected["mode"] == "cpu-smoke"):
            raise RuntimeError("Live device differs from the explicitly selected probe mode")
        runtime_options = None
        if 'runtime_options' in expected:
            from comfy.cli_args import args
            runtime_options = {'reserve_vram_gib':args.reserve_vram,'headroom_gib':args.vram_headroom}
            if runtime_options != expected['runtime_options']:
                raise RuntimeError('Actual isolated Core headroom/reserve arguments differ')
            if device != 'cpu':
                import comfy_aimdo.control
                if comfy_aimdo.control.get_simple_vram_headroom() != int(args.reserve_vram*1024**3):
                    raise RuntimeError('DynamicVRAM simple reserve did not take effect')
        validation = []
        for name, graph in expected.get("pilot_graphs", {}).items():
            valid, error, outputs, node_errors = await execution.validate_prompt("progressive-validate-" + name, graph, None)
            validation.append({"case": name, "valid": valid, "error": error,
                               "outputs": outputs, "node_errors": node_errors})
            if not valid or node_errors:
                raise RuntimeError("Instrumented pilot graph failed Core validation: " + json.dumps(validation[-1]))
        resolved_assets = []
        for asset in expected.get("assets", []):
            path = Path(asset["path"]).resolve()
            relative = path.relative_to(Path(expected["core"]["core_root"]))
            if relative.parts[0] == "models":
                category, name = relative.parts[1], str(Path(*relative.parts[2:]))
                selected = Path(folder_paths.get_full_path_or_raise(category, name)).resolve()
            elif relative.parts[0] == "input":
                selected = Path(folder_paths.get_annotated_filepath(str(Path(*relative.parts[1:])))).resolve()
            else:
                raise RuntimeError("Asset leaves the fixed model/input scope")
            if selected != path or selected.stat().st_size != asset["bytes"] or selected.stat().st_mtime_ns != asset["mtime_ns"]:
                raise RuntimeError("Actual Core asset lookup differs from the frozen input: " + str(path))
            resolved_assets.append({"path": str(selected), "externally_verified_sha256": asset["sha256"]})
        return io.NodeOutput(json.dumps({"status": "pass", "core": core,
            "project_root": str(root), "sources_checked": len(expected["sources"]), "pid": os.getpid(),
            "device": device, "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "cuda_initialized": torch.cuda.is_initialized(), "pilot_validation": validation, "resolved_assets": resolved_assets,
            "runtime_options":runtime_options,
            "scope": "actual_imports_and_asset_resolution_not_trained_model_execution"}))


def comfy_entrypoint():
    return ProgressiveProbeExtension()
