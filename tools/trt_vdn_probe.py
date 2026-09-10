"""Fixed short VDN8+4 capture route for the owned progressive probe controller."""
from copy import deepcopy
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"
COMPOSER = "MiniMaxH3VDNModelComposerT8Advanced"


def is_vdn(graph):
    return graph.get("5", {}).get("class_type") == COMPOSER


def recipe(graph):
    g = deepcopy(graph)
    if g["6"]["inputs"].get("task_type") != "T2VA" or g["10"]["class_type"] != "T8ProgressiveNativeBaseline":
        raise ValueError("VDN capture requires the fixed T2VA native8 starting graph")
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    g["5"] = node(COMPOSER, model=["4", 0], vdn_root="OpenVDN/vdn-minimax-h3",
                  stage="stage_dmd_8nfe", verify_hashes=True, allow_structural_base=True)
    g["6"]["inputs"].update(width=512, height=256)
    g["7"] = node("MiniMaxH3VDNExecutionPlanT8Advanced", model=["5", 0], av_latent=["100", 1])
    g["10"]["class_type"] = "T8TRTVDNSamplerProbe"
    g.pop("104")
    g["103"] = node("PreviewAny", source=["5", 1])
    g["60"] = node("MiniMaxH3LearnedLatentUpscaleT8Advanced", av_latent=["10", 0],
                   model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors", size_mode="scale_by", scale_by=2.,
                   target_megapixels=0.5, target_width=1024, target_height=512, aspect_policy="preserve_source",
                   max_anisotropy=1.05, precision="fp16", release_policy="offload_after")
    g["61"] = deepcopy(g["6"])
    g["61"]["inputs"].update(width=["60", 1], height=["60", 2])
    g["62"] = node("MiniMaxH3TwoPassLatentReconcileT8Advanced", learned_latent=["60", 0], highres_template=["61", 1],
                   positive=["61", 0], audio_policy="first_pass", second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    g["63"] = node("MiniMaxH3VDNRefinePlanT8Advanced", model=["5", 0], av_latent=["62", 0], first_pass_latent=["10", 0], refine_steps=4)
    g["66"] = node("T8TRTVDNSamplerProbe", model=["63", 0], positive=["62", 1], av_latent=["62", 0],
                   sampler=["63", 1], sigmas=["63", 2], seed=g["10"]["inputs"]["seed"] + 1)
    g["67"] = node("MiniMaxH3TwoPassAudioAuditT8Advanced", second_pass_input=["62", 0], second_pass_output=["66", 0],
                   expected_audio_strength=0., fail_on_locked_mismatch=True, locked_atol=1e-5)
    g["11"]["inputs"]["av_latent"] = ["67", 0]
    for key, source in (("68", ["63", 3]), ("69", ["60", 3]), ("74", ["67", 1]), ("111", ["66", 1])):
        g[key] = node("PreviewAny", source=source)
    g["18"]["inputs"]["filename_prefix"] = "MiniMaxH3/TRTVDN8plus4/UNREVIEWED"
    return g


def report_nodes(graph):
    nodes = {"sampler": "21", "conditioning": "101", "decode": "102", "lora": "103", "save": "19"}
    if is_vdn(graph):
        nodes.pop("lora")
        nodes.update(composition="103", refine="111", audio_lock="74", refine_plan="68", upscale="69")
    return nodes


def sample(model, positive, av_latent, sampler, sigmas, seed, vdn):
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    import torch
    receipt = model.get_attachment(vdn.ATTACHMENT_KEY)
    if not isinstance(receipt, dict) or receipt.get("status") != "configured" or receipt.get("stage") != "stage_dmd_8nfe":
        raise ValueError("Expected the configured pinned VDN8 MODEL")
    branches = model.additional_models.get(vdn.ADDITIONAL_MODEL_KEY, [])
    if len(branches) != 1 or len(branches[0].model.blocks) != 50 or len(sigmas) not in (5, 9):
        raise ValueError("VDN branch or 8/4 schedule mismatch")
    vdn.validate_vdn_runtime_options(model.model_options.get("transformer_options", {}))
    clone = model.clone()
    counts = [0] * 50
    forwards = 0
    def measure(executor, *args, **kwargs):
        nonlocal forwards
        vdn.validate_vdn_runtime_options(clone.model_options.get("transformer_options", {}))
        forwards += 1
        return executor(*args, **kwargs)
    hooks = []
    try:
        for index, block in enumerate(branches[0].model.blocks):
            def observe(module, args, output, index=index):
                counts[index] += 1
            hooks.append(block.softmax_gate.register_forward_hook(observe))
        clone.add_wrapper_with_key("diffusion_model", "t8_trt_vdn_probe", measure)
        noise = RandomNoise.execute(seed).result[0]
        guider = BasicGuider.execute(clone, positive).result[0]
        output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av_latent).result[0]
        if forwards != len(sigmas) - 1 or counts != [forwards] * 50:
            raise ValueError("Not every actual VDN block executed on every planned forward")
        if any(not bool(torch.isfinite(value).all()) for value in output["samples"].unbind()):
            raise ValueError("VDN output contains nonfinite values")
        return output, {"status": "actual_vdn_execution_pass_quality_unreviewed", "actual_network_forwards": forwards,
                        "branch_gate_calls": counts, "cfg": 1., "seed": seed, "stage": receipt["stage"],
                        "no_extra_ema_lora": True, "scope": "Branch-local native Core sampler plus50actual VDN gate counters, not a quality claim"}
    finally:
        clone.remove_wrappers_with_key("diffusion_model", "t8_trt_vdn_probe")
        for hook in hooks:
            hook.remove()


def verify_reports(graph, reports):
    if not is_vdn(graph) or any(node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced" for node in graph.values()):
        raise ValueError("VDN probe must not add an EMA LoRA")
    for key, count, node in (("sampler", 8, "10"), ("refine", 4, "66")):
        result = reports[key]
        if (result.get("status") != "actual_vdn_execution_pass_quality_unreviewed"
                or result.get("actual_network_forwards") != count or result.get("branch_gate_calls") != [count] * 50
                or result.get("cfg") != 1 or result.get("seed") != graph[node]["inputs"]["seed"]):
            raise ValueError("VDN actual8+4 or50block counts mismatch")
    composed = reports["composition"]
    adapters = {row["name"]: row for row in composed.get("adapters", [])}
    if composed.get("status") != "configured" or composed.get("stage") != "stage_dmd_8nfe" or composed.get("main_block_count") != 50:
        raise ValueError("VDN composition evidence mismatch")
    for name, count in (("default", 104), ("turbo", 259)):
        row = adapters.get(name, {})
        shapes = row.get("shape_validation", {})
        if (row.get("patch_targets") != count or row.get("applied_targets") != count
                or shapes.get("checked_targets") != count or shapes.get("all_shapes_exact") is not True):
            raise ValueError("VDN adapter shape/application mismatch")
    plan = reports["refine_plan"]
    if (plan.get("first_pass_nfe") != 8 or plan.get("refine_nfe") != 4 or plan.get("additional_turbo_lora") is not False
            or plan.get("audio_matches_first_pass") is not True or plan.get("high_video_shape") != [1, 24, 22, 32, 64]):
        raise ValueError("VDN second-pass plan/shape/audio mismatch")
    audio = reports["audio_lock"]
    if (audio.get("status") != "locked_audio_replaced_exact" or audio.get("audio_relocked_exact") is not True
            or audio.get("audio_within_tolerance") is not True or audio.get("video_finite") is not True
            or audio.get("audio_finite") is not True):
        raise ValueError("VDN second-pass audio-lock evidence mismatch")


def freeze_assets(output):
    # A derived manifest only; pinned common model recipe remains untouched.
    from tools.progressive_probe_control import file_identity
    root = PROJECT.parents[1] / "models/diffusion_models/OpenVDN/vdn-minimax-h3/stage-dmd-step-250"
    names = ["model_spec.json", "metadata.json", "linear_branch/config.json", "linear_branch/model.safetensors"]
    names += [f"adapters/{adapter}/{name}" for adapter in ("default", "turbo") for name in ("adapter_config.json", "adapter_model.safetensors")]
    manifest = json.loads((RESEARCH / "pilot-identities-v1.json").read_text())
    manifest["installed_assets"] += [file_identity(root / name) for name in names]
    manifest["scope"] = "Common fixed pilot assets plus VDN DMD8 full-base adapters and branch; no new model conversion"
    output = Path(output).resolve()
    if not output.is_relative_to(RESEARCH):
        raise ValueError("Write only a new research manifest")
    with output.open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
