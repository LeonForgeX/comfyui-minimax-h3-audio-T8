"""Build API graphs for VDN Core compatibility and learned two-pass refinement."""

from __future__ import annotations

import copy
import uuid

try:
    from . import run_openvdn_h3_validation as base
except ImportError:
    import run_openvdn_h3_validation as base


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def build_prompt(args, run_id):
    variant = getattr(args, "variant", "t2va")
    if variant != "t2va":
        try:
            from .run_openvdn_h3_multimodal_validation import build_variant_prompt
        except ImportError:
            from run_openvdn_h3_multimodal_validation import build_variant_prompt
        graph = build_variant_prompt(args, run_id, variant_name=variant)
    else:
        graph = base.build_prompt(args, run_id)
    if variant == "t2va":
        graph["6"]["inputs"]["prompt"] = (
        "A locked-off medium close-up of one adult woman in a softly lit concert room. "
        "She says exactly once in Mandarin: <d>[Chinese] 你在哪里</d>. Quiet classical "
        "piano and cello music underneath a clear voice. One continuous shot, natural "
        "face and skin, no other words, no subtitles, no cuts, no hiss or distortion."
    )
    if variant == "t2va" and getattr(args, "image", None):
        graph["20"] = node("LoadImage", image=args.image)
        graph["6"]["inputs"].update(task_type="I2VA", first_frame=["20", 0])
    choice = getattr(args, "attention", "default")
    if choice == "pytorch_before":
        graph["50"] = node("ModelAttentionBackend", model=["4", 0], attention="pytorch attention")
        graph["5"]["inputs"]["model"] = ["50", 0]
    elif choice in ("sparse_before", "sparse_after"):
        graph["50"] = node("BlockSparseAttention", model=["4" if choice == "sparse_before" else "5", 0],
                           selection="Sol-Attn (adaptive tau)", **{"selection.tau": 1.3},
                           start_percent=0.2, end_percent=0.8, dense_blocks="", min_tokens=12288,
                           extra_tokens=0, sink_conditioning="exact_kv_and_rows", verbose=True)
        graph["5" if choice == "sparse_before" else "7"]["inputs"]["model"] = ["50", 0]
    if getattr(args, "two_pass", False):
        graph["60"] = node("MiniMaxH3LearnedLatentUpscaleT8Advanced", av_latent=["10", 0],
                           model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors",
                           size_mode="scale_by", scale_by=args.scale_by, target_megapixels=0.5,
                           target_width=960, target_height=544, aspect_policy="preserve_source",
                           max_anisotropy=1.05, precision="fp16", release_policy="offload_after")
        graph["61"] = copy.deepcopy(graph["6"])
        graph["61"]["inputs"].update(width=["60", 1], height=["60", 2])
        graph["62"] = node("MiniMaxH3TwoPassLatentReconcileT8Advanced", learned_latent=["60", 0],
                           highres_template=["61", 1], positive=["61", 0], audio_policy="first_pass",
                           second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
        model_link = graph["7"]["inputs"]["model"]
        graph["63"] = node("MiniMaxH3VDNRefinePlanT8Advanced", model=model_link,
                           av_latent=["62", 0], first_pass_latent=["10", 0], refine_steps=args.refine_steps)
        graph["64"] = node("RandomNoise", noise_seed=args.seed + 1)
        graph["65"] = node("BasicGuider", model=["63", 0], conditioning=["62", 1])
        graph["66"] = node("SamplerCustomAdvanced", noise=["64", 0], guider=["65", 0],
                           sampler=["63", 1], sigmas=["63", 2], latent_image=["62", 0])
        graph["67"] = node("MiniMaxH3TwoPassAudioAuditT8Advanced", second_pass_input=["62", 0],
                           second_pass_output=["66", 0], expected_audio_strength=0.,
                           fail_on_locked_mismatch=True, locked_atol=1e-5)
        graph["11"]["inputs"]["av_latent"] = ["67", 0]
        graph["68"] = node("PreviewAny", source=["63", 3])
        graph["69"] = node("PreviewAny", source=["60", 3])
        graph["70"] = copy.deepcopy(graph["11"])
        graph["70"]["inputs"]["av_latent"] = ["10", 0]
        graph["71"] = copy.deepcopy(graph["12"])
        graph["71"]["inputs"].update(frames=["70", 0], audio=["70", 1])
        graph["72"] = node("CreateVideo", images=["71", 0], audio=["71", 1], fps=24., bit_depth=8)
        graph["73"] = node("SaveVideo", video=["72", 0], filename_prefix=f"VDN_Core_Compat_First_Pass/{run_id}", format="mp4", codec="h264")
        graph["74"] = node("PreviewAny", source=["67", 1])
        if getattr(args, "refine_backend", "vdn") == "native_h3":
            # A distinct clean loader branch, never the VDN Composer output.
            # Reuse the established learned-upscale high-resolution schedule;
            # its coarse sigma output is deliberately NOT used for VDN pass 1.
            graph["80"] = copy.deepcopy(graph["4"])
            graph["81"] = node("MiniMaxH3LoRACompatibilityLoaderT8Advanced", model=["80", 0],
                               lora_name=args.native_refine_lora, strength_model=1.)
            graph["63"] = node("MiniMaxH3DualClockSamplerT8", model=["81", 0],
                               av_latent=["62", 0], steps=8, shift_video=12., shift_audio=3.,
                               sampler_name="dual_clock_euler", scheduler="native_flow")
            graph["82"] = node("MiniMaxH3LearnedTwoPassParityPlanT8Advanced", model=["63", 0],
                               base_steps=8, coarse_steps=4, refine_steps=args.refine_steps)
            graph["66"]["inputs"]["sigmas"] = ["82", 1]
            graph["68"]["inputs"]["source"] = ["82", 2]
            graph["83"] = node("PreviewAny", source=["81", 1])
            if getattr(args, "native_probe", False):
                graph["85"] = node("T8VDNNativeBranchProbe", model=["63", 0], vdn_model=["5", 0])
                graph["65"]["inputs"]["model"] = ["85", 0]
                graph["86"] = node("T8VDNNativeBranchResult", av_latent=["67", 0], receipt=["85", 1],
                                   expected_forwards=args.refine_steps)
                graph["11"]["inputs"]["av_latent"] = ["86", 0]
                graph["87"] = node("PreviewAny", source=["86", 1])
    # Use the native VIDEO contract rather than relying on a VHS output suffix.
    graph.pop("13")
    graph["17"] = node("CreateVideo", images=["12", 0], audio=["12", 1], fps=24., bit_depth=8)
    graph["18"] = node("SaveVideo", video=["17", 0], filename_prefix=f"VDN_Core_Compat/{run_id}", format="mp4", codec="h264")
    if getattr(args, "safe_probe_save", False):
        graph["18"] = node("T8VDNSafeProbeSave", images=["12", 0], audio=["12", 1],
                           filename_prefix=f"VDN_Core_Compat/{run_id}")
        graph["88"] = node("PreviewAny", source=["18", 0])
        if getattr(args, "two_pass", False):
            graph["73"] = node("T8VDNSafeProbeSave", images=["71", 0], audio=["71", 1],
                               filename_prefix=f"VDN_Core_Compat_First_Pass/{run_id}")
            graph["89"] = node("PreviewAny", source=["73", 0])
    elif getattr(args, "video_save", "native") == "isolated":
        graph["18"] = node("MiniMaxH3SafeAVSaveT8Advanced", images=["12", 0], audio=["12", 1],
                           filename_prefix=f"VDN_Core_Compat/{run_id}", crf=18)
        graph["88"] = node("PreviewAny", source=["18", 2])
        graph.pop("17")
        if getattr(args, "two_pass", False):
            graph["73"] = node("MiniMaxH3SafeAVSaveT8Advanced", images=["71", 0], audio=["71", 1],
                               filename_prefix=f"VDN_Core_Compat_First_Pass/{run_id}", crf=18)
            graph["89"] = node("PreviewAny", source=["73", 2])
            graph.pop("72")
    return graph


def build_frontend(graph, object_info, *, refine_backend):
    """Serialize the tested API graph with the live Core widget contract."""
    graph = copy.deepcopy(graph)
    safe_probe = graph["18"]["class_type"] == "T8VDNSafeProbeSave"
    if safe_probe:
        # Draft GUI remains a native-node graph until the isolated saver is
        # promoted. Do not claim this transformed graph was GPU validated.
        for save_id, create_id, report_id in (("18", "17", "88"), ("73", "72", "89")):
            if save_id in graph:
                prefix = graph[save_id]["inputs"]["filename_prefix"]
                graph[save_id] = node("SaveVideo", video=[create_id, 0], filename_prefix=prefix,
                                      format="mp4", codec="h264")
                graph.pop(report_id, None)
    if graph.pop("85", None) is not None:
        graph.pop("86")
        graph.pop("87")
        graph["65"]["inputs"]["model"] = ["63", 0]
        graph["11"]["inputs"]["av_latent"] = ["67", 0]
    # Node 50 deliberately introduces an attention conflict for the probe.
    # It remains in prompt.json, but is not part of the user two-pass recipe.
    diagnostic = graph.pop("50", None)
    if diagnostic is not None:
        upstream = diagnostic["inputs"]["model"]
        for node_data in graph.values():
            for name, value in node_data["inputs"].items():
                if value == ["50", 0]:
                    node_data["inputs"][name] = list(upstream)
    try:
        from .api_to_frontend_workflow import convert
        from .frontend_workflow_compat import normalize_native_widget_inputs
    except ImportError:
        from api_to_frontend_workflow import convert
        from frontend_workflow_compat import normalize_native_widget_inputs
    title = f"OpenVDN 8 + {refine_backend} Refine / 潜空间二采 EXP"
    workflow = convert(graph, object_info, title)
    normalize_native_widget_inputs(workflow)
    task = graph["6"]["inputs"]["task_type"]
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8:vdn-two-pass:{refine_backend}:{task}"))
    titles = {"4": "原始 H3 底模 / Base model", "5": "VDN 一采模型（已含适配器）",
              "6": "低分辨率条件 / LOW", "10": "VDN 完整 8 步一采",
              "60": "学习型潜空间放大 / Learned upscale", "61": "高分辨率条件 / HIGH（尺寸自动同步）",
              "62": "二采条件与一采音频锁定", "63": f"{refine_backend} 二采设置",
              "66": "高分辨率二采", "67": "锁定音频校验", "80": "原生二采独立底模（不要接 VDN MODEL）",
              "81": "仅原生二采使用新版 EMA B", "82": "原生二采 sigma（不使用 coarse 输出）"}
    for source_id, node_data in zip(graph, workflow["nodes"]):
        if source_id in titles:
            node_data["title"] = titles[source_id]
    note_id = max(node["id"] for node in workflow["nodes"]) + 1
    text = (
        "## OpenVDN 学习型潜空间二采 · EXP\n\n"
        "一采是完整 DMD 8 步，不是 LightX2V 4+4。只改 LOW 的宽高和放大倍率，HIGH 尺寸已自动连接。"
        "默认保留一采音频；第二次采样后先校验并重锁音频，再保存。两次采样分别使用固定种子。\n\n"
        "VDN 分支不要额外加载 EMA/Turbo/SLA LoRA。"
        + ("本图原生二采使用另一条原始 MODEL 分支加载新版 step600 EMA B，不要换接 VDN Composer 输出。\n\n"
           if refine_backend == "native_h3" else "本图两次采样都使用 VDN，二采沿用该阶段自身 sigma 尾段，无额外 LoRA。\n\n")
        + "模型：完整 minimax_h3_fl2va_int8_convrot.safetensors；OpenVDN/vdn-minimax-h3 放在 diffusion_models；"
        "minimax_h3_latent_upscaler_3d_fp16.safetensors 放在 latent_upscale_models；原生二采 EMA B 放在 loras。"
        "复用原 H3 文本编码器和音视频 VAE。全局 Sage 可保留，但这不代表 VDN 内部改用了 Sage 算法。\n\n"
        "此图是待人审实验路线，不保证二采一定更清楚。高分辨率峰值显存仍由二采决定；16GB 不保证所有尺寸/帧数安全。"
        "探针专用的后端冲突测试节点不包含在此用户工作流中；完整测试接线见同目录 prompt.json。"
    )
    if safe_probe:
        text += "\n\n警告：本次真实探针使用隔离安全编码；此GUI草稿仍为原生SaveVideo，编码路线不同，尚不可当作最终已验收工作流。"
    elif graph["18"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced":
        text += "\n\n使用安全音视频保存节点直接输出MP4，需要FFmpeg在PATH中；VIDEO输出可供后续节点读取，不要再接SaveVideo重复编码。固定24fps、偶数尺寸。"
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "使用说明 / Read first",
                              "pos": [0, -620], "size": [1050, 540], "flags": {}, "order": note_id,
                              "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [text]})
    workflow["last_node_id"] = note_id
    return workflow
