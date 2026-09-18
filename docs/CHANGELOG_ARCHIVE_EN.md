# Historical README release entries

Historical records only. Old pending or unpublished statuses are superseded by the current homepage and latest release notes.

# MiniMax H3 Audio T8

**v1.85.0 (2026-09-18): Meridian, Avatar, native voice/emotion and opt-in previews.**
The bound samples passed user review, including direct fury and corrected parallel camera slide.
Use the formal [Avatar/voice/dual/long-video workflows](../examples/workflows/36-avatar-voice/README.md),
[four-node Meridian workflows and space/time editor](../examples/workflows/37-meridian/README.md),
and [diagnostic/audio examples](../examples/workflows/38-diagnostics-preview/README.md).
Meridian uses its own merged-DMD native ConvRot INT8 model and authorized Omega geometry;
no full-precision Meridian inference substitutes for INT8 qualification.
TAEH3 temporal and standalone2D tiny previews observe LOW x0 with request-scoped cancellation.
Existing wiring, defaults and accepted seam algorithms stay unchanged. NA seam ratings are not passes;
exact voice identity, word timing and universal16GB performance are not guaranteed.
The reproduced Core same-process LoRA-residency difference remains a known limit.
Restart ComfyUI yourself after updating; see [release notes](RELEASE_1.85.0.md).

**Completed H16 source update (2026-09-17; version remains1.84.0).**
Use the native VAE loader for official ConvRot INT8; add a standard H3→LTX video
LATENT adapter and Tao two-request streaming through the existing Prepared nodes.
Four bound groups/eight clips passed human review; cancel/fault recovery into
ordinary H3 also completed. See [setup, wiring and limits](H16_SOURCE_UPDATE_20260917.md)
and the [conversion/save template](../examples/workflows/35-h3-ltx-latent/README.md).
These remain EXP routes requiring external assets, not automatic download,
one-click refinement or a universal16GB guarantee. H16-3 and Meridian are excluded;
accepted dual sampling, Topaz, Sol/Sage and saved workflows stay unchanged.
Restart ComfyUI after updating. No new tag or Registry release is created.

**v1.84.0 (2026-09-17): Semantic Bridge / BUNNY and the accepted dual-sampling workflow.**
Two nodes support ordinary conditioning, Prompt Relay and independent LOW/HIGH
loop configurations. Download [lossless ComfyUI wrappers](https://huggingface.co/t8star/Semantic-Bridge-Comfy)
under `ComfyUI/models/semantic_bridge/t8_compat/`. Start at0.10; disabled/zero bypasses
unchanged. Do not stack bridges or apply again after Relay binding. Saved workflows
without the new optional inputs are unchanged. Restart ComfyUI, then use the
[five examples](../examples/workflows/34-semantic-bridge/README.md) and
[wiring/parameter guide](SEMANTIC_BRIDGE_EXP.md). This is not a LoRA or speedup
node. The repaired two-segment8s recipe was explicitly accepted: LOW640x320,
HIGH896x448, independent4+learned-upscale+4, both Bridges at0.10.
Use the [accepted Advanced workflow](../examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json).
Other inputs, singing and reference voices still require independent review.
H16/Meridian are excluded; see [release notes](RELEASE_1.84.0.md).
Registry availability is established separately by the publishing service.

GitHub source update (2026-09-17, no new Registry release): merge the completed repairs
below while retaining published Topaz, FastH3 V2 and live Qwen cache-identity fixes.
Unfinished research and local handoffs are excluded. See [source sync and deployment
gates](GITHUB_SOURCE_SYNC_20260917.md). Historical CPU/GPU evidence remains scoped;
this is not a full-repository or universal quality certification.

The user-authorized [customized Chunked workflow](../examples/workflows/13-latent-upscale/2026-09-17_H3_Chunked_4plus4_接线修正版（低显存双分块双采样）.json)
is included as EXP. Its Ref2VA/Turbo0.7/KJ/Sol/LoRA stack and numeric choices are retained;
dependencies, actual2x geometry and duration are documented on the canvas.
Schema qualification is not third-party kernel or full-video quality certification.

Interface repair (2026-09-17): SelfLift now executes real LOW
checkpoints, producer identities, independent HIGH MODEL and per-stage TST/EAV/Prompt
Relay. The five separately recorded interface failures are repaired, not ignored.
Existing defaults and accepted seam recipes are unchanged. Preserve selected Sage/Sol/
LoRA patches, warn about unverified compositions and propagate real errors. Final affected
CPU regressions passed792 cases with2 conditional child-worker entry skips; this is not
new pretrained-model GPU quality review. Saved EAV, TST and Guide Mean graphs pass actual
Core validation. Restart ComfyUI to load the local
changes; see [scope and regression evidence](SELFLIFT_INTERFACE_REPAIR_20260917.md).

Policy update (2026-09-17): owned nodes no longer deny execution solely
because LoRA, KJ Sage, SageAttention, Sol, or callable patches are present or unverified.
Preserve/delegate existing patches and report incomplete coverage; users assume composition
risk. Real input/kernel errors and receipt/cache integrity remain enforced. Opaque stacks
cannot reuse portable caches. See [patch-stack policy](PATCH_STACK_POLICY.md).
This is not universal quality qualification or a migration of existing sampling recipes.

**v1.83.0** (2026-09-17) adds three independent FastH3 V2 EXP nodes: full student checkpoint, exact eight-step AV recipe, learned-gate VSA, authenticated T8 memory bridges, and dual MODEL4+upscale+4 loops. Bound trained/template samples and both eight-second loops were accepted in their recorded review scopes. Repaired Dense Sol B was explicitly accepted; original quiet A is not recommended. [Six accepted control workflows](../examples/workflows/10-speed/FAST_H3_V2_README.md) passed actual native frontend/API round trips. Artificial [frame-count ceilings](FRAME_LIMITS.md) are removed, while alignment, context and actual resource constraints remain. Existing nodes and workflows are not migrated. In the fixed832x480/73-frame cold/hot pair, old EMA-B graphs took103.35/99.90s versus trained V2's78.20/71.92s. Only this pair was faster, with no lower whole-card VRAM observed; not identical models or quality equivalence. See [release notes](RELEASE_1.83.0.md) and [models, wiring and compatibility boundaries](FAST_H3_V2_EXP.md).

**v1.82.0** (2026-09-16) hardens Core H3 VAE compatibility, speech/timeline boundaries, cache identity, the PDD lifecycle, and frontend workflow round trips. It adds one KJNodes-independent dual-model 4+4 long-video workflow whose LOW and HIGH paths each use T8 `LowVRAM(head_chunks=4)` plus `ChunkFFN(chunks=2)` while retaining Prompt Relay. Real two-segment/eight-second GPU mechanical runs passed. The user rejected `head_chunks=1` and the former hard HIGH boundary, then found the accepted-picture plus three-cell HIGH ramp much better but still saw a slight color jump. Color Match V2 was already active; the remaining spike was a short dark/bright oscillation in continuation frames2–3. An append-only `bounded_spatial_temporal_exp` mode now stabilizes only low-frequency RGB means over the first12 continuation frames, with no frame blending, geometry change, or audio edit. Existing workflows remain on `bounded_spatial_v2`. A CPU reuse A/B reduced the measured peak by about85.4%. The separate `bounded_motion_color_exp` targets motion-confident local low-frequency outliers. The user accepted C for release with slight residual seam-color changes documented as a known limitation. The new recommended workflow saves the accepted 2:3 first-frame control composition (256x384 to512x768, h4+c2,4+4,8s); old workflows and node defaults are not migrated. See the [local-color scope](MOTION_COLOR_EXP.md). No universal 16GiB, memory, speed, or quality claim is made. See the [release notes](RELEASE_1.82.0.md).

**v1.81.0** (2026-09-15) adds two independent H3 low-memory EXP nodes without a KJNodes dependency: head-grouped attention with early intermediate release, and conditional packed-token SwiGLU chunking. In one fixed three-second run, the default `head_chunks=4 + chunks=2` reduced observed peak use by 306.73 MiB (2.06%) while adding about 2.92% wall time; this is not a universal memory or speed claim. New Speech Studio reference-voice graphs conservatively trim low-energy alignment padding while preserving explicit settings in old workflows, and the Sol verification probe now follows the current `sink_blocks` interface. See the [release notes](RELEASE_1.81.0.md).

**v1.80.0** (2026-09-14) adds SelfLift LOW4 → learned latent upscale → HIGH4, independent LOW/HIGH MODEL branches, TST, and the in-node long-video composition, together with nine bound-sample-reviewed SelfLift/TaoMate EXP workflows. EAV now defaults to the reviewed `tau=8` over 15%–90%; users may raise it gradually for stronger motion but must recheck identity, deformation, flicker, and audio. Qualification is limited to the bound short samples and is not a universal speed, VRAM, or quality claim. See the [release notes](RELEASE_1.80.0.md).

**v1.79.6** (2026-09-14) closes the official non-Starlight Topaz scope: all 14 regular and 5 interpolation dropdown models completed real official production-worker runs. It adds an explicit 10-bit SDR to HEVC Main10 path, GPU-index selection, and a wired upscale-then-interpolate workflow, while fixing lossless-master output auditing. HDR, VFR, interlaced and rotated inputs are still refused rather than silently normalized. Commercial weights remain local and are not shipped. See the [release notes](RELEASE_1.79.6.md).

**v1.79.5** (2026-09-14) hardens Topaz delivery: manual controls are filtered by each model definition; official network templates prevent false missing-weight errors for Rhea, Nyx XL and Theia; audio is preflighted into MP4 or MKV before GPU work; unqualified 10-bit/HDR inputs cannot be silently reduced to 8-bit; and ComfyUI receives frame/audit progress. Apollo completed a real short mechanical run, with visual quality still pending. See the [release notes](RELEASE_1.79.5.md).

**v1.79.4** (2026-09-14) fixes the regular Topaz route that produced a 12.9GB lossless MOV from a 15-second clip and then exhausted the disk through a redundant SaveVideo. The node now directly saves a high-quality H.264 NVENC MP4 by default; lossless is an explicit audit-only profile. It adds model-use guidance, auto/manual parameters with visible controls, and a separate official `tvai_fi` 2x/4x interpolation node/workflow. Topaz need not remain open, but it must be licensed and the selected model must already be downloaded. See the [release notes](RELEASE_1.79.4.md).

GitHub main update for 2026-09-13: the specific LTX clip, recovered five-second Tao dialogue clip, and latest [eight-second Dance accepted-picture LOW-context contrast](DANCE_ACCEPTED_PICTURE_20260913.md) passed full human review. The accepted Dance workflow preserves 4+4, HIGH and original music; do not regenerate accepted clips. Older Dance and both [depth experiments](DEPTH_REFERENCE_EXP.md) remain failed and are not shipped as recommended workflows. Tao's original finalization guard failure is retained; accepted media does not qualify fresh packaged-worker end-to-end GPU reliability. See the [original seam safeguard](DUAL_MODEL_SEAM_FIX_20260913.md) and [resource, AdaLN, HJL and audio boundaries](LOCAL_RELIABILITY_SCOPE_20260913.md). No universal quality or duration claim.

New in **v1.79.0**: [dual-model4+learned-upscale+4 loops](DUAL_MODEL_LONG_VIDEO_EXP.md), [official regular Topaz upscaling](TOPAZ_EXP.md), and [R1 reliability fixes](R1_RELIABILITY_20260911.md). New templates default to two segments/eight seconds; the reviewed sample was accepted for visuals, audio, lip-sync and seams. Existing workflows remain unchanged. OpenVDN no longer rejects an upstream Sol attention override by presence alone; VDN still uses its own attention math, with no combined acceleration claim. Starlight remains paused and is not a qualified feature. See [release notes](RELEASE_1.79.0.md).

New in 1.77.0: independent EXP workflows for [progressive T2VA/I2VA sampling](../examples/workflows/28-progressive-sampling) and [DLSS 2x frame interpolation](../examples/workflows/29-dlss-fi). Progressive sampling reduced end-to-end time by about 37–38% in the tested three-second warm runs. Reviewed visuals were accepted as comparable, and scored portrait dialogue/audio/lip-sync cases passed; game audio differences remain, and the 32-second route has not passed. Frame interpolation doubles FPS while preserving duration, resolution and the original audio; it is not upscaling or H3 generation acceleration. Separately supplied DLSSG runtime files and dependencies are required. Nothing is downloaded or installed automatically, and existing workflows are unchanged. See the [release notes](RELEASE_1.77.0.md).

[简体中文](../README.md) | English

Implementation files now live under `h3_t8/` to keep the repository homepage short. Workflows remain in `examples/workflows/`; model locations, node IDs, parameters and links are unchanged. The three existing TRT command-line entrypoints still work from the root. See [repository layout](REPOSITORY_LAYOUT.md) for developer details.

MiniMax H3 Audio T8 is a ComfyUI node pack for joint video and audio generation. It includes practical workflows for text and image animation, first/last-frame control, image/video/audio references, long video, lip sync, acceleration, and final-video restoration.

Current version: **1.85.0** · 354 nodes · GPL-3.0-or-later

Version 1.78.1 organizes implementation code under `h3_t8/` for a shorter repository homepage. Existing workflows, model locations, node parameters and the three root TRT commands are unchanged. No model downloads are needed. See the [layout and update guide](REPOSITORY_LAYOUT.md).

New in 1.78.0: an optional [TRT VAE backend](TRT_VAE_EXP.md) and [five workflows](../examples/workflows/30-trt-vae), including local compilation, decoder-only, Full encode/decode and one-sampler same-latent comparison. The combined short-video and static-text review was accepted overall; this remains EXP, without long-video qualification. A separate TensorRT environment and locally compiled engines are required. Existing workflows, generation weights and audio VAE are unchanged.

**No total output-stage speed benefit was measured:** native 30.54s versus TRT 31.33s including loading, decoding and saving, excluding sampling. Faster bare decoding is not a promise of faster complete generation. See [1.78.0 release notes](RELEASE_1.78.0.md).

The previous release's video outpainting and 12 EXP workflows remain available, with joint decoding by default. Results vary by source: some sections can still show strips, repeated textures or visible seams. Try a short candidate first. This is not a seamless-quality guarantee, and the remaining defects have not been proven to be solely model limitations.

