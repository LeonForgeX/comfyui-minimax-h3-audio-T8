# Core compatibility and VDN two-pass refinement

Local 2026-09-17 policy override for all owned nodes: LoRA, KJ Sage, SageAttention, Sol
and foreign callable owners are not admission bans merely because composition is unverified.
Warn, preserve/delegate, and report actual coverage. Real input/kernel errors and own
receipt/cache integrity remain enforced; opaque state requires a fresh execution identity.
See [patch-stack policy](PATCH_STACK_POLICY.md). Historical qualification below is not a
universal guarantee for newly selected combinations.

Version1.75.0 verification record, 2026-09-08. Outpaint remains paused and is not part of this delivery.

## Latest delivery checkpoint — 2026-09-08

The isolated Core/VDN candidate passed its complete CPU suite: **2435 passed, no failures/errors/skips**, 21 warnings. The report checks real isolated Core import paths and unchanged Python sources. The candidate starts from published ad43472 (299 nodes), adds two nodes at299/300 and excludes paused outpaint. It is not the main workspace's full-v7 result (2948 passes and15 still-failing paused-outpaint checks).

All four normalized T2VA/I2VA × VDN/native-H3 workflows have now completed a new actual native sidebar open, Save As and API-download roundtrip. Every execution input/edge matches the expected graph, and16 altered API configurations are rejected. The first tab's failed download attempts remain recorded; reopening the actual saved workflows in a fresh tab of the same browser restored downloads without changing security settings. Evidence: `artifacts/core-vdn-isolated-delivery-20260908/final-api-roundtrip-receipt-v1.json`.

The four JSON files and their bilingual `VDN_TWO_PASS.md` are installed in the main project `examples/workflows/10-speed` and actual `ComfyUI/user/default/workflows/MiniMax H3 T8/10-speed`. Source/user bytes match. Deployment preserves the exact tested JSON graph, adding at most a terminal newline. The user's older10-speed README was missing the existing OpenVDN installation section; that missing text was added after proving all pre-existing text matched. Post-install workflow regression:25 passes, no failures/errors/skips. Evidence: `local-workflow-install-receipt-v1.json` under the same artifact directory. Old single-pass workflows were not replaced.

The isolated candidate has now also passed a trained-model DynamicVRAM run: VDN8+4,320x192→640x384x39, full INT8 base, global Sage, official sparse installed after VDN, reserve5 and the registered isolated saver. The run started only after free VRAM reached14214MiB. It exited0; all mechanical/media/source gates passed, actual logs show DynamicVRAM enabled, and minimum free VRAM was2231MiB. Evidence: `artifacts/core-vdn-isolated-delivery-20260908/candidate-trained-gpu-v1/completion.json`, validation-report SHA256 `382023864ab46caa94e6dd66c5f839fad69e434ebbdbd93fd297003ff394c8de`. Candidate identity is bound by the isolated custom-node layout/config and actual Core import log, not an independent per-object Python-origin probe. The earlier main-project run (minimum2117MiB) remains separately recorded; neither run is an additional human quality verdict.

Both human feedback items are closed: the T2VA classical-music/“你在哪里” pair has normal music, speech and both-side lip-sync with a visual tie; the separate I2VA pair has no ambient noise and a visual tie. No preferred route or universal quality guarantee is inferred.

**Version1.75.0 validation complete within the stated scope.** This version has301 nodes and consistent release metadata. Full regression, trained-model integration, four installed workflows and an official Comfy archive with isolated301-node CPU import pass. The available hardware is RTX4060Ti16GB, not a physical4070TiSUPER or Blackwell certification. Outpaint stays paused. Historical pending descriptions below are superseded by these results; they are not an active to-do list.

**Later same-day update:** the native-H3 I2VA comparator passed at1024x512x73 with EMA B259 targets, native-branch ownership checks and2850MiB minimum free. Its actual first-pass decoded RGB and PCM match the historical VDN-refinement I2VA candidate exactly; final PCM also matches. `artifacts/vdn-native-i2va-comparator-20260908/pair_validation.json` binds both media/report hashes. The user reviewed this pair and reported “没有杂音，画面效果基本一样”: visual tie and no audible ambient noise. The separate `human_feedback_v1.json` binds the exact feedback to the page, pair receipt, media and reports without rewriting the original mechanical evidence. This pair contains no dialogue or music, so neither lip-sync nor music quality is accepted by this verdict. Subsequent sparse-hooks fixes changed runtime sources, so the18/18 matrix and full-v6 below are historical pre-fix snapshots, not a final baseline for the new changes.

**CUDA compilation update:** four serial small-model tests passed using the actual official TorchCompileModel node and inductor: native and multikeyframe paths, each with compilation installed before/after the VDN ownership guard. Session17200 exited0. Reports are bound in `artifacts/attention-hooks-core-compat-20260907/cuda-compile-inductor-summary-v3-20260908.json`. Real compiled regions/kernels, numerical agreement, error/interrupt recovery and ownership restoration were checked. Existing VS2022 developer-shell settings and Python UTF-8 were required; no software was installed and no global environment or Core source was changed. Graph breaks and local eager fallback remain (`fullgraph=False`); this is not trained-VDN branch inference, an all-compiled guarantee or a speed benchmark. Cudagraphs/allocator and broader complete-graph combinations remain open.

**Latest result: the current-Core two-pass input matrix is complete, 18/18 mechanical passes.** On 2026-09-08, session16067 ran only the five remaining pruned cases serially and exited0. All18 report/media receipts and current source/CPU-scope bindings were revalidated afterward. The previous13 were not rerun. New-five minimum free VRAM was3371MiB; the combined18 minimum was3294MiB. Evidence: `artifacts/vdn-two-pass-resume-20260908/completion_summary.json` (SHA256 `e1fcb787accf5adb668d5cb350f8228ac289c0ff2ec5eaab8cc0463012995655`). This completes the input matrix, not human quality acceptance or the remaining compatibility/release gates. No production Python changes or workflow installation were made in this continuation.

**Resumed at the user's request on 2026-09-08.** All 13 completed two-pass receipts were independently revalidated against report/media hashes, 17 runtime sources, Core sources/import paths and six input assets. The interrupted pruned/ref2va attempt20260907-222749 remains unchanged and has no final report. A new continuation under `artifacts/vdn-two-pass-resume-20260908` refreshes the full CPU baseline first, then runs only the remaining five pruned-input cases, strictly serially. No old completed case is rerun. The separate DLSS v1.74.1 patch changed six Python files since full-v5; each matches the published release workspace, and none changes the 17 recorded GPU-path sources. GPU launch still requires the existing 12000 MiB free-memory preflight; desktop applications are not closed automatically.

## What is fixed, and what is still being checked?

**Graph lifecycle follow-up:** four additional CUDA Graph cases passed with an explicit `torch.compiler.cudagraph_mark_step_begin()` in the probe. Each Chrome trace contains56 actual `cudaGraphLaunch` calls; numerical results, interruption recovery and ownership checks pass. Without that explicit boundary, both the short and repeated-warmup controls ran correctly but did not demonstrate replay. Production code was not changed to insert the marker, so default-workflow acceleration remains unproven. Graph breaks and skipped regions remain visible.

The real Core `PromptExecutor` also passed14 allocator lifecycle checks: native and multikeyframe small resident CUDA models, each normal → error → retry → interrupt → retry → error → retry. Each block observed a live native AIMDO graph handle. Core's own execution `finally` removed the active graph, module graph references and prefetch state; successful retries were bit-exact. This establishes the tested cleanup boundary, not VBAR eviction, nonempty offload-allocation replay, full-model memory safety or trained VDN execution. The first fixture attempt failed because AIMDO's lazy Linear weights had no checkpoint; the corrected fixture creates random resident weights before enabling AIMDO for all actual execution. Failed attempts remain intact.

All report/source and trace hashes are verified in `artifacts/attention-hooks-core-compat-20260907/cuda-graphs-allocator-summary-v1-20260908.json`. Fresh complete-suite `current-full-v7` finished:2948 passed,15 failed,0 errors/skips,21 warnings in934.82 seconds. Its independent `scope-qualification.json` verifies all664 focused Core/VDN cases pass, unchanged recorded Python sources and correct Core origins. The15 failures match full-v6 exactly:14 paused-outpaint workflow references to the removed DLSS input, plus one old workflow aggregate baseline whose four changes match the published DLSS patch. The full suite remains failed; this is not release approval. Outpaint was not changed.

The VDN compatibility changes let the global Sage launch option remain enabled and distinguish a normal attention backend from a patch that replaces VDN's algorithm. The two-pass route now runs a complete VDN first pass, learned latent upscaling, and a second refinement pass while retaining the first-pass audio.

The historical full CPU suite and five selected Core source trees pass their recorded checks. Fresh full-v6 completed with 2940 passes and 15 failures, all in paused outpaint workflow support: 14 references to the removed DLSS acknowledgement input and one old aggregate workflow baseline. All 656 selected Core/VDN CPU cases passed; source/origin checks passed. The exact failure set and four published DLSS workflow changes are independently qualified in `artifacts/vdn-two-pass-resume-20260908/scope-qualification.json`. The full suite and release gate remain failed; outpaint is not modified. The user subsequently freed GPU memory (14801MiB available), satisfying the unchanged12000MiB preflight; the five-case continuation is now complete.

All18 single-pass GPU cases and all18 two-pass GPU input cases are complete. Both cover full/pruned bases across nine inputs. The final five two-pass cases were pruned Ref2VA, multiple reference images, reference video/audio, reference audio, and hybrid first-frame/audio. Each retained the complete VDN8 first pass, learned2x latent upscale and VDN4 refinement, with identical first/final decoded audio. This is local current-Core input coverage, not all-hardware, all-combination, human-quality or release acceptance.

For the two existing 1024×512 comparison clips, the user reported **“左右差不多” — no clear visual preference**, then explicitly confirmed **“音乐、人声、两边口型都正常”** on2026-09-08. Classical music, Mandarin speech (“你在哪里”) and both candidates' lip-sync are accepted for this pair. The new `artifacts/vdn-two-pass-0p5mp-human-review-20260907/human_audio_lipsync_feedback_v1.json` binds the exact feedback to the page, both media/reports and the unchanged prior visual receipt. No winner or broader input/hardware acceptance is inferred.

## Attention compatibility policy

| Combination | Behavior | Evidence and limit |
|---|---|---|
| Global `--use-sage-attention` + VDN | Leave the launch option enabled; VDN-specific rows keep exact SDPA | Real current-Core DMD8 and two-pass GPU runs; this does not mean VDN internally uses Sage |
| Core plain backend selector + VDN | Recognize the actual backend wrapper, not its name or a copied marker | CPU identity tests and a real PyTorch-selector-before-VDN run |
| Official BlockSparseAttention + VDN | Remove authenticated sparse replacements only from the VDN MODEL branch, including downstream installation and prepare callbacks | CPU ordering/lifecycle tests and real before/after GPU configurations, including 0.52MP refinement |
| Official sparse + T8 SLA Precision V2, FastH3 VSA, H3-World or basic EAV | Keep the T8 algorithm on its own branch and report the local bypass | CPU ownership checks; broader GPU combinations remain open |
| Official sparse + SPEED | Retain official sparse attention; SPEED changes resolution/sigma transitions | CPU composition/lifecycle checks, not sparse-SPEED visual acceptance |
| Official sparse + FFN Activation Chunk | Keep the attention selected by the official callback and chunk only the token-local FFN | CPU composition and numerical checks; eligible GPU producer path still needs validation |
| Unknown algorithm-changing patch | Explain the conflict without erasing arbitrary hooks or silently changing algorithms | Negative identity and ownership tests |

The global Sage flag does not itself install the per-model `optimized_attention_override`. The previous VDN guard rejected every such override. Compatibility must distinguish normal backend selection from genuine algorithm replacement; turning off Sage globally is not the solution.

The external Sol plugin is a separate integration. Recognizing Core's official sparse factory does not automatically authorize external Sol implementations, compiler wrappers or arbitrary object-level forward patches.

### Other affected paths

- **Native H3 and multikeyframe:** capability adapters cover old/new FinalLayer argument lists, attention arguments and prefetch interfaces. Core revisions and hashes are diagnostic evidence, not the runtime capability selector.
- **EAV + STG:** enabled/disabled combinations, main/weak branches, post-CFG re-entry and cancellation/retry have real-Core CPU checks. With both effects disabled, the original MODEL is returned unchanged.
- **Prompt Relay and EAV/LongVideo composers:** authenticated official sparse state is handled on the model clone. Actual wrapper/binding identity is checked; copied markers and unknown hooks do not become trusted. Independent routed-attention/FETA math and lifecycle tests pass.
- **EAV + external BlockCache:** a non-mutating finalization view adapts the old external wrapper to the current Core sigma/head interface. Small real 50-layer CPU models test full/hit paths, dual PDD heads, audio shifts, both node orders and cancellation/retry. The external plugin was not edited; this does not certify its standalone node or trained GPU output.
- **Attention Hooks:** the oldest selected Core has raw tensors rather than AttentionTensorContainer. That mismatch was reproduced (7 failures/1 pass), fixed through capability selection, and the same eight cases then passed. Hooks/RoPE/math/cancellation tests are included in the current matrix.
- **Compilation:** four small real H3 CPU tests execute captured FX graphs with a counting eager backend, both installation orders and cancellation/retry. Four additional real CUDA inductor cases now pass through the official compiler node, as detailed above. All use `fullgraph=False` and retain graph-break warnings. Allocator/cudagraphs and broader compiler combinations remain separate open checks.

The actual CUDA probe reproduced hook bypass in all six Sol/SLA/VSA x before/after configurations: direct control called QKV/output hooks once; the eligible official chunked producer called neither. Source-bound evidence is `artifacts/attention-hooks-core-compat-20260907/cuda-native-producer-pre-fix-20260908/report.json`.

The fix authenticates the actual Core-injected attention closure and adapts it at the MODEL-local block boundary, including the T8 FFN-chunk path. No-hook calls retain Core's original chunked producer. Hook-bearing calls materialize full QKV because a hook may mix arbitrary token rows; they still call the real sparse kernel and retain VSA tile ordering, live-row lengths, prefix sinks and the learned coarse gate. This may increase VRAM use with hooks; it is not a low-memory equivalence claim. Unknown external attention callbacks are not replaced. Layout/dtype/device-changing QKV hooks are rejected explicitly instead of silently using a different algorithm.

Three v3 extended CUDA reports under that same directory each pass8 cases:188-token padding controls,5197-token multi-chunk controls, and5197-token FFN-chunk composition. Each verifies actual sparse operator dispatch, hook effect/call counts, no-hook bit identity, abort/retry and cleanup restoration. VSA cases exercise padding and fine/coarse variants; removing the real coarse gate changes output. These use small randomly initialized native blocks, not trained-video quality or complete graph/compiler coverage. Fresh five-Core expanded-v15 CPU regression completed with zero failures/errors; actual origins and source snapshots were revalidated after the terminal result. Final full-suite, broader composition and delivery gates remain open.

## Two-pass routes and required assets

Both routes use VDN DMD8 for the first pass, the existing learned 3D latent upscaler, rebuilt high-resolution conditioning and locked first-pass audio. Neither is the previously tested LightX2V 4+4 recipe.

1. **VDN → upscale → VDN:** refine using the VDN stage's own trailing sigma grid. Do not add a separate Turbo/EMA LoRA to this VDN branch.
2. **VDN → upscale → native H3:** use an independent base MODEL with the new EMA B LoRA. Do not send the VDN Composer output into the native-refinement loader. The GPU probe checks every native forward for VDN state and audits 259 EMA targets; that is not a tensor-by-tensor weight-equality proof.

Existing assets are reused; no additional converted or trained adapter has been shown necessary:

- `models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`
- `models/diffusion_models/OpenVDN/vdn-minimax-h3/`, keeping its complete internal structure
- `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`
- `models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors` for the independent native refinement route only
- The usual H3 text encoder, video/audio VAEs and FFmpeg.

The two-pass GUI drafts use the registered experimental SafeAVSave node to encode raw IMAGE/AUDIO and return a reusable VIDEO without another SaveVideo re-encode.

## Verification record

### CPU: current source baseline

| Core source | Expanded-v15 passed | Skipped |
|---|---:|---:|
| Current `fbed745` | 664 | 0 |
| `cf10c5c` | 604 | 60 |
| `e7051b0` | 605 | 59 |
| `86aedfd` | 604 | 60 |
| `563b98e` | 583 | 81 |

All five have zero failures/errors, correct actual import origins and unchanged recorded Python source hashes. Missing historical APIs produce explicit skips, not inferred passes. Historical Core sources use this host's current dependencies, not reconstructed historical package installations.

The expanded-v15 reports/XML are under `artifacts/core-contract-matrix-20260908`. Historical `current-full-v5` had2947 passes; later full-v6 had2940 passes and15 paused-outpaint failures. Current full-v7 has2948 passes and the same15 failures, independently qualified as above; an isolated delivery candidate still requires qualification. The18 historical GPU report/media receipts are unchanged; of their17 recorded runtime sources, only `vdn_attention_compat.py` changed (addition of the authenticated attention-closure kind). This difference is recorded explicitly, not hidden by rewriting old reports.

### GPU: available hardware and scope

All real runs here use **RTX 4060 Ti 16GB**, serially. Installed Core is `fbed745c8d7d62573b099cd61fe51cb64b9b807e`. They do not establish 4070 Ti SUPER or Blackwell compatibility.

| Evidence | Completed scope |
|---|---|
| `vdn-current-nine-input-matrix-20260907/completion_summary.json` | All 18 full/pruned × nine-input single-pass cases; 320×192×39, DMD8, global Sage, legacy VRAM reserve5, registered SafeAVSave; minimum free VRAM across the batch 2802 MiB |
| `vdn-current-two-pass-nine-input-matrix-20260907/progress-05.json` | Original first five full-base two-pass cases; later sixth attempt interrupted without a final report |
| `vdn-current-two-pass-nine-input-matrix-resume-20260907-v2/progress-13.json` | 13/18: all nine full-base inputs plus pruned T2VA/I2VA/L2VA/FL2VA; all 13 receipts revalidated on 2026-09-08 |
| `vdn-two-pass-resume-20260908/completion_summary.json` | 18/18: previous13 revalidated plus five new pruned cases; VDN8+4,320x192->640x384x39, Sage and registered saver; first/final PCM identical; current source and scoped CPU bindings checked; full-v6 still has15 paused-outpaint failures |
| `vdn-old-core-cf10c5c-sage-two-pass-20260907/20260907-182527` | Exact historical cf10c5c source, current host packages, Sage, VDN8+4, 640×384×39, registered saver, 3097 MiB minimum free; one old-Core configuration, not the entire historical GPU matrix |
| `vdn-core-compat-sage-sparse-before-legacy-vram-20260907/20260907-143858` | Official sparse before VDN, Sage, VDN8+4, 640×384×39; 1156 MiB minimum free |
| `vdn-two-pass-0p5mp-safe-review-20260907/20260907-155312` | VDN8+4, 1024×512×73, isolated raw-AV encoding, identical first/final decoded PCM; 1275 MiB minimum free |
| `vdn-native-refine-0p5mp-safe-review-20260907/20260907-160134` | VDN8 + independent native4, 1024×512×73, all four native forwards checked, no VDN branch state, 259 EMA targets; 2895 MiB minimum free |
| `vdn-i2va-two-pass-registered-save-v2-20260907/20260907-174246` | I2VA, VDN8+4, 1024×512×73, Sage and official sparse before VDN, registered SafeAVSave, identical first/final PCM; 2830 MiB minimum free |

Each accepted mechanical report checks adapter shapes, absence of LoRA runtime errors, actual output dimensions/frames, strict H.264 decoding, finite nonempty audio, audio preservation, resource margin and recorded project/Core source identity. These checks do not grade naturalness, lip-sync or music quality.

The resumed batch independently revalidated the original five reports/media, 17 runtime sources, five Core sources and six inputs. The six Python changes between full-v4 and full-v5 were confined to the audited Hooks/shared-controls/tests/runner changes and do not occur in the previous GPU graphs. The new batch binds full-v5, uses new case directories and stops on any failure. Python runtime sources stay frozen while it runs.

The old-Core GPU archive is verified against all 1063 tracked Git blobs, with explicit source and runtime roots and actual imported-module checks. The earlier CRLF CPU archive remains intact; accidentally inheriting the parent plugin repository's Git HEAD is not accepted as Core provenance.

### Preserved failures and interruptions

- The 0.52MP attempt `20260907-153645` failed strict H.264 reference-frame checks and had only 280 MiB free. It remains failed. The later run encoded new raw IMAGE/AUDIO through an isolated encoder and changed reserve from 4 to 5 GiB; it is not a repair of that corrupt MP4 or a single-variable benchmark.
- I2VA attempt `20260907-163127` stopped during first-pass sampling with no final report. The separately successful `174246` does not retroactively pass it.
- Original two-pass attempt `full/multi_ref_images/20260907-210409` has no final report. Its batch/processes were confirmed absent before resuming in a new artifact root; the exit cause remains unknown.
- Earlier failing CPU regressions and earlier test matrices remain in their original artifact directories. This document summarizes their resolved or still-open outcomes rather than presenting old “running” notes as current state.

### Workflow and human review

The four v2 drafts in `artifacts/vdn-two-pass-workflow-drafts-20260907-v2` share one prompt and one existing Duration Planner across both conditioning lengths and both output trim durations. At default values, complete resolved API graphs equal their original four-route baselines; the fixtures are included in formal tests.

All four completed actual native sidebar open, Save As and API-download checks in an isolated user directory:

| Route | Execution nodes | Checked edges |
|---|---:|---:|
| T2VA → VDN refinement | 33 | 61 |
| T2VA → native refinement | 37 | 63 |
| I2VA → VDN refinement | 35 | 64 |
| I2VA → native refinement | 39 | 66 |

Each rejects four altered configurations. Receipts are in `artifacts/vdn-workflow-native-roundtrip-20260907-v2`. These checks did not install production workflows or queue inference. The file-chooser route is not verified; actual sidebar serialization and saved API files are.

The 0.52MP public comparison has identical first-pass RGB and decoded audio across the candidates; final images differ. Actual browser playback and single-audio routing were checked, then the user reported no clear visual preference. This feedback does not choose a winner or establish audio/lip-sync acceptance for unreviewed cases.

## Delivery checklist and explicit limits

| Scope | Local outcome |
|---|---|
| P0 old/new interfaces | Capability-based adapters; five real Core source matrices pass their available APIs; missing historical APIs are explicit skips |
| P1 attention ownership/backend | Global Sage stays enabled; known ordinary selectors compose, authenticated algorithm overrides are isolated per MODEL; unknown patches are not erased. Exact-SDPA selection has CPU policy tests and available Ada GPU evidence, not physical Blackwell proof |
| P2 affected nodes | Native/multikeyframe, SLA/VSA, hooks, SPEED, EAV/World, FFN/prefetch, audio/long-video/Face Refine and saver/DLSS are covered by the source-bound regression. Eligible sparse hooks, bounded compiler/graph/allocator paths have separate real CUDA evidence. This is not exhaustive trained inference for every cross-plugin combination |
| P3 two-pass | Existing learned latent2x, high-resolution conditioning, VDN8+4 and independent native4 routes; retained first-pass audio. Two reviewed0.52MP pairs accepted within the limits above; no new model conversion or universal improvement claim |
| P4 validation |18/18 single-pass and18/18 two-pass input cases; an actual old-Core Sage VDN8+4 run; current candidate2435-test full suite plus trained DynamicVRAM integration. Earlier failures and source snapshots remain intact |
| P5 delivery | Four native UI/API-verified JSON files and bilingual usage notes installed to source/user folders;25 post-install tests;1.75.0 metadata, official archive and isolated301-node import. No commit/push |

The remaining unverified areas are physical4070TiSUPER/Blackwell, every full-model offload/compiler/external-plugin combination, exhaustive long-video stress and general quality improvement over single pass. These are not silently certified by the current bounded checks. Optional future extensions need their own tests; repeated inference of the already completed matrix is not required unless its affected runtime code changes.

The Comfy configuration check passes with one retained S102 security warning for a test fixture executing pinned historical Core source; the fixture is excluded from the runtime archive. No warning suppression or test weakening was used. Final local receipts are in `artifacts/core-vdn-isolated-delivery-20260908/package-v2`; the archive contains the compatibility document despite the normal development-doc exclusion.

The user authorized GitHub publication on2026-09-08 after local acceptance. Outpaint remains paused and is excluded from this version.

Historical text removed from this summary is preserved locally in `artifacts/core-vdn-doc-consolidation-20260907/compatibility-before-consolidation.md`. Raw reports and media were not rewritten.
