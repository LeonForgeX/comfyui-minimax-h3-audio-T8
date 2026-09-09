# H3 Video Outpaint — implementation and acceptance plan

## 2026-09-09 release scope supersedes historical gates below

The user explicitly requested ending this iteration and publishing the current workflows with known visual limitations. v1.76.0 retains joint_decode as the primary output mode and preserve_source as an option. The real 32-second/22-window output completed, but visible strips and seams remain; this is not an all-material quality pass or proof of a model-only cause. The isolated 6.6-second strict-mask experiment is excluded. Further GPU generation and remaining visual-improvement gates are deferred, not marked passed. See [current source-mode notes](OUTPAINT_SOURCE_MODES.md).

Requested on 2026-09-06: implement the complete proposed integration without changing existing workflows.
Baseline: T8 v1.74.0, commit `91c1b4e9b680d07a6eacee6a3aa6b449a4697554`, 299 nodes, 207 workflows.
Reference: TwoAbove/ComfyUI-H3VideoOutpaint v0.4.0, commit `27df0ff0538896dd494e3541c5a374cf2fe00aab`, MIT.
The reference checkout is isolated under `artifacts/upstream-h3-video-outpaint-27df0ff`; it is not a globally installed custom node.

## Full scope (not reduced by partial implementation)

The sections below are a chronological engineering log. Statements such as
"unregistered" or "pending" describe the checkpoint in which they were written;
the current state is summarized first here.

### Current release-candidate checkpoint (2026-09-07)

**2026-09-09 superseding status:** the September7 material below was not finally
human-accepted; the source-preserving seam fix remained visibly discontinuous.
The user accepted the short joint-decode comparison and selected joint decoding
as the new primary, with exact-source preservation optional. All51 outpaint CPU
test files now pass590 tests with a source/fixture/workflow freeze. The207-file
baseline guard keeps the original baseline and permits only the exact4 DLSS
updates already published in3769d70, verified against immutable Git objects;
no legacy workflow was changed by outpaint. All12 new live-schema workflows
passed native sidebar open/save/API comparisons in an isolated user directory,
including the compact subgraph's15 public widget bindings. Public installation
is still pending. The fresh768-frame/22-window joint-decode run under
`artifacts/outpaint-v4-joint-full32-20260908` stopped when Windows self-crashed
and rebooted on September9 at01:42/01:46 (0x1E/C0000005). Five committed
payloads passed read-only CPU integrity/shape/finite checks; no final video exists.
GPU stays paused pending crash investigation and a safe recovery decision.
The old running report is historical, not a live process. Subsequent candidate
probe mode/geometry inheritance fixes passed64 focused tests including16 new
cases; the590-test receipt predates that tool-only fix. No full32 quality or final
release acceptance is implied. Older full-repository/package receipts are historical
and must not be presented as verification of the new seam-fix revision.

September9 CPU review-tool update: the regional first-frame builder now requires
`--plan <sealed-plan.json>` and obtains top/bottom crops from its validated output
source rectangle. It checks preview/manifest/payload bindings and paired sampling,
pixel policy, VAE, post-processing, recorded base-weight files and base MODEL graph.
Only the known outer regional router is removed from that graph comparison; its
composed MODEL hash is expected to differ. This is not proof of identical full
live model states. The specialized page requires both top and bottom expansion
with no side margins; use the general material-review path for other geometries.
32 focused tests and an unchanged historical preserve-source pair pass CPU checks;
the current review revision still needs browser qualification. No new joint regional
sampling or human acceptance has occurred, and the historical tie selects nothing.

- The proposed integration is implemented as 17 append-only EXP nodes at
  registration positions 299-315. The previous 299 IDs and schemas retain their
  positions. The existing 207 workflow files remain byte-identical under the
  v1.74 baseline guard.
- Twelve released EXP workflows now live in `examples/workflows/27-video-outpaint`
  and are mirrored to the normal ComfyUI user workflow directory. They cover
  geometry preview, expanded and compact generation, ordinary and region-guided
  candidate review/confirmation/continuation, completed-cache recovery, optional
  DLSS-NR 2x and a model compatibility audit. Source and user copies have equal
  SHA-256 values. Promotion from the browser-saved drafts changes only labels and
  status; IDs, nodes, modes, links, native subgraphs and widget values are locked.
- Real Stock20/native-noise tests now cover the selected 32-second 22-window
  continuation and decoder-safe VAE-only recovery, a two-shot hard cut with no
  cross-shot context, a moving game scene with fixed subtitles, and a 124-frame
  speaking clip with exact AAC packet/timeline and decoded PCM/timeline retention.
  The three latest learned outputs pass strict one-thread and bounded four-thread
  decoding, source-region pre-encode RGB receipts, black/freeze screening and the
  default-on shot-reset Color Match report.
- The odd-height 736x607 learned output uses exact yuv444p High444 geometry and
  played to completion in the in-app browser. The completed outpaint plus DLSS-NR
  2x route also passed its strict mechanical media contract. Neither result is a
  blanket player/quality guarantee.
- `pytest -k video_outpaint` passed 523 tests after formal registration and
  workflow normalization, with 2214 deselected and five existing dependency
  warnings. The subsequent full repository rerun passed 2737 tests with six
  existing dependency warnings. Packaging, registry and publication remain
  release gates.
- A real learned cancellation/restart gate now passes on the two-window hard-cut
  material. The first owned ComfyUI process was interrupted only after free VRAM
  fell to 8712.914MiB and real sampling had begun. It returned
  `execution_interrupted`; the manifest retained exactly the selected first
  window and became `interrupted`. A fresh PID resumed the same cache to 2/2
  windows. Both window-asset hashes match the uninterrupted learned reference,
  and the final MP4 plus decoded video/audio frame hashes exactly match a
  current-code uninterrupted control. Minimum observed free VRAM was
  2312.852MiB. Evidence is under
  `artifacts/outpaint-hardcut78-interrupt-resume-learned-v3-20260907`.
- `comfy node validate` passes when the Windows CLI is placed in UTF-8 mode. Its
  default GBK console can raise a display-only `UnicodeEncodeError` while
  printing the green check mark after validation has already passed; the UTF-8
  rerun exited zero. The final post-tooling full repository run passed 2739
  tests with six existing dependency warnings and zero failures. Version metadata
  is synchronized at v1.75.0.
- The final official package candidate contains 599 files and is 2,988,821
  bytes, SHA-256
  `F673CF4359D2FE559FEB47FE9926D7BE24F4DD7DE5E43BE5755AC7C3BC6A06E6`.
  All 42 outpaint runtime
  Python modules and 12 outpaint workflows are present. No model, media,
  executable, nested archive, artifact, test, tool, `docs`, `SKILL.md` or
  `roadmap.md` entry is present. Its isolated extraction imports 316/316 unique
  node schemas and finds all 219 workflows at v1.75.0. This is a local release
  candidate only; publication remains pending.
- Human perceptual acceptance is still required for the supplied 32-second,
  hard-cut, game/subtitle, speech/audio, odd-size, DLSS and regional A/B material.
  The region-guided full continuation will run only after the reviewer explicitly
  chooses a generated regional candidate; the software must not auto-select it.
- The final review hub and every linked page were opened in an actual Chrome
  session. The 32-second, hard-cut, game/subtitle, speech and odd-height media all
  visibly advanced. The DLSS page initially exposed a review-only synchronization
  bug: per-animation-frame hard seeks kept the larger right-hand stream on its
  first frame. Its drift correction now uses bounded rate adjustment and at most
  one hard correction per second; the displayed master/right timestamps stayed
  within 0.02 seconds during the browser check. Material and hard-cut review page
  generators now mute the source-side video and label the outpaint side as the
  only audible track, preventing two identical but slightly offset audio tracks
  from being mistaken for generated noise. Four focused generator tests and Ruff
  pass. The subsequent full repository rerun also passed 2739 tests with the same
  six existing dependency warnings and zero failures. These changes affect review
  tooling only, not runtime nodes or workflows. The included feature record was
  refreshed, so the official package was rebuilt and its isolated import/audit
  repeated with the hash above.

No new outpainting model was created. The released workflows use the existing H3
FL2VA checkpoint, Qwen3-VL encoder, video/audio VAEs and the separately installed
ComfyUI-KJNodes H3 low-memory Attention/FFN pair. Turbo, SPEED, SLA, OpenVDN,
FastH3, Prompt Relay and unknown MODEL/Attention owners currently fail closed in
the compatibility audit rather than being advertised as tested combinations.

### Completed-selection recovery and decoder-safe output checkpoint (2026-09-07)

- The explicit learned candidate was continued from its one-window paused state to
  all 22/22 committed sampling windows. The selected first-window asset stayed
  byte-identical (`84b35363aa51fc43559aa16ad2c89c23029b36ba7b82891090960ef622902d21`)
  and the Comfy graph reached `execution_success`. The original inter-frame H.264
  publication was nevertheless rejected: strict FFmpeg decoding failed
  nondeterministically with CABAC/reference/macroblock errors. The failed file,
  report and immutable sampled cache are retained under
  `artifacts/outpaint-selected-continuation-20260907`; this is failure evidence,
  not an accepted deliverable.
- Repeated controls showed that uncontrolled automatic-thread decoding is itself
  unstable on this machine: the unchanged input failed 15/20 times with both
  FFmpeg 7.1.1 and the ImageIO FFmpeg 4.2.2 build. Therefore the gate was not
  relaxed to a lucky retry. Outpaint encoding now uses one x264 worker, one
  reference, no B-frames, keyint/min-keyint 1, scenecut 0 and CABAC off. Even
  yuv420p outputs use Constrained Baseline; exact odd-size yuv444p outputs use
  High444 with the same all-intra structure. Validation retains one strict
  single-thread decode and adds three strict fixed-four-thread decodes.
- A real no-diffusion recovery graph loaded the explicit completed selection and
  the unchanged 5.2GB video VAE only; it contained no UNET, CLIP, KJ patch,
  sampler or continuation node. It preserved the old 22-window manifest and the
  selected first-frame RGB, then published
  `artifacts/outpaint-safe-recompose-20260907/output/T8_H3_Outpaint/selected_test32_safe_00001.mp4`
  (SHA-256 `B837FC50DD1ADE76353A68752D7599C89BF7106602DDF64D243846253E6C1755`).
  The output is 736x608, 768 frames at 24fps/32s with the original 32kHz stereo
  AAC. Packet bytes/timestamps and decoded PCM/timestamps match the source;
  strict decode passed 20/20 with one thread and 20/20 with fixed four threads.
  Minimum free GPU memory was 7126.098MiB. This closes mechanical/media recovery
  only; human visual acceptance is still pending.
- The unregistered `MiniMaxH3VideoOutpaintLoadCompletedSelectionT8` node now
  rejects incomplete caches and returns a fully sampled explicit selection for
  Compose without reloading diffusion. The workflow builder has a fifth,
  independent "Save Completed Selection" recovery route. All five drafts were
  built from the fresh 8196 live schema, opened and saved in native ComfyUI.
  The read-back audit preserved 37 execution nodes, 33 edges and 98 widget
  values; the fifth route preserved its 6 nodes, 5 edges and 16 values and the
  queue stayed empty. Evidence:
  `artifacts/outpaint-candidate-ui-five-20260907/browser_roundtrip_five_final.json`.
- Targeted runs passed 39, 14, 13 and 29 tests for the encoder/media, bounded
  decoder checks, completed-selection node/probe and combined workflow areas;
  these are distinct focused runs, not one full-repository regression. Production
  remains at v1.74.0 / 299 nodes / 207 workflows. These five are accepted as
  isolated UI drafts, not yet installed user workflows. Registration, advanced regional
  prompts/person protection, scene/color material validation, acceleration and
  upscale combinations, odd-size learned playback, additional resume/cancel and
  multi-material tests, full regression, human review and release remain open.

### Archive UI restart and learned candidate checkpoint (2026-09-07)

- Actual CPU ComfyUI jobs verified archived PNG display, rejection with the
  confirmation switch off, explicit test confirmation, and copying the complete
  64-character selection ID from native PreviewAny into LoadSelection. The
  isolated server was stopped only after verifying ownership and empty queues;
  a fresh process loaded the saved workflow and restored the same image.
- All four displayed RGB images (including the expected-error job's upstream
  preview) equal the archived preview hash. The original sampled-prefix manifest
  is unchanged after restart. This uses a clearly labeled tiny CPU oracle fixture,
  not learned image-quality evidence. Evidence is in
  `artifacts/outpaint-candidate-ui-20260907/cpu_archive_fixture/browser_before_restart.json`
  and `browser_after_restart.json`.36focused UI-evidence/workflow tests pass.
- A separate actual learned-model candidate test is now successful:
  `artifacts/outpaint-learned-candidate-20260907/report.json`. It reopens copied
  source/audio/text caches for the32second736x416source, then generates only the
  first56frame window at736x608 / Stock20 / native noise / default-on color match.
  The first AV window file SHA exactly equals the old uninterrupted test's first
  window: `84b35363aa51fc43559aa16ad2c89c23029b36ba7b82891090960ef622902d21`.
  The candidate ends in normal paused state with one committed window. Preview
  PNG and decoded RGB hashes pass.198.187seconds including server startup and
  execution, minimum free GPU1787.305MiB; no full-clip or human acceptance claimed.
- Candidate ID: `f394dde92b40494d05c723dac27cf3c7f42d0a50105d144000ed307b72c03b3d`.
  The learned PNG was visually inspected before explicit operator test selection.
  `tools/run_video_outpaint_selected_probe.py` now starts a separately guarded
  continuation test from a copied paused candidate, not a regenerated first
  window. It uses the real Select/Continue/Compose chain and preserves the
  selected first-frame RGB guard. Its `--confirm-test-selection` flag is a
  technical test choice, never the user's visual acceptance.7probe-graph tests
  pass; real32second continuation is in progress under
  `artifacts/outpaint-selected-continuation-20260907`; inspect its live process
  and terminal report before citing completion. Old original22windows untouched.
- Production299nodes/207workflows/v1.74.0 remain unchanged. Advanced regional
  prompts/person protection, combinations, remaining material/geometry/cancel
  matrices, codec stability, full regression, user review and release remain open.

### Independent candidate workflow checkpoint (2026-09-07)

- Four independent drafts now cover Generate Candidate, Review Candidate,
  Confirm Candidate and Continue Selection. Review and Confirm contain no
  MODEL/CLIP/VAE loaders or sampler. Confirmation defaults off; Continue takes
  an explicit saved selection ID. Native PreviewAny displays the generated
  candidate ID or confirmed selection ID for copying; actual populated-ID
  usability and live archive-image reload in the browser remain pending.
- `tools/outpaint_candidate_extension` registers the12draft classes only in a
  separate CPU test server on8192. Production registry and207workflows are not
  changed. The previous geometry test server on8191 and its images remain intact.
- Builders use live object_info, not invented schemas. Browser inspection found
  that this installed frontend adds an after-generate control for an input named
  seed even when the V3 schema omits that option. An initial draft removed it and
  shifted steps/resume. That draft was never queued. The corrected serializer
  preserves `fixed`,20steps,resume=false,color_match=true. No global frontend or
  old workflow serializer was modified.
- The final four drafts in
  `artifacts/outpaint-candidate-ui-20260907/workflows-v3` were each opened and
  saved with native ComfyUI. All82widget values and28edges match the API graphs.
  Evidence: `artifacts/outpaint-candidate-ui-20260907/browser_roundtrip_final.json`.
  A read-only auditor rejects missing named widgets, shifted values, changed
  types/modes and rewired edges.25focused builder/auditor/previous-workflow tests
  passed with5existing warnings; `workflow-final-tests.xml` in the same root.
- This is import/save correctness only. No inference was queued in this CPU
  service. Actual copied IDs, archive review/confirmation execution, real GPU
  candidate continuation, advanced features, multi-material long clips, codec
  stability, human review and release gates remain open. Use V3, not the retained
  earlier diagnostic drafts. Prompt/VAE/text-encoder changes require a fresh
  run_name/preparation cache, not merely a new candidate name.

### Prepared-cache reload checkpoint (2026-09-06/07)

- `video_outpaint_prepared_reload.py` and the unregistered LoadPrepared node now
  accept the same inspected source/Plan/run_name and reopen saved video, audio
  and conditioning providers. No CLIP, VAE or diffusion model is needed for this
  stage. It reuses the saved prompt embeddings, not newly edited prompt text.
- Read-only subclasses reject constructor attempts to create missing source or
  audio manifests. All three metadata files are required before opening stores;
  video chunks are validated separately, audio assets and each conditioning shot
  are checked, and all metadata bytes are compared again at the end. Cancellation
  is checked between video chunks, audio assets and conditioning shots.
- Existing source/audio/text implementations and their cache identities remain
  unchanged. Subsequent Continue still verifies the actual MODEL, and Compose
  still verifies the actual VAE and selected first-frame RGB.
- Final targeted tests: **13 passed**,5existing warnings,74.21seconds;
  `artifacts/outpaint-prepared-reload-20260906/final-tests.xml`. The fresh-process
  archive worker now uses this production loader, rather than reconstructing
  providers in test-only code. Missing/corrupt/incomplete inputs are not rebuilt.
- A read-only real32second check verified46video chunks,20audio chunks and one
  conditioning shot, with the three manifest hashes unchanged and no model call.
  Evidence: `artifacts/outpaint-prepared-reload-20260906/real32/report.json`.
  After the per-audio/per-shot cancellation addition, the real32-final probe
  passed again in6.078seconds (excluding imports/startup), with identical hashes:
  `artifacts/outpaint-prepared-reload-20260906/real32-final/report.json`.
  Independent UI workflows, copied-ID usability, actual browser restart and
  learned-GPU candidate continuation remain pending; this is not final release.

### Durable candidate/selection reload checkpoint (2026-09-06)

- `video_outpaint_candidate_archive.py` publishes content-addressed RGB PNG,
  candidate JSON and a separate explicit-selection JSON in the existing candidate
  cache directory. Existing mismatching files are never overwritten. No Python
  object, pickle, model weights or arbitrary referenced file paths are serialized.
- Candidate reload takes an explicit SHA ID, verifies PNG bytes and decoded RGB,
  prepared source/text/audio, sampling identity and first AV asset. Missing
  sampling manifests do not create replacements. A candidate reload is not an
  implicit confirmation; selection reload requires the previously confirmed ID.
- Candidate and Select outputs now include their saved IDs. Two extra unregistered
  LoadCandidate/LoadSelection nodes display the stored image without invoking a
  VAE or sampler. Later committed windows do not invalidate the original prefix
  or force candidate regeneration. The original generation entry remains bounded
  to a first-window run; reopen through the explicit archive path instead.
- A fresh CPU child process reconstructed actual source/text/audio store/provider
  objects from existing files and loaded the same selected image after the second
  sampling window had completed. This is stronger than in-memory object reuse,
  but it is not a full ComfyUI browser restart workflow or learned-model test.
- Production reload nodes currently take PREPARED as input. Independent restore
  workflows still need a saved-preparation loader (or explicit revalidation via
  Prepare), readable/copyable candidate IDs and real browser validation. Do not
  describe the entire user-facing restore workflow as delivered yet.
- Final archive/node regression: **23 passed**,5existing warnings,84.78seconds,
  `artifacts/outpaint-candidate-archive-20260906/combined-tests.xml`. Includes
  corrupted metadata/PNG/AV assets, missing manifest, explicit ID validation,
  wrong image/selection settings, cancellation, fresh-process restoration and
  completed-store node reload/zero-window continuation. Not a full-suite rerun.

### Candidate image and native-node checkpoint (2026-09-06)

- `video_outpaint_candidate_preview.py` decodes only the first seven-token chunk
  of the actual sampled prefix and closes the decoder immediately. It extracts
  one frame, uses the final compositor's canvas mapping/source paste-back/color
  correction and RGB8 quantization, then returns a standard one-frame IMAGE.
  Source, source-cache, AV prefix, manifest and loaded VAE identity are checked.
- The candidate image is no longer a geometric placeholder. CPU oracle tests
  compare its bytes with actual raw RGB sent to the MP4 encoder, including
  color enabled/disabled and odd exact output dimensions. These do not establish
  learned image quality or stability on the target GPU.
- `nodes_video_outpaint_candidates.py` adds four unregistered draft classes:
  Candidate (native image output), explicit Select (confirmation off by default),
  Continue (same stored seed/model/provider binding), and candidate Compose.
  Candidate Compose inherits the selected preview's color settings and supplies
  its expected RGB8 SHA to the compositor. A first-frame mismatch aborts before
  publication; normal composition leaves this new optional guard unset.
- Existing registered nodes and207workflows remain unchanged. Independent
  candidate/continuation workflows and actual UI execution are not delivered yet.
  Persistent candidate reload after completing later windows also still needs
  explicit design/testing: the current first-window generation entry rejects a
  store with more than one committed window. Do not silently rerun it or pretend
  a frontend in-memory handle is a durable cross-session selection workflow.
- Preview/node/compose tests: **20 passed**,5existing warnings,150.90seconds;
  `artifacts/outpaint-candidate-preview-20260906/nodes-compose-tests.xml`.
  The tiny-native/injected-backend node chain covers90frames and two windows,
  actual PNG-equivalent IMAGE, explicit selection and real MP4 composition.
  A wrong first-frame hash publishes neither MP4 nor sidecar and keeps the cache.
  These are scoped CPU checks, not the full regression or learned-GPU review.

### First-window candidate protocol checkpoint (2026-09-06)

- `video_outpaint_candidates.py` now uses a bounded store view to run the existing
  serial coordinator for exactly the first window. It returns normally, with
  the durable store marked `paused` when more windows remain. It does not use a
  cancellation exception, discard generated audio context or alter the plan.
- A candidate descriptor binds the plan/model/text/audio/settings identity and
  first AV asset. Explicit selection creates a separate integrity-checked
  receipt. Continuation validates that receipt and reuses the actual selected
  prefix via the existing temporal context mechanism. Scene cuts still reset it.
- `video_outpaint_candidate_execution.py` binds actual live MODEL and text/audio
  providers, rather than trusting a supplied identity. Tests use a tiny native
  MODEL type and injected backend, not learned GPU inference. Candidate image
  decoding, selection nodes/UI, workflows and real GPU checks remain to be done.
- The four files contributing to the previous sampling implementation identity
  are byte-unchanged. A read-only check verified all22old assets, implementation
  SHA `1397feb305b33e8669256b28e6ba37f542b0eb833b6a07fb3d6abdfd76aea915`
  and unchanged manifest SHA `50575f31e93ea09bb0bf3b74897bfefa9a43d3644da65170c50e7ab57ce799fb`.
  No implicit migration, cache rewrite or sampling regeneration was performed.
- Candidate/runtime/native-binding/compose regression: **43 passed**,5existing
  warnings,99.62seconds (`artifacts/outpaint-candidate-prefix-20260906/expanded-tests.xml`).
  This includes distinct generated AV prefix propagation, two noise algorithms,
  exact uninterrupted equivalence, corruption/identity rejection, cancellation,
  scene cuts, single-window selection and old composition regressions. It is not
  the full415-test suite rerun, actual learned sampling or human acceptance.

### Geometry-preview browser checkpoint (2026-09-06)

- An isolated CPU server on localhost8191 registers only the two draft Plan and
  Geometry Preview nodes. Its input/output/temp/user directories are under
  `artifacts/outpaint-geometry-ui-20260906`, not the normal user workflows.
- The generated workflow was opened from the native workflow sidebar. The
  placeholder was replaced with `source32.mp4` through its selector; actual
  browser Run submissions at frames0 and384 both succeeded (4.90s/5.52s).
- Visually checked the original frame and blue checker margins, and the changed
  source pose at16s. Canvas736x608, original rectangle[0,96,736,512]; the
  preview-only104px footer makes the cached PNG736x712, not a changed video size.
- Ctrl+S cleared the unsaved indicator. The isolated saved JSON has source32
  and frame384; SHA256 `8030a1576e2c2d71098a2e3658c4c9feb8d462d3fcb81ecdc40e02d61e2e46ec`.
  Full prompt/status and image hashes: `artifacts/outpaint-geometry-ui-20260906/browser_evidence.json`.
- This verifies one custom-margin preview UI route, not every aspect/anchor UI
  combination, a generated first-frame candidate, or any generation acceptance.
  No GPU test, normal node registration, normal workflow edit, or release occurred.

### Current checkpoint — isolated media inspection (2026-09-06)

- Input properties/full source contracts and final media validation now run in
  separate CPU-only processes. The worker imports the existing validators;
  published DLSS/Skin Finish/Comfy modules are unchanged. Native exceptions or
  abrupt worker exits are reported to the parent, not accepted as a good video.
- Request-bound JSON, rational rate/time-base round trips, bounded metadata/logs,
  cancellation, timeout and observed-owned-descendant cleanup are covered.
  Windows worker GPU visibility uses `-1`; an empty value caused an invalid
  device compatibility error in the initial test and was corrected.
- Focused outpaint plus preflight/registration regression: **415 passed**,
  2179 deselected, 5 existing warnings, 396.19 seconds. Evidence:
  `artifacts/outpaint-inspection-isolation-20260906/focused-tests.xml`.
  This is not a full-repository pass, nor human or all-feature acceptance.
- Read-only actual32second source/output validation reproduced every prior
  internal media/audio/timeline result and the output hash remained unchanged.
  The retained corrupt H.264 control was rejected for actual decode errors.
  Evidence: `artifacts/outpaint-inspection-isolation-20260906/real32/report.json`
  (19.844 seconds). No source, output video, model or workflow was rewritten.
- Previous nondeterministic multithread decoding and zeroed-byte observations
  remain unresolved. Inspection isolation contains one crash boundary; the
  in-process bounded source-frame reader and other native callers are not
  claimed fully isolated. No BIOS/driver/system change was made.
- No task probe remains running. All sixteen requirements below remain in scope;
  generated first-frame selection, regional/person control, combinations,
  remaining native workflow UI, multiple long materials, human review and
  append-only registration/release still need completion. Baseline299nodes and
  207workflows are unchanged. No version bump, commit or push.

### Earlier checkpoint — 32-second final decode failure and diagnostic retry

- The resumed32-second probe has terminated **failed**, despite22/22sampling
  windows saved and minimum free VRAM2920.637MiB. Final strict H.264 decoding
  reported `Reference 4 >= 4` / macroblock decode errors. No final MP4 was published.
  Encoder/mux temporaries were cleaned by the previous implementation, so their
  bytes are unavailable for diagnosis and the failing boundary is not yet known.
  Runtime4812.282s; peak process RSS78404.867MiB/private66663.930MiB. This resumed
  run reused source/audio caches and is not a cold full-preparation memory proof.
- Source-only32second copy/mux and same736x608 padded-source encode/mux CPU
  controls pass the same single-thread strict/explode decode. The latter preserves
  all768video packet records/extradata/timing. These are controls, not the failed
  generated candidate and not evidence of a fix. No upstream dependency upgraded.
- Added separate pre-mux strict validation and private failure-media retention
  (hardlinks and diagnostic JSON, never a deliverable path). Retention failures
  are reported without hiding the media error. A read-only standalone packet
  comparison tool distinguishes packet/codec metadata changes from decode quality.
-55targeted media/mux/preview/workflow tests pass after correcting7invalid
  aspect/margin test inputs (5existing warnings,15.22s). However, broader focused
  regression terminated in native access violation at VideoFromFile.get_bit_depth
  during input inspection. That regression remains failed; native inspection
  isolation/stability still needs work. No full-repository or browser acceptance.
- Current diagnostic retry is **compose-only**, reusing all22sampled windows:
  `artifacts/outpaint-qipao32-compose-diagnostic-20260906`, hidden launcher9612,
  port8191/reserve5GiB/timeout1800s. Logs are in the sibling
  `outpaint-qipao32-compose-background-20260906` directory. Check its actual
  process/report terminal state before doing more tests. Preserve the old failure
  evidence; do not modify captured runtime files while this retry executes.
- All original advanced, multi-material, human, workflow and release requirements
  below remain open. Existing299nodes/207workflows and published version unchanged.
- The compose diagnostic has now terminated **failed** at the new **pre-mux**
  strict validation boundary (260.781s). Audio mux had not run. Both launcher9612
  and server22924 exited; no current probe remains. Retained video is
  `artifacts/outpaint-qipao32-compose-diagnostic-20260906/output/T8_H3_Outpaint/.outpaint-failure-qmldm8sd/video_only.mp4`,
  22866868bytes, SHA`1683734e9daf3de9983548fa0052d45a219d79bd714d80653296d93cf9ee0168`.
  Independent PyAV18/libavcodec62 strict decoding also fails after357frames;
  FFmpeg7without explode (but with xerror) still fails. This is not just a strict
  flag or old-decoder-only observation. Actual encoder/raw-RGB boundary diagnosis
  is next; do not transcode the corrupt H.264 and call that original-RGB recovery.
  Original22sampling windows remain reusable. Broader native inspection crash and
  every original advanced/human/release requirement still need completion.

### Latest checkpoint — 2026-09-06 native-noise reproduction

- Added an **explicit** native-noise mode to the unregistered draft Sample stage.
  It streams the original FP32 CPU video-then-audio RNG sequence in <=1MiB blocks
  into one requested window, replays the shot stream on demand and does not touch
  global RNG. This bounds allocation, not total CPU random-number work. Existing
  coordinate mode remains available; different noise identities cannot reuse the
  same sampling store. No existing published schema/default/workflow changed.
- Real serial probe `artifacts/outpaint-t8-native-noise-p0-20260906` exited 0:
  512x672x90@24fps, stock20/res_multistep/simple, unchanged seed20260808 and source,
  KJ4+4, reserve3GiB, color off. Candidate SHA is
  `C7A580DA76DBAD3B840BA09EF1D854C763AFBE29BA3140F3F889C0A9FA136D72`.
  Strict silent media checks passed; minimum free VRAM593.309MiB, max used15786.691MiB,
  approximately10Hz telemetry, server elapsed313.187s. Not sampler-only speed or
  repeated long-run memory proof. No audio track exists in this source.
- Frame45 and contact frames0/17/34/51/68/85 show coherent grass/flowers instead of
  the preceding orange imagery. Encoded-file SSIM over90frames versus pinned
  upstream is0.928369 whole-frame and0.969288 for just the expanded top/bottom.
  This links the previous fixed-probe mismatch to differing initial-noise inputs;
  it does **not** prove coordinate noise is universally bad, exact learned latent
  parity, normal-speed human acceptance or every extension capability. The old
  failed candidate and all reports remain intact; separate agent_review.json
  records the limited evidence. No full blind verdict was produced.
- Pinned-upstream fake-VAE/fake-DiT input comparison passed aligned90/124frames
  but initially failed80frames and1frame on masks. Fixed source locking to stop
  at the last fully observed native FRAME_PER_TOKEN token; padded/partially
  observed tail tokens are generated internally and never substituted for the
  original RGB source during final paste-back. This is distinct from the90frame
  noise diagnosis. All five input/trajectory cases now pass. The older cut test
  incorrectly assumed padded latents were observed; it now explicitly checks
  13 source tokens and generated padding for each45frame shot.
- Final focused run:327 passed,2179 deselected,5 existing warnings; Ruff and diff
  checks passed. Full repository/release validation remains open. The first new
  oracle invocation from the plugin cwd hit the known `nodes` import collision;
  rerunning from Comfy root exposed the actual two mask failures before their fix.
- Next: persist complete compose receipts, use the verified native-noise mode in
  new delivery workflows, test real partial-tail/audio/overlapping long windows,
  then complete first-frame/region/person/combination/UI and full short/32second
  human/release gates below. No probe remains running; old299nodes/207workflows
  remain unchanged. The full goal stays active, not narrowed to this short clip.

| Requirement | Evidence required before completion |
| --- | --- |
| Fixed upstream reproduction | Same-input original/T8 inference, pinned models/settings, intermediate mask/window checks and decoded media comparison |
| Four independent expansion sides, ratios and placement | Deterministic geometry tests; UI preview matches delivered bounds; no silent source crop or stretch |
| Actual sampling MP budget | Planned and executed tensor dimensions agree and stay within the budget; no quality knob merely relabels a full-size render |
| Original-region preservation | Before lossy encoding, source pixels including subtitles/auxiliary channels equal the input in unscaled mode; transformed modes explicitly identified |
| Original duration and sound | Padding never delivered; exact source frame count/timestamps and audio content/timing checked, with compatible original-packet mux |
| Serial temporal expansion | No overlapping GPU jobs; consistent spatial ownership, H3 phase alignment and overlap conditioning across windows |
| Scene cuts | Source cuts reset generated context; no prior-scene carryover; per-shot prompts/reference ownership |
| Bounded long-video memory | Source read, VAE encode/decode, sampling and encode all audited; repeated windows measured, not only transformer attention |
| Resume/cancel/retry | Source/config-bound state, checksummed window assets, atomic commits, explicit retry and actual interruption/recovery tests |
| Optional boundary color match, on by default | Only expanded region changes; source remains exact; temporal color stability and seam blind review |
| First-frame preview/selection | Explicit reference selection and source-bound propagation; preview is not misrepresented as guaranteed full-clip quality |
| Regional prompts and person protection | Region/time ownership plus masks tested on multiple people; no claim that prompts guarantee no hallucinations |
| Post-upscale and acceleration combinations | Independently validated combinations with existing upscalers and H3 accelerators; unavailable/unsupported combinations named, not silently substituted |
| Native UX and workflows | Separate Plan/Prepare/Serial/Compose components, compact shortcut, standard reusable VIDEO output and progress; not a one-shot generator hidden in SaveVideo |
| No regressions | Existing node IDs/schemas/defaults and all 207 existing workflow bytes preserved; append-only registration and full regression |
| Perceptual acceptance | Short probes and approximately 32-second multi-window clips: talking person, game character/scene, moving camera, subtitles and hard cuts; full human review |

## Execution sequence

### Geometry-preview draft checkpoint — 2026-09-06

- Added separate `video_outpaint_preview.py` and an unregistered V3 preview node.
  They do not change any runtime files captured by the running32-second probe.
  Authoritative output/source bounds come from the verified Plan. A single chosen
  source frame is decoded by an isolated CPU FFmpeg process and downscaled for
  display; blue checkerboard areas are **not generated imagery**. Exact output
  dimensions, actual model pixels, source frame and shot are reported separately.
- Thumbnail canvas is bounded to1536px per edge (default768), plus a small footer.
  This does not restrict the output resolution. Full-resolution decoder working
  memory still depends on source size; only the returned RGB thumbnail is bounded,
  not a claim that decoding arbitrary source sizes costs constant memory. No VAE,
  diffusion, complete IMAGE batch, source edits or audio edits are performed.
  Source identity is checked before and after reads, including failed/cancelled
  reads. Only the owned child process is reaped. Labels are outside the canvas.
- Ruff and compilation passed. Geometry/mapping, selected-frame real-media,
  source mutation, cancellation/owned process and native UI-output tests are
  written **but not run** while the long GPU test continues. The earlier seven
  workflow tests are also pending. No preview workflow is delivered yet; a separate
  Plan→Preview graph is needed to avoid automatically running a connected sampler.
- This does not satisfy generated first-frame candidate selection/propagation,
  regional/person protection, browser acceptance or final workflow/release gates.
  Latest process check: launcher29140/server3412 live,8/22windows committed in the
  resumed32-second root. No duplicate GPU job started, no terminal result yet.
- Follow-up: the workflow builder now supports `--preview-only`, requiring the
  **actual available** LoadVideo/Plan/GeometryPreview schemas before writing into
  a fresh directory. It emits a separate three-node API/frontend graph with no
  model loader or sampler. A new explicitly isolated CPU-preview extension
  registers only Plan and GeometryPreview; the live GPU extension is untouched.
  Three additional workflow tests use JSON-roundtripped native V3 schema (raw
  Python tuples are not the HTTP schema representation). Ruff/compile passed;
  these tests and actual preview execution/import are still pending. Do not run
  the new CPU server concurrently with the ongoing GPU test. Latest live check
  confirms10/22sampling windows committed, not completion.
- Preview fault handling adds an owned-decoder stderr size bound and six negative
  cases (child exit, missing/extra RGB bytes, stderr, timeout, missing executable).
  All29preview tests and10workflow tests remain **unexecuted**, awaiting the serial
  GPU run. Ruff/compile pass only. A50second wait on actual launcher29140 ended
  with both29140/3412 still live and11/22windows committed at2467.203s.

### Generated first-frame selection: next integration boundary

- Source-frame geometry preview above does not replace generated candidate choice.
  Current `sample_prepared_outpaint_windows` consumes every remaining window and
  `OutpaintWindowStore` requires a contiguous prefix; there is no successful
  partial-return mode yet. Do not emulate a user-visible pause by intentionally
  raising a cancellation exception or claim that an ordinary preview sink pauses
  other queued nodes.
- Add an explicit bounded initial-window candidate stage, using the same actual
  source/audio/text/model/noise binding as full generation. Candidate media and
  its selected window assets must be tied together. Choosing a first-frame
  appearance should retain that candidate's temporal context for continuation,
  not silently create a new seed or merely display a disconnected image.
- Explicit selection must name an existing candidate and bind source, plan,
  prompt, model and sampled-prefix hashes. Continuation must actually use its
  preserved prefix/overlap, with tests that reject a different source/plan or
  changed candidate and preserve non-selected candidates. Scene cuts must reset
  reference ownership; a previous scene's chosen appearance must not leak across
  a cut. This is still implementation work, not a completed capability.

### Workflow/UI and long-probe recovery checkpoint — 2026-09-06

- Original32-second launcher/server disappeared at13/20 in the first window with
  zero complete sampling commits. Two native process inventories and the closed
 8191port confirmed termination; the old `running` report was stale. Cause unknown.
  Preserved all old evidence. A hidden detached launcher now resumes into
  `artifacts/outpaint-t8-qipao32-reserve5-resumed-20260906` with the same settings,
  reusing completed source/audio caches and restarting only unsaved sampling work.
  Latest verified3/22windows committed, launcher29140/server3412 live. Logs are in
  `artifacts/outpaint-qipao32-background-recovery-20260906`;7200s timeout retained.
- Added independent frontend workflow builder and actual-schema fixture. It emits
  a full four-stage graph and native compact subgraph. Only the new four stages
  are collapsed, upload/loaders remain exterior, and advanced controls remain inside.
  Static expansion verifies identical real nodes/widgets/named edges. No old files,
  registrations, versions or normal user workflow directory were changed.
- Real browser checks imported both drafts without missing nodes; the placeholder
  video naturally requires user selection. The initial compact layout was too tall,
  so advanced controls moved inside and loader panels were collapsed. Selected
  outer prompt/color edits persisted and appeared correctly in actual frontend
  API export, preserving both VAEs/model/seed/native noise and all four stages.
  No generation was submitted through the UI. Evidence is under
  `artifacts/outpaint-workflow-drafts-v4-20260906/ui_review.json`.
- Seven new workflow unit tests are written but await serial execution after the
  GPU probe; Ruff and generator checks passed. This is not final workflow delivery,
  complete frontend generation acceptance or any perceptual gate. Geometry preview,
  reference selection/propagation, region/person protection, combinations and all
  original long/multi-material/human/release requirements remain open.

### Latest audio/recovery checkpoint — 2026-09-06

- Real416x416x80 audio input with56-frame windows completed two stock20/native-noise
  sampling windows, but final mux validation failed: FFmpeg7.1.1 rounded the final
  AAC packet duration from171 to192 samples. The strict validator correctly refused
  publication. This run's minimum free VRAM310.473MiB also failed the512MiB margin.
- Replaced only the outpaint packet-copy boundary with an isolated PyAV18.1.0 /
  libavformat62.12.102 worker. It merges unchanged packets by rational DTS and holds
  at most one pending packet per stream. No Torch/Comfy imports in the child, no
  relaxed audio checks, no changes to existing DLSS routes. This contains this mux
  boundary, not every native-library crash elsewhere.
- Explicit `--resume-compose` reuses only fully sampled, checksummed matching-plan
  windows when sampling/model/source implementations are unchanged. It copies into
  a new root and retains the failed original. Corrected run
  `artifacts/outpaint-t8-audio-partial80-compose-fixed-20260906` exited0 with80frames,
  106 original AAC packets, exact payload/PCM/both timelines and a verified persistent
  pixel receipt. Final SHA:
  `886944073a08fd87b95c5e7276c5a80d33d946fc636ac185e0425d34f0ef58c8`.
  Two sampled window assets were preserved; no new diffusion steps. Minimum free
  VRAM818.512MiB is compose-only evidence and does not repair the original sampling
  margin. Frame40 shows coherent room expansion; no full human acceptance yet.
- 347 focused tests passed,2179 deselected,5 existing warnings; Ruff/diff passed.
  Includes real32k/48k dual-audio-track exact preservation, cancellation and child
  failure, but not full-repository release validation.
- Separate original32-second fan-dance input was copied byte-for-byte, not looped
  or rescaled. Its SHA is
  `10CE6352F704700A3DBC24CBF19F503D1B6A6B244258FD6B14CCD98DF3D42BA0`.
  One serial probe is running at `artifacts/outpaint-t8-qipao32-reserve5-20260906`:
  736x608x768@24,0.447488MP,56-frame windows,stock20,native noise,color on,reserve5GiB.
  Check its terminal result before another GPU job. All original advanced/UI/workflow,
  multi-material human and release gates remain open.

### Delivery-report checkpoint — 2026-09-06

- New `video_outpaint_delivery.py` persists a complete `.mp4.outpaint.json`
  sidecar: final-video SHA, actual packet/PCM/timing validation, full pre-encode
  pixel receipt, plan and execution identity, color/encoding options and current
  composition implementation hashes. The returned report matches the saved one.
  The verifier reads actual video bytes and rejects missing/mismatched files.
  Hashes detect corruption and mismatch; they are not authenticity signatures.
- The report is fsynced and linked without replacement before publishing video.
  These are **two links, not a filesystem-wide atomic pair**. An abrupt exit can
  leave an orphan report, which does not count as completion. Ordinary failure
  removes only the newly owned report link. Existing files/orphans are preserved,
  and the draft node chooses another numbered output name.
- Compose is now an output node itself; the isolated probe no longer re-encodes
  its already verified VIDEO through a redundant SaveVideo execution sink. Probe
  success now requires the persisted report to bind the source and final file.
  This changed test graph has **not yet had a new real GPU run**. Old short GPU
  artifacts do not acquire retroactive receipts or new runtime hashes.
- Real tiny MP4/AAC with fake models verifies round-trip report equality and
  original audio/pixel evidence; unit tests inject fsync failure, cancellation
  between links, racing user files, corruption and metadata override. An actual
  child exits47 between links: report remains, video does not, verifier refuses
  completion. This is CPU publication-boundary recovery, not CUDA process kill.
- Final focused regression:336 passed,2179 deselected,5 existing warnings. Ruff
  and diff checks passed; published `sampling.py`, `nodes.py`, old299nodes and
  the207workflow snapshot remain untouched. No current task probe is running.
- Next concrete test: parameterize the isolated probe for a source-bound80-frame
  audio/partial-tail case with56-frame overlapping windows and native noise,
  then move to original approximately32-second multi-window material. A suitable
  existing audio source is
  `artifacts/native-masked-context-new-ema-b-classical-single-utterance-pair-review-20260902/lipsync_original_segment0.mp4`,
  SHA`610F5CEE8F99BCCBB53046B79C2FB191C66D9A1BE419B34C193CCB74E23CF6CA`,
  verified416x224x124@24fps with audio. Any80-frame derivative must be a separately
  documented test input, not overwrite the original or claim to preserve its
  entire124-frame duration. No new derivative or GPU audio test was made here.
  All original first-frame/region/person/combination/UI/human/release gates remain.

1. P0: pin/license audit, original unit tests, existing-contract snapshot, real upstream baseline with stock FL2VA/20 steps and the reference KJ memory pair (no extra LoRA/attention acceleration).
2. P1: geometry/ownership/actual-budget planner, source and audio protection, reference-compatible sampler and standard composition.
3. P2: masked boundary color match, scene resets, progressive decode/encode and resumable serial state.
4. P3: first-frame selection, regional prompts/protected regions and independent upscale/acceleration workflows.
5. P4: real short and 32-second serial matrices, strict media/VRAM checks and full human review; then documentation, workflow synchronization and release checks.

## Current evidence

Latest: the resumed T8 real GPU clip exists and passes media/1118MiB minimum free
memory, but **fails visual screening**: unrelated orange content appears in its
expanded regions. No full-clip human acceptance or release is claimed. The two
upstream probes remain separate reference evidence. Older checkpoints below are
chronological, not replacements for this current failed-quality status.

### 2026-09-06 current provider, draft-node and real-reference checkpoint

- The native MODEL provider fingerprints loaded state in <=4MiB blocks, configuration,
  implementation and numerical runtime settings. Native CLIP output is frozen to
  shot-local tensor/metadata snapshots; prompt changes or different embeddings do not
  silently reuse a cache. Actual source/audio/text/MODEL providers now reach the serial
  sampler coordinator. Model hashing currently costs about 39 seconds on the real CPU
  checkpoint and repeats around sampling; no speed claim is made.
- Four V3 draft stages exist in `nodes_video_outpaint.py`: Plan, Prepare, Serial Sample
  and Compose. They are **not registered** in `nodes.py`, and no delivered workflow
  advertises them. Fake-model/real-MP4 tests cover the complete stage chain, early
  shot/track validation, optional default-on color match and no-overwrite reusable VIDEO.
- The reference API graph uses external KJ `MiniMaxLowVRAMAttention` (4 head groups)
  followed by `MiniMaxChunkFeedForward` (4 chunks, threshold 4096). It is not a bare
  unpatched MODEL baseline. The new execution adapter accepts those complete pinned
  patch sets, verifies bound owners/live code/FFN closures/settings, and handles Comfy
  application/restoration state. KJ source SHA is
  `C371576B1BB31A2F518BDB4CEDA43CB10B20338F0C9D68F99ED1BE76CE06478F`.
  Unknown LoRA/attention owners remain pending, not silently enabled. Tiny CPU native
  block math agrees within 1e-6; this is not learned INT8/CUDA parity.
- The earlier real CPU bare-model fingerprint report at
  `artifacts/outpaint-model-identity-real-cpu-20260906/report.json` covered the previous
  v1 provider (933 tensors; no forward). After the KJ/runtime identity extension it is
  historical evidence, not a current-code pass. The new KJ real-CPU probe at
  `artifacts/outpaint-kj-model-identity-real-cpu-20260906` exited 0: 933 tensors and
  150 actual patch bindings verified twice, 19.453 seconds for one hash. Current
  identity SHA is `1d06e6fe8769450b3c3ff054b1a0bd0a01c706cbb8da60091b3c96a9b2fc914d`.
  This probe loaded CPU weights but did not execute a learned forward or CUDA.
- First real upstream GPU probe: `artifacts/outpaint-upstream-p0-20260906`,
  640x832x90@24fps, 345.35s prompt time; strict silent-video/joint decode passed, but
  minimum free VRAM was 289MiB. Frame 45 has repeated rabbits and black source-boundary
  bars. Artificial source padding was a test confound; the result is not accepted.
- Second real upstream GPU probe: `artifacts/outpaint-upstream-clean-p0-20260906`,
  clean 512x288 input without bars, output 512x672x90@24fps, 205.36s prompt time.
  Candidate SHA `09F5F4B257A6DFA03EE5B583C4B2DCE734F8D75C0B1103233FE7FB896B66487E`.
  Strict decode passed. Frame 45 has no obvious duplicated subject/black bands; only
  that representative frame was inspected, not the full clip. Both geometry and padding
  changed, so this does not isolate padding as the sole cause. Minimum free VRAM was
  430.664MiB: still **memory margin failed**, human acceptance false. Both isolated
  servers terminated. No T8 GPU output exists yet.
- Focused regression after KJ/runtime binding: **302 passed, 2179 deselected, 5 existing
  warnings**. Earlier 27 targeted tests are a separate run, not additional full coverage.
  Old 299-node registration and the 207-workflow snapshot remain unchanged. The original
  complete advanced/UI/real short-and-32-second/human/release requirements remain open.
- The first T8 draft-node GPU probe has now been launched in its own port 8191 server:
  `artifacts/outpaint-t8-clean-p0-20260906`. It reuses the exact clean reference source,
  512x672x90/stock20/KJ4+4, color matching off. T8 coordinate noise and 3GiB VRAM reserve
  differ from the upstream probe, so this is not a numerical/speed-parity claim.
  All four actual model files are hashed before submission. Inspect the live session
  and final report; a launch is not a pass. Draft nodes are exposed only through the
  isolated tools extension; published registration/workflows remain untouched.
- Recovery observation: after an application/session refresh, the original T8 exec
  handle was missing, two native inventories found no Python process, and port 8191
  was closed. Its log stopped at 16/20 and its store had zero committed windows;
  `running` in the raw report was stale. The cause is unknown, not assumed to be an
  outpaint or driver defect. Original evidence remains unchanged with a separate
  `process_recovery_observation.json`; no final video or VRAM margin is claimed.
  The new `--resume-from` probe checks prior PID absence, source/runtime/model hashes,
  copies caches to a new run and explicitly resumes only at a committed-window
  boundary. This cannot recover the unsaved first 16 sampling steps. It now flushes
  each telemetry observation to a live JSONL journal. Two focused harness tests and
  Ruff passed. The serial retry is `artifacts/outpaint-t8-clean-p0-resumed-20260906`;
  inspect its real process/report before another run.
- The resumed T8 process exited 0 with `media_pass_human_review_pending`. Output is
  512x672x90@24fps, SHA `9D1FFC9135CEAE416510D697FAF8FDEA3FE871082A1C466A2D190E40717C0CF1`;
  native video/joint decode passed and source cache was reused with zero re-encoding.
  Minimum free VRAM was 1118.082MiB at approximately 10Hz; elapsed server-run time
  265.812s includes loading/hash/prepare/sample/decode, not a comparable speed metric.
  The GPU server terminated. **Frame 45 fails visual screening**: unrelated orange
  imagery in the expanded areas, while the source center is retained. Raw report is
  preserved; separate `agent_review.json` rejects visual acceptance.
  A VAE-only diagnostic now compares native full-source encoding with saved bounded
  source latents, and native/full versus bounded decoding of the *same* saved sampled
  latent. No diffusion is rerun for this isolation step. Diagnostic directory:
  `artifacts/outpaint-vae-isolation-20260906`; inspect the live session/result.
- VAE isolation is now terminal, exit 0. With the unchanged FP16 VAE file SHA
  `7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`, the actual
  native whole-90-frame source encoding and saved bounded source latents agree
  **bit-for-bit (max/mean error 0)**. Native full decoding and bounded decoding of
  the same sampled 27-token trajectory also agree bit-for-bit across all 90 frames.
  Both frame-45 images contain the same unrelated expanded imagery. This excludes
  a bounded/native VAE mismatch as the cause in this exact case, not all possible
  materials or modes. The diagnostic intentionally used a short whole-RGB oracle;
  it is not the production memory route. Peak GPU used 9160.977MiB.
  Next: sampling-input/initial-noise parity at the unchanged seed. The current
  coordinate-noise route intentionally differs from upstream whole-tensor RNG;
  do not claim it is the root cause until a controlled comparison proves that.
  No probe remains running at this checkpoint. No new blind review is presented
  for a visibly failed candidate, and no release or full-goal completion is claimed.

- Upstream checkout and MIT license verified. Original 11 tests pass with CPU selected; they use fake MODEL/VAEs/sampling and are **not real-model reproduction**.
- Initial GPU observation: RTX 4060 Ti 16GB, 1038 MiB free. No GPU job was started and no other process was stopped.
- Geometry and compositing foundations are being implemented separately; no new node is registered yet.
- All real-model, long-video, UI, perceptual and advanced-combination gates remain open until their actual evidence exists.

### 2026-09-06 component checkpoint — not an end-to-end release

- The user authorized unloading the loaded Ollama model before a real test, without deleting weights or stopping the service. The unload request encountered a refused connection; Ollama was already absent on recheck. This is not recorded as a successful assistant unload. Subsequent GPU free memory was approximately 6.4GB, below the isolated upstream probe's 10GB initial gate. No other application was closed and no GPU generation started.
- `video_outpaint_sampling.py` now wraps one native joint-AV sampling call, protects source/previous-window latents, discards generated audio, and validates source video/audio overlap identity. Its tests inject CPU fake backends; encoded source preparation and the serial coordinator are not implemented yet. Future per-window VAE encoding must not bypass phase/overlap identity checks.
- `video_outpaint_color.py` implements optional default-on paired inner-strip RGB offset matching, clamped strength, outward fade, shot-local EMA and source/parameter/sequence-bound JSON continuation state. Source and auxiliary channels stay unchanged; whole-chunk and split-chunk outputs agree. This is new finishing logic, not a claim of exact upstream algorithm parity or perceptual acceptance.
- `video_outpaint_media.py` reuses read-only source/media validators without loading a DLSS binary. It verifies actual file SHA, untrimmed 24fps SDR CFR input, geometry/frame count and original start time. Finalization copies source audio with an isolated FFmpeg process, checks packet payload/timestamps and decoded PCM/timestamps, and atomically publishes a new standard reusable VIDEO without overwriting an existing file. Hard-link publication requires a supporting target filesystem. The validator retains O(frame count) timestamps but no complete IMAGE batch.
- The initial in-process PyAV remux/validation integration encountered a Windows access violation. The new route replaced remux with external FFmpeg; a subsequent combined run passed 225 tests (5 existing warnings), including actual tiny silent and AAC MP4s, cancellation-before-publication, invalid candidate rejection and existing registration/DLSS regression. This does not establish the cause of all PyAV failures or close long-run stability testing. An additional source-overlap sampler guard was added afterward and requires its own rerun.
- File finalization does **not yet** verify a hash-bound pre-encode pixel-preservation receipt from the generator. Source encoding, bounded VAE, durable resume/cancel, workflow integration, real short/32-second tests and all advanced gates above remain open. Old nodes and workflows remain unchanged; nothing has been committed or published for this feature.
- Follow-up verification after the source-overlap guard: all 13 sampler/file tests passed, including another serial real-file run; all newly added Python files passed Ruff. The earlier 225-test result and this targeted rerun are distinct runs, not a claimed full-repository pass.

### 2026-09-06 source preparation/cache progress

- `video_outpaint_prepare.py` follows the native H3 VAE's independent 17-frame clips, five tokens per clip and final three-token drop. Each bounded public encode call supplies 22 frames and retains its first five tokens (two for the final shot chunk). This retains the original VAE/model-management methods without mutating token-drop settings, at the cost of one sacrificial padded internal clip per call. The learned VAE still needs a real-model comparison.
- CPU tests reuse the actual native `encode_temporal`/`encode` methods with a cheap deterministic fake spatial encoder. Exact equality holds against the whole-shot oracle for 1, 5, 16, 17, 18, 22, 34, 39, 73, 90 and 768 input frames. The 768-frame case proves temporal indexing only, **not** a 32-second learned generation or memory benchmark.
- Grid-aligned native-scale sources follow the reference's source-only encode/latent embedding. Other dimensions use explicit isotropic grid mapping with edge extension, then keep only the source-locked cells; that geometry is intentionally not labeled exact reference parity. Final original pixels still come from the source compositor, not these VAE reconstructions. Pixel-coordinate tests cover identity, placement, inverse mapping and actual MP dimensions. A low-MP planner corner case with minimum-height canvases was fixed when this new test exposed it.
- A real CPU MP4 test covers the sequential 17-frame reader, forward resume skips and final-frame hold. Native model weights were not loaded. Source-cache preparation supports starting at a committed five-token boundary and never rereads another shot's scene for padding.
- `video_outpaint_source_store.py` uses the existing OS-owned manifest lock and atomic-file utility without changing their code. Chunks are immutable SHA-named safetensors, source/plan/VAE-identity bound, committed in strict shot/token order. Window reads hash-check only bounded relevant chunks and have a 57-token cap. Tests cover restart position, exactly shared overlapping source latents, gap/duplicate/wrong-identity rejection, corruption and failure after asset write but before manifest advance. These are source-preparation transactions, **not** a completed durable sampling coordinator or a real process-kill recovery gate.
- Combined new preparation/store tests: 22 passed with 5 existing warnings. The runtime coordinator must compute and verify the loaded VAE identity rather than trusting a caller-supplied label, revalidate source file bytes before/after preparation, provide source audio conditioning/noise, and connect sampled latent storage, decode, pixel receipts and publication. All full-scope gates remain as listed above.
- Final focused rerun for this checkpoint: `pytest tests -k 'video_outpaint or preflight_and_registration'` passed 214 tests (2179 deselected, 5 existing warnings), and Ruff passed the new preparation/store modules and their tests. This is a focused run, not a full-repository pass. Latest GPU observation: 5543MiB free, below the upstream probe's 10000MiB initial threshold; no GPU job started.

### 2026-09-06 verified source coordinator and serial sampling persistence

- `video_outpaint_source_runtime.py` now computes loaded VAE identity from actual tensor bytes, nonpersistent buffers, dtype/tiling settings and implementation files with <=4MiB copy blocks. This SHA is a **loaded-state fingerprint**, not a safetensors filename/file SHA. Pending weight/object patches and states requiring an unbounded conversion are rejected. Production learned-VAE/offload/quantized-state coverage is still pending; the current tested model is a small CPU oracle.
- The source coordinator checks file identity, verifies reused latent assets, resumes encoding at the first uncommitted chunk, and checks file/VAE identity even after cooperative cancellation. A mid-run change writes an invalidation marker and prevents later reuse. The source receipt explicitly says that audio and generated video are not complete. Real tiny MP4 + fake-VAE tests cover interruption/resume, zero re-encode on a completed source cache, changed weights/buffers/configuration and invalidation.
- A same-user OS lock under the system temporary directory serializes this new route across Comfy processes/roots. It does not control other apps or old Comfy workflows; real GPU probes still need queue/headroom checks and must not interfere with them.
- `video_outpaint_noise.py` creates bounded, stateless shot/token-addressed noise, preserving exact overlaps across retries and window sizes without altering global RNG. It is explicitly **T8 coordinate noise v1**, not the upstream whole-tensor RNG order; exact upstream numerical comparisons must arrange matching noise separately.
- `video_outpaint_window_store.py` stores immutable SHA-checked AV window assets in serial order, binds model/conditioning/source/audio/settings, rejects gaps/duplicates/corruption and derives context from the previous committed window. `sampled` means all latent windows exist, never that a final video has been decoded/accepted.
- `video_outpaint_sampling_runtime.py` connects complete source-window reads, phase-consistent audio providers, coordinate noise, the single-window sampler and durable commits. It requires a live execution-identity verifier and an explicit resume request after interruption. Internal provider construction/real model+conditioning+audio fingerprint verification at the workflow layer remains to implement; a caller-supplied report is not sufficient proof. No silent-audio substitution is made for real source sound.
- CPU fake-sampler tests verify uninterrupted/resumed equality, exact source/audio/context preservation, shot resets, backend failure/retry and corrupted-output rejection. An actual child process exits with `os._exit(43)` after the first commit; the parent observes the terminal exit, acquires the released OS lock and resumes only the second window without changing the first. This is a real **CPU process-abrupt-exit** test, not a CUDA process-kill/OOM recovery gate. Sampling-runtime tests: 6 passed; source-runtime tests: 3 passed.
- Next implementation: bounded source-audio encoding (native posterior attention is **causal**, not independent per window), verified real model/conditioning providers, bounded VAE decode, original-pixel receipt to final-file binding, node/workflow/UI delivery and all original advanced and real/perceptual gates. No learned H3 model was loaded this checkpoint and no feature release occurred.
- Final focused rerun: 223 passed, 2179 deselected, 5 existing warnings; new runtime/test Ruff, JSON parse and diff checks passed. Latest free VRAM was 4529MiB. No real-model probe was started, and the 299 existing nodes/207 workflow baseline remains unchanged.

### 2026-09-06 bounded global decode and encoded-file evidence chain

- `video_outpaint_decode.py` reads at most seven tokens from the committed global sampled trajectory, uses the original VAE's `prepare_decode`/device context and raw spatial decoder, and applies native five-frame temporal blending **before** RGB normalization/clamping. It does not decode independent sample windows and blend their already-clamped RGB. Only a 28-frame raw result and five-frame overlap are retained; delivered chunks are <=17 frames and omit source padding.
- The CPU oracle uses native temporal decode/blend/finalization methods with a deterministic fake spatial decoder deliberately producing out-of-gamut values. Tests cover 1 through 768 original frames. Initial strict equality failures were diagnosed as <=1.1920928955078125e-7 differences from the fake decoder's mean reduction on strided versus compact tensors; making that fake reduction explicitly contiguous isolates temporal correctness. All 12 decode tests then passed. This does not claim bitwise equality for a learned CUDA decoder; real numerical and perceptual parity remain required.
- `OutpaintWindowStore.read_video_range` provides bounded global reads with earliest-committed-window ownership, so decoding crosses sampling seams without independently restarting the decoder timeline. Stored AV assets are still hash-checked.
- `video_outpaint_compose.py` now joins verified loaded-VAE/source/checkpoint identities, global decode, explicit model-to-output mapping, default-on optional boundary color matching, exact raw-source RGB8 paste-back, a bounded rawvideo pipe into isolated FFmpeg, and the original-audio finalizer. It verifies source pixels before encoder input and binds both input/source-region RGB digests to plan/source/window-manifest/candidate-file SHA in a pixel receipt. The finalizer checks the matching receipt before atomic no-overwrite publication. It never claims pixel equality after lossy H.264 encoding.
- RGB-to-YUV conversion explicitly uses full-range RGB to limited-range BT.709, not just BT.709 tags. Even geometry uses yuv420p. Odd exact dimensions use yuv444p without changing requested geometry, with an explicit player-compatibility review flag; odd-size browser playback is **not** accepted yet.
- A cancellation test exposed the Windows PATH FFmpeg launcher/child lifetime: killing only the launcher left the encoded temporary briefly open. Cleanup now sends EOF, waits and, if needed, terminates only that owned process's descendants; it never searches for or kills unrelated FFmpeg jobs. Pipe writes are unbuffered, handle partial writes and check cancellation between writes. This fixed actual cancel/retry tests for even and odd outputs.
- Composition tests use an actual 39-frame MP4/AAC source and fake VAE/diffusion backends, then real FFmpeg encode/remux/decode. Output geometry/frame count, raw-source RGB receipt, original audio packet payload/timing and decoded PCM/timing pass; cancel publishes nothing, leaves sampling checkpoints intact and supports compose retry. Initial composition tests had one cleanup failure; after the lifecycle fix the combined compose/media/decode run passed 25 tests (5 existing warnings). Two later ownership/receipt-negative tests were added and await the focused rerun.
- This is a CPU fake-model/real-media component chain, not H3 generation acceptance or a shipped node. Still required: learned source-audio/model/provider integration; node/workflow/UI wiring; all first-frame/region/person/acceleration capabilities; real short/32-second and CUDA stability matrices; human review and release gates.
- Focused rerun after the ownership/receipt tests: 242 passed, 2179 deselected, 5 existing warnings; Ruff/JSON/diff checks passed. Latest free VRAM was 3556MiB, so no learned-model run started. Loaded-VAE identity was then extended to include local preparation/decode implementation hashes, so a later code correction cannot silently reuse an older cache merely because VAE weights stayed the same; targeted source/compose rerun is required after this small identity change.
- That follow-up identity-change rerun passed all 8 source-runtime/compose tests, with Ruff and diff checks passing. This is separate from the preceding 242-test focused run, not an additional full-repository validation.

### 2026-09-06 source-audio semantics and bounded native posterior

- Corrected an earlier component assumption: only **observed** source-audio latents are locked. Missing source audio and unobserved padding may generate **internal model context**, which is retained/frozen across overlapping sampling windows. It is never delivered as a replacement soundtrack; final composition still copies all original audio tracks. Sampling masks are cloned before calling the backend so it cannot mutate ownership. CPU tests cover mixed observed/generated cells, context recovery after interruption and backend mask mutation. Two misplaced/incorrect test assertions from the preceding edit were corrected before the passing rerun.
- Sampling checkpoint identity now includes local sampling/runtime/noise implementation hashes, preventing reuse of the older all-audio-locked semantics. No old published node or workflow was changed.
- `video_outpaint_audio_file.py` writes timestamp-positioned, resampled 32kHz stereo decoder frames to a temporary disk file. Source gaps remain observed silence when an audio track exists; a genuinely audio-free source returns no observed audio. Reader sizes are bounded, shot sample boundaries use rational arithmetic, source identity is revalidated and the temporary PCM SHA is checked after use. Default conditioning selects the **first** audio track, whereas the pinned upstream selects the last; multi-track reference probes must explicitly align this choice. Final output still retains all original tracks.
- `video_outpaint_audio.py` validates native CNN geometry, derives the 800-sample hop and 13-token waveform halo, and encodes bounded chunks. The posterior head retains its full causal prefix using SHA-checked temporary disk KV blocks and FP32 online softmax. Tensor memory is bounded; KV disk/metadata grows linearly and attention compute remains quadratic. This is not independent per-window VAE encoding and is not a speedup claim.
- The managed adapter uses Comfy's device context, bounded memory estimate and model loading/offload entry point. It does not mutate VAE options or silently fall back to independent tiled audio after OOM. A loaded audio-state fingerprint combines bounded actual tensor/buffer verification with audio implementation identity; GPU wrapper and durable audio-cache/coordinator integration remain unverified/unimplemented respectively.
- Tiny native-operation tests and actual CPU MP4/AAC tests cover causal attention, waveform halos, delayed 48kHz audio, exact timestamp placement, shot partition, source-to-posterior parity, cancellation cleanup, tampering and model lifecycle. Focused regression: **263 passed, 2179 deselected, 5 existing warnings**; Ruff passed. This is not a full-repository test run.
- Real learned encoder CPU probe: `tools/run_video_outpaint_audio_cpu_probe.py` loads only the encoder/posterior tensors from the unchanged local `minimax_h3_audio_vae_fp32.safetensors`, comparing the actual native encode method against bounded encoding. Initial source `A.mp4` was rejected because its declared real/average rates differ; no silent retiming was applied. The validated `lipsync_original_segment0.mp4` was used instead. Three sample counts (17613, 55181, 103217), with blocks 16/32/64, passed the predeclared `atol=rtol=1e-4`; maximum absolute error was 5.304813385009766e-6. Report: `artifacts/outpaint-audio-cpu-parity-20260906/report.json`. A final source-hash-aligned rerun follows after adding the managed wrapper; inspect its terminal report before citing it as passed.
- Latest free GPU memory was 2483MiB. No H3 diffusion, generated outpaint video, real GPU VAE wrapper, node/UI/workflow or human acceptance is claimed. Next: durable audio store and verified real audio/model/conditioning providers, complete node/workflow integration, every original advanced capability, serial short/32-second generated comparisons and full release gates. The original full goal remains active.
- Final probe process exited 0 and `artifacts/outpaint-audio-cpu-parity-final-20260906/report.json` is `passed_cpu_learned_encoder_parity_only`. Its encoder/file-adapter source hashes match the current files; the three error measurements match the first probe. Actual unchanged checkpoint SHA: `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`; 345473024 bytes of encoder/posterior tensors were loaded. This closes only the limited real learned **CPU audio encoder** comparison. JSON/diff checks also passed; no processes from these probes remain running.

### 2026-09-06 durable audio provider and native-decoder isolation

- `video_outpaint_audio_store.py` commits immutable, SHA-checked observed-audio posterior chunks in shot/token order. Identity binds plan, actual loaded audio VAE, canonical PCM, selected track, block size/device and implementation files. Sampling reads <=320 audio tokens, keeps exact cached overlap and supplies unobserved masks only for missing/padded source audio. Silent files need no audio VAE. These are conditioning assets, never replacement audio tracks.
- `video_outpaint_audio_runtime.py` holds the route's serial OS lease and an audio-worker lock, checks source/model identity and verifies existing assets. Resume must be explicit. Completed shots are skipped; an interrupted shot replays its causal prefix and requires exact equality with previously committed posterior chunks before appending new ones. This is intentionally **not zero-recomputation resume**. Source/VAE/temporary-PCM changes or a changed replay invalidate the cache; sampled windows cannot consume an invalidated store.
- `OutpaintAudioProvider` computes its binding from the actual completed manifest, verifies the current source file and rechecks the selected latent assets on reads. The sampler chain test uses a real 80-frame MP4/AAC, tiny native VAE and fake diffusion backend; original observed audio stays exact across two windows while padded internal audio can be generated. Real model/conditioning execution providers and public node wiring are still missing.
- The actual Comfy `sd.VAE` CPU probe loads all 917 checkpoint tensors/buffers (605149280 bytes), intentionally cancels after the first audio commit, replays that prefix once, appends seven remaining chunks and preserves the first asset. Two shots match whole native learned encode with maximum errors 4.887580871582031e-6 and 3.337860107421875e-6, within the predeclared 1e-4 absolute/relative tolerance. A completed cache reuses all chunks with no VAE encoding. Initial report: `artifacts/outpaint-audio-cache-real-cpu-20260906/report.json`; no CUDA/DiT/human acceptance is implied.
- The subsequent combined test run **failed with a Windows native access violation** at PyAV open during repeated audio preparation. Python exceptions cannot contain that fault. New `video_outpaint_audio_decode.py` runs the same timestamp-based resampling in an isolated CPU Python child with no Torch/Comfy imports; the parent checks exit code and exact PCM size, owns cancellation cleanup and hashes worker code into cache identity. This contains this decoding boundary; it does not prove the root cause of every PyAV crash or isolate all existing media-inspection code.
- After isolation: focused runs passed 272 and then 274 tests (2179 deselected, 5 existing warnings). An additional negative rational-origin command fix/test was followed by 17 audio-file/runtime tests passing. Tests include an actual child exit 55 (parent survives), cancellation of an owned live child, corruption/invalidation, and failure after asset write but before manifest commit; retry reuses identical asset bytes without overwriting. Ruff/diff checks pass. These are focused runs, not full-repository regression.
- A source-hash-aligned real Comfy CPU rerun with isolated decoding is in `artifacts/outpaint-audio-cache-isolated-real-cpu-20260906`; inspect terminal status before citing it as passed. Earlier report hashes are historical after the worker extraction. Latest GPU free memory: 1843MiB; no H3 diffusion probe started. Remaining full scope is unchanged: real model/conditioning providers, node/workflow/UI and every original advanced feature, short/32-second serial generated comparisons, human review and full release gates.
- The isolated real CPU rerun exited 0; its terminal report is `passed_real_comfy_audio_cache_cpu_only` and all five implementation hashes match current source. It preserved the first commit, replayed one prefix chunk, appended seven new chunks and reused the completed cache with zero encoding. Two-shot maximum absolute errors were 7.003545761108398e-7 and 6.183981895446777e-7; both pass the same predeclared tolerance. This verifies this CPU audio path after isolation, not CUDA/DiT generation or universal native-library stability. No probe/test session remains running at this checkpoint.
