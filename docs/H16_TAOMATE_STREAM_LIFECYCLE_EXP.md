# TaoMate multi-request lifecycle candidate

H16-5 scoped completed EXP: two-request public sampling and repaired public decode
passed mechanical checks. Cancellation AND injected-fault cleanup followed by
ordinary H3 eight-step generation in the same Core completed; owned children0.
Formal delivery and the bound10second picture/audio/lipsync/event/seam human
review completed on2026-09-17. See [current scope](H16_SOURCE_UPDATE_20260917.md).
Progress notes below preserve historical failed/pending attempts; they do not
override this latest status or qualify arbitrary requests.
Pinned upstream: TaoMate-H3 `ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad`.

## Latest actual public-route evidence (2026-09-17)

- Latest overriding result: `public-tao-stream-repaired-v3` finished decode,
  and `independent-AV-audit.json` passed. The unchanged completed generation was
  explicitly reused; sampling did not rerun. Full240-frame/10second H264/AAC
  SHA256 is `e65749a5de8e68a4671ae81c22d87ccab43dd613d9d473895d2aac487afd46b6`.
  Native243-frame video and324000 stereo/F32 audio samples were checked;
  PCM content SHA256 is `6b667834036b3468ab221fefb932e994d8a2be088a92437a1d33b8c55ae8f0a1`.
  WAV-container PEAK timestamps are not PCM changes. Original failed runs and
  receipts remain failed and untouched. Cleanup left no owned child processes.
- These historical runtime/media receipts remain in the isolated
  `bridge-h16-meridian-20260917` checkout. The current candidate is rebased on
  released v1.84.0 in `meridian-h16-resume-20260917`;624 selected CPU tests and
  Core343/prefix342 checks passed. They do not replace the real GPU recovery
  tests or human listening. `recovery-serial-controller-v5` is running the owned
  cancellation→ordinary H3 and injected fault→ordinary H3 cases serially.
  Never relabel a controller launch or incomplete recovery receipt as success.
- `public-tao-stream-v1` generation exited0 with32 real forwards and full
  postflight verification. `independent-generation-audit.json` separately
  verified both per-request saved hashes, matching execution reports, and
  bit-exact joined video/audio tensors against the earlier artifact experiment.
  The joined container hash differs because metadata differs, not tensor values.
- The first public decoder hit its900-second wall deadline during initial
  hashing, before VAE work. Its request bound74,587,693,930 bytes, including the
  full generation base; hashing twice shares the same wall budget as inference.
  Process receipt confirms the owned worker/children were cleaned. The original
  chain still contains only its verified generation stage. This is a failure,
  not a successful public media run or a quality failure of the saved latents.
- A separate decode-only recovery experiment ran in
  `public-tao-stream-decode-recovery-v1`. It uses the **unchanged** request and
  packaged decoder, with budget `min(7200,900+ceil(2*bytes/32MiB))` =5346seconds.
  It passed initial hashes and both VAE decodes, but the FFmpeg raw-pipe
  publication failed with BrokenPipeError. The worker exited1, with0 owned
  children remaining and no cleanup errors. Its incomplete MP4 is not delivery.
  It did not rerun sampling, edit failed receipts, or promote the original chain.
- CPU replay from the earlier verified native243-frame RGB/audio files reproduced
  the encoder crash: `stream-encoder-auto-v1` failed after233 input frames with
  exit3221225615. Three single-thread H264 replays (`stream-encoder-one-v1` through
  `v3`) each consumed all243 native frames, exited0, and strictly decoded. All
  three output SHA256 values were
  `e65749a5de8e68a4671ae81c22d87ccab43dd613d9d473895d2aac487afd46b6`.
  The only encoder parameter change was `-threads:v 1`; timing, audio filters,
  CRF and sampling were unchanged. This supports a bounded encoder fix, not
  public-route success or a claim that the native pixels changed. The old
  multithreaded encoding is not expected to be byte-identical to a new encoding.
- Full CPU v5 subsequently completed6225passed/196skipped,1726 source files
  unchanged and CUDAfalse. After its termination, production `run_worker`
  gained bounded two-scan hash-time accounting for the three decoder routes
  (maximum7200seconds; identity verification remains intact). The stream encoder
  now uses `-threads:v 1`, preserves decode-stage diagnostics on failure and
  reports the encoder exit code when its pipe breaks.
- `stream-delivery-repair-cpu-v2` passed168 selected tests with1727 sources
  unchanged/CUDAfalse. The preceding v1 collection failed because of a reserved
  pytest parameter name; no tests ran in that failed invocation.
- `public-tao-stream-repaired-v2` is a fresh public-node decode attempt with an
  explicit generation-only checkpoint. Its controller verifies all old request
  inputs and generation math identities, permitting only the named controller
  and decoder source changes. It preserves the failed original state/media;
  no32-forward resampling, failed decode adoption or silent fingerprint rewrite.
  This task is running, not complete. Independent full AV comparison and real
  cancellation→ordinary-H3 recovery remain required. Do not edit its bound
  engine sources while running. No public release or human acceptance yet.

## Observed boundary

Upstream `release_retained_state()` clears KV, transport tails and normalization
anchor, but retains request index and video/audio/global frame offsets. This is
final cleanup, not reset-to-new-stream. Reusing that object after release is invalid.
An exception can occur after some phases or even counters committed; those partial
states cannot safely resume as the next request.

## Local candidate

The existing isolated single-GPU runtime now owns an explicit request state:
ready → running → ready on success; failed/closed owners cannot generate again.
One model object owns the retained stream. A different model cannot borrow its KV.
Successful requests must commit exactly once and advance each global offset by
the actual execution's published count, not a hardcoded five-second assumption.
Prompt/teacher/geometry semantics remain upstream; lifecycle checks alone do not
prove prompt switching or the second teacher artifact correct.

Exceptions and cooperative cancellation permanently invalidate the owner, clear
its upstream cache/tails and CUDA timing events, remove only its block hooks, and
let the existing weight lease restore CPU state. Cleanup errors are retained
without masking the original exception, including Python3.10 without add_note.
Closing is idempotent; closing during an active request is rejected. No model
unload/global CUDA-cache purge, new distributed group or sampling change is added.

## Evidence and remaining work

- Latest artifact result: `tao-two-request-video-v1` completed32 actual forwards
  with distinct prompts, prefix/KV checks and full postflight hashes; exit0,
  owned children empty. Report SHA15800c913a1532afdb9731157c13af0c2b75968bcf2810824db74d8bd119fa73.
  `tao-two-request-decode-v1` then jointly decoded243 native frames and324000
  audio samples, delivering240frames/320000samples/10seconds H264AAC. Full AV
  decode passed, PCM RMS0.17981/0.18175 and peak0.97521, no gain or duplicated
  segment audio. Movie SHA3d7c74f8b607abeb0e25f520952036d615e2b2c0ec0d0ce6a3b2a2f9876f6071.
  Decoder exit0/no children; first/mid/final stills are visible, not black.
  No human speech/seam/quality acceptance is inferred from these checks.
- Public bundle preparation finished31 assets, bundle SHA
  4ea94c3734d8b1e3328de6a3ef450dc1e5d96b34db9427affcfaaf5a5aba6b73.
  Actual public-node/native-input CPU inspection confirmed distinct text,
  audio lengths207/198 and video seeds8301/8302, without CUDA initialization.
  The public `--run` was explicitly started only after artifact decoder cleanup;
  its terminal is now failed at decode. Read stage receipts rather than treating
  the successful generation or earlier launch record as complete media.
- Public candidate now includes `tao_stream` in the existing bundle builder,
  strict contract, shared process controller and unchanged two public nodes.
  New packaged assembly/execution/generation/decode helpers preserve the old
  single-request worker. Per-request file/teacher identities and video seeds,
  globally rounded timeline/prefix handling, joint VAE and global H264/AAC
  delivery are explicit. Huge diagnostic RGB tensor files are not default output.
- `public-stream-contract-cpu-v3`:155 selected tests passed, including original
  Prepared contracts/nodes/process/cache migration, nested identities, seed
  binding, changed-request invalidation and decode-only retry;1715 sources
  unchanged/CUDAfalse. The v1 invocation selected a nonexistent test filename
  and ran no tests; it remains a failed command, not a runtime failure/pass.
- `public-stream-entry-cpu-v1`:34 selected tests passed,1716 sources unchanged,
  CUDAfalse. Covers actual packaged worker entry with CPU weight/runtime doubles,
  owned model→meta disposal and cyclic reference release before final hashes,
  primary-error preservation,2/4-request synthetic execution with real pinned
  geometry, and real FFmpeg delivery with VAE/CUDA substitutes. Not actual GPU
  model performance, voice quality, or public-route GPU qualification.
- `run_public_tao_stream.py` explicitly separates `--prepare`, `--inspect` and
  `--run`. It binds the real switched teacher; only `--run` starts a public GPU
  task. Do not auto-queue behind an unverified live GPU owner. Current preparation
  progress is not completion; read its preparation/terminal files and handle.
- `tao-switch-teacher-v2` is now terminal: exit0, no owned children, report SHA
  `7953762da751dfc26d41a4434e4428c0cb28ebcb6c4cd1d9d7c2dc9059ece3d6`.
  Both prompts are distinct and the nine new teacher forwards and complete
  postflight hashes passed. This qualifies the prepared audio guidance only;
  two-request video generation and decode still require their own evidence.
- Latest: `tao-second-teacher-v1/terminal.json` completed with exit0 and no owned
  children. Its nine actual new Base10 audio forwards used the real previous40
  clean latents; all source/base hashes passed postflight. This same-prompt
  continuation is NOT proof of prompt switching or a completed two-video stream.
- The changed-prompt teacher is now `tao-switch-teacher-v2`, using the separately
  verified earlier native text features for the new Chinese sentence. Its v1
  controller failed before any worker/GPU launch due Windows GBK decoding of a
  UTF8 report. That failure is retained; v2 explicitly reads UTF8. The first
  teacher is reused byte-for-byte, only the changed second guide is computed.
  The completed v2 terminal above permits the next serial video worker.
- `tao-two-request-worker-cpu-v3`:9 selected entry/assembly/error tests passed;
  the video controller now requires distinct prompts and the switched teacher.
- The actual switched-prompt two-request video run is now started under
  `artifacts/h16/tao-two-request-video-v1`. Read its terminal/owned-process state;
  startup or a completed teacher is not evidence of completed video sampling.
- `stream-assembly-cpu-v1`:12 tests using real upstream geometry and synthetic
  tensors cover1/2/3/4 requests, including the fourth request's199 audio latents
  (first207, then198/198/199), exact transport-tail removal and failed append
  without partial commit. This artifact helper is not yet a public worker.
- `stream-manifest-cpu-v1`:63 selected manifest tests, including old contracts,
  passed. The new import-free helper binds ordered per-request asset identities
  and separate teacher/audio versus controller/video seeds. Public route wiring
  is still pending; no new executable bundle kind is advertised yet.
- `tao-two-request-decode-cpu-v1`:8 selected tests passed, including real FFmpeg
  encoding of synthetic RGB/PCM, exact243→240 frame and324000→320000 sample
  global timing, duplicate-prefix rejection and owned decoder failure cleanup.
  VAE and CUDA were substituted; this is media plumbing, not real VAE evidence.
- `taomate-third-request-cpu-v1`:41 selected tests passed, including the actual
  upstream teacher loader/continuation plans for two and three saved synthetic
  requests, prefix source indices and lifecycle counters/cleanup. No third GPU
  request or long-video quality claim. All these CPU runs kept1708 bound sources
  unchanged with CUDA uninitialized; counts overlap and must not be summed.

- `artifacts/h16/taomate-lifecycle-cpu-v1`:115 selected CPU tests; not a full suite.
- `artifacts/h16/taomate-upstream-lifecycle-cpu-v1`:20 selected tests including
  actual pinned upstream runtime subclass integration with synthetic forwards.
  Tests cover two commits, in-block/post-forward cancellation, factory/hook
  restoration, close rejection and a fresh runtime starting at request0.
- These are CPU lifecycle evidence, not two real requests, media or voice quality.
- `artifacts/h16/taomate-lifecycle-cpu-v2`:81 selected CPU tests passed with1707
  frozen sources unchanged. Includes Python3.10-compatible cleanup diagnostics.
- Added `taomate_stream_inputs.py`:CPU-only ordered prompt/text/seed/teacher
  binding. Each request's audio count comes from the pinned native continuation
  plan; second-request states cannot be substituted with the first request's
  states. Text metadata, raw5120 BF16 features, text tags, three finite F32
  milestones and consumer noise seeds are verified before allocating weights.
  `artifacts/h16/taomate-stream-inputs-cpu-v1`:32 selected tests passed,1708
  sources unchanged/CUDAfalse. Uses the actual upstream artifact loader and
  continuation plans with synthetic saved tensors; not genuine teacher inference.
- The real missing second Base10 audio teacher is now prepared by the owned
  `artifacts/h16/run_tao_second_teacher.py` controller, with exact existing first
  teacher reuse, previous40 clean audio latents, seed8302 and nine new forwards.
  Completion must be read from `tao-second-teacher-v1/terminal.json`; the script
  existing or its worker being alive is not proof of completed guidance.
- Actual two-request video worker/controller have been prepared in artifacts,
  not run or promoted into public execution. They retain one runtime, verify
  KV reuse/global offsets, strip native transport prefixes exactly as upstream,
  and preserve the full joint video/audio timeline for later VAE decode. The
  native shortest two-request delivery is10seconds (243 native→240 published
  frames); this is not a24/32second test. It cannot start before teacher success.
- The existing first-request worker and preflight still accept only one teacher
  request. Next prepare a genuinely matching second prompt/teacher with rollover
  audio, wire ordered request execution and offset-aware media assembly, then run
  the shortest two-request GPU case. Do not duplicate request0's teacher or relabel
  previously accepted recovered media as a reliable fresh generation job.
- Final integrated regression, package rebuild and formal-core delivery remain.
  Backend source fingerprints include the new lifecycle file, so stale prepared
  checkpoints are not silently treated as unchanged.
