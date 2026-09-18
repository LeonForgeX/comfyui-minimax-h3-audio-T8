# Meridian ConvRot INT8 — development evidence

Download: [t8star/Meridian-Comfy](https://huggingface.co/t8star/Meridian-Comfy).
Merge its `models/` into `ComfyUI/models/`: the converted main checkpoint belongs in
`models/meridian/`; the original Omega1B512 PT belongs in
`models/meridian/vggt-omega/checkpoints/`. Main weights retain the MiniMax H3 license;
Omega has its separate FAIR Noncommercial Research License. Pinned source/assets and H3 VAE
are still separate prerequisites. The historical conversion-only receipts below are not current pending tasks.

## Release1.85.0 (2026-09-18)

The converted ComfyUI native INT8 model, four-node camera/time integration and bound samples
were accepted by the user, including the corrected parallel slide. Use the
[formal workflows and asset layout](../examples/workflows/37-meridian/README.md).
Historical pending/P2-gate statements below are superseded only for those tested recipes.
Correct Omega1B512 is provided separately under its own license in the model repository, not an ordinary-VGGT substitute.
No full-precision Meridian inference or universal16GB performance claim is introduced.

## Current P2 increment (2026-09-18; overrides historical pending entries)

Authorized official Omega1B512 is now available and strictly loads all1411 keys.
Real single-image geometry and the pinned73-frame CPU camera warp completed.
The converted whole Meridian checkpoint actually ran250 native ConvRot INT8
layers and three forwards with DMD sigmas `[1, .857142806, .599999964, 0]` and
video/audio shifts3. No original/BF16 full Meridian model ran on GPU.

The first FP32 dynamic-VAE decode failed because pre_norm weight stayed on CPU;
its failed receipt and sampled latent remain unchanged. Independent static,
full-resident FP32 computation with the same VAE weights decoded that exact latent:
73 complete H264 frames,864×1184, no audio as upstream freeze mode.
MP4 SHA `4698b7800c6bb1daa61af590c6afd6d7a3fc048acf51f35d45019b889911dc75`.
Receipts: `artifacts/five-track-development-20260918/meridian-p2-gpu-v1/terminal.json`
and `meridian-decode-gpu-v1/terminal.json`. Decode122.547s; generation229.281s.
This is mechanical P2 completion, not human acceptance or a universal16GB budget.

The trained864×1184 bucket uses an explicitly recorded centred FOV crop from the
2:3 input; it does not stretch the portrait and is not the entire original canvas.
Core noise stream is also explicitly different from upstream seed bit identity.
P3/P4 presets, camera editor and public Meridian integration still require P2
human acceptance. Do not repeat conversion/diffusion or bypass that gate.

## Latest verified increment (2026-09-17, completed conversion)

- Conversion session59781 ended with exit0. The actual final checkpoint is
  34,038,894,278 bytes,535 converted tensors plus500 quantization auxiliaries
  (1035 serialized keys),250 quantized main-block layers. SHA256:
  `2c31fe6cd336b3d67cc23cdcb87965a925b6be6005062a228c0e2a8e0a8f0e67`.
  All17 bound input identities passed full source verification and postflight.
  Its receipt explicitly records CUDAfalse and no full-model GPU inference.
  The checkpoint exists as a finalized file, not a partial assembly.
- `artifacts/meridian/whole-native-cpu-v1/terminal.json` records successful
  **whole-checkpoint standard Core CPU strict loading**, with zero missing and
  unexpected keys,250 exact native INT8 QuantizedTensor layers,ConvRot enabled
  and `_full_precision_mm=false`. Complete SHA before/after matched. Loader
  time19.1595s and RSS5,838,942,208 bytes are CPU measurements, not a whole-model
  GPU budget. CUDA was not initialized; full-model GPU and human qualification
  remain false. Do not rerun conversion because an older entry says it is pending.
- `camera-geometry-contract-cpu-v1` passed32 pinned-upstream path cases
  (8 released frame lengths by realtime/freeze/half-time/hold-then-move),
  ties-to-even source-frame rounding, rejection of invalid key times/reverse
  source time, orthonormal camera rotations, synthetic identity reprojection,
  near-surface z-buffer selection, grey holes and pixel-centre intrinsic scaling.
  These are CPU mathematical checks only, not learned VGGT geometry or P2 media.

### Earlier increment (superseded where stated above)

- Real CPU conversion continues in `meridian-convrot-v1`; do not treat the
  per-layer state as a final published checkpoint. Read its actual `result.json`
  before whole-model loader qualification or resuming conversion.
- `artifacts/meridian/real-parts-GPU-v1/terminal.json`: three **real converted
  weight slices**, 128 rows each, loaded through native Core mixed-precision
  Linear. Input dimensions5376/2688/14336 used ConvRot groups256/64/256. The
  observed CK dispatcher selected `comfy_kitchen.backends.cuda.int8_linear`
  with CUDA input and INT8 weights; actual profiler events included
  `comfy_kitchen::int8_linear`. Relative L2 versus CPU-dequantized slice linear
  was0.00987/0.00899/0.01016. This is not comparison with the original full
  teacher, end-to-end quality, or evidence that the whole model fits16GB.
  The probe exited0; no full Meridian model was moved to GPU.
- `artifacts/meridian/native-layout-cpu-v1/terminal.json`: all8 released frame
  assets against all9 official canvas pairs (72 cases). Fixed5120-wide raw
  embeddings, two video references, video/audio indices, modality tags, condition
  row counts and expanded row timesteps matched the pinned Diffusers pure
  helpers and current Core. Float64 positions were bit-exact in8 cases; the
  remaining differences were bounded by1.4211e-14 (explicit tolerance1e-12),
  not advertised as universal bit-exactness. Patchification was exact.
  DMD sigma grid was `[1, 0.8571428061, 0.5999999642, 0]`, three forwards.
  No weights, VAE or geometry were run by this CPU contract probe.
- `artifacts/meridian/audit_complete_native_cpu.py` is prepared for the finished
  checkpoint: full SHA, exact1035 serialized keys,250 quantized layers and
  strict standard Core CPU loading. It **has not run on the complete checkpoint**;
  do not cite the script itself as a successful qualification.
- Whole-checkpoint completion/strict loading, actual geometry/conditioning,
  complete73-frame P2, user review and final-core delivery remain outstanding.

This is a partial P0/P1 implementation with completed conversion and strict whole-model
CPU loading, not a qualified whole-model GPU runtime, public node, or completed P2 sample. Existing node registration and
workflows are unchanged. Do not publish a model based on this document.

## Fixed inputs

- Meridian: `Viggle/Meridian@2083d059d8544ff7eaaf86966b83e4964a904737`.
- Full DMD SHA256:
  `8c3c331b4ead47a3e8b3b133dd61c288325686d94e2f4f50e949f5fad9b09f0a`.
- Its metadata declares ordinary rank128/alpha128 LoRA. Actual604 FP32 tensors
  form302 complete pairs, including `proj_in` and `proj_out`. The teacher has638
  indexed tensors in14 shards,66,280,430,080 tensor bytes. All14 teacher downloads
  completed; index/header verification is not full shard SHA verification.
- Upstream inference and model code are inspected from the fixed revision and
  Diffusers `d6726f3`. No upstream whole-environment installation was performed.
- Geometry expects `facebook/VGGT-Omega`1B512, revision
  `1041e80fc0e911235d3426b0a3d9a81075111a53`. The authorized local HF session's
  download returned403/manual-gated/not-in-authorized-list. No workaround,
  alternate416 model, or unapproved mirror is substituted. This prevents the
  actual geometry/P2 stage unless access becomes available; conversion can proceed.

## Implemented contracts

`h3_t8/meridian_conversion.py` is import-side-effect-free: no downloads, CUDA,
registry changes or full-model allocations. It provides an exhaustive rule list,
teacher index validation, exact DMD target/shape/math checks, FP32 CPU DMD fusion,
Q/K/V concatenation, SwiGLU half exchange, derived RoPE and row-chunked actual
Comfy Kitchen ConvRot INT8 serialization.

Main-block250 dense matrices follow the existing native full-H3 quantization
policy: group256 where applicable and64 for2688-input AdaLN. Norms, text refiner,
conditioning, time and input/output high-precision islands remain unquantized;
source dtype is preserved for their serialized merged values. FP32 video
projection DMD updates must not be omitted. There is no AdaLN curve pruning,
FastV2 substitution or double DMD application.

Returned quantized keys are native `weight`, `weight_scale`, and uint8 JSON
`comfy_quant` (`int8_tensorwise`, ConvRot true, exact group). Scratch rotation and
error measurement are row-bounded. Row chunk size belongs in future conversion
identity: CPU BLAS reductions at different sizes need not be bit-identical.
The converter now implements bounded checkpoint assembly, per-layer restart,
complete source SHA binding, immutable source/job identities and atomic publication.
Full checkpoint completion/loading and actual GPU dispatch are still required.

## Current CPU conversion implementation

- `h3_t8/meridian_checkpoint_io.py` validates actual safetensors payload ranges,
  hashes all copied bytes, streams without a full-model dictionary, and publishes
  with atomic no-replace semantics. OS-owned locks release on process death;
  no PID guessing or deletion of source/user files is used.
- `h3_t8/meridian_convert_job.py` binds implementation versions, environment,
  row chunk, CPU thread count, source root/17 identities and destination. Saved
  parts must match hashes, exact target keys and job metadata. Every original
  asset is revalidated even when recovering a fully published result.
- A publication journal is durable BEFORE the final no-replace link. Crashes
  before/after link or before `result.json` recover without repeating completed
  layer math. Unrelated existing output is never adopted. Incomplete downloads
  retain verified layer state and cannot produce a final model. Cancellation is
  checked between rows and file-hash chunks.
- `tools/convert_meridian_convrot.py --prepare-manifest` uses fixed Hub metadata
  and actual Git-blob identities for config/index, not local filenames as proof.
  The installed older `hf` lacks `models info`; its official SDK supplies this
  read-only metadata, without installing/upgrading the environment.
- HF download46536 completed35 files/14 teacher shards, exit0. The pinned17-file
  conversion manifest accounts for68,947,009,155 bytes including DMD/config/index.
  Download completion is not yet full local SHA verification: the real CPU
  converter performs that verification. Actual CPU conversion is running with
  4 threads/256 rows in research `conversions/meridian-convrot-v1`; no Meridian
  teacher/full BF16 GPU run was performed. Inspect its state/result before retry.
- `conversion-job-cpu-v4`:85 tests passed,1725 sources unchanged/CUDAfalse.
  Covers real tiny file conversion, corruption, wrong shapes/types, missing files,
  cancellation, three publication-crash windows, no-overwrite, pinned Hub blobs
  and source verification. Earlier v1 found Windows read-only fsync invalid;
  only the converter's own temporary file now uses r+b. Failures remain recorded.
- `native-core-cpu-v2`:5 tests passed,1726 sources unchanged/CUDAfalse. A complete
  tiny converted checkpoint strictly loads all native Core keys and separately
  passes the standard diffusion loader. Small64/256/2688-input matrices execute
  the actual CK INT8 linear operator with ConvRot metadata. These are CPU cases,
  not full-model GPU or quality proof. The earlier Python dispatch observer did
  not see nested QuantizedTensor calls; actual profiler events provide the proof.

## Evidence

- `artifacts/meridian/actual-mapping-inspection-v2.json`: full DMD SHA verified;
  all638 indexed teacher names consumed, native Core535 state names/shapes agree
  (534 mapped weights/buffers plus derived RoPE). Actual Core constructor uses
  meta parameters; its16-value CPU RoPE placeholder is explicitly identified.
  CUDA remained uninitialized. No teacher shard had completed at that snapshot.
  This receipt binds the then-current conversion implementation; later quantizer
  additions do not retroactively change that historical proof.
- `conversion-contract-cpu-v5`:39 tests passed,1719 source files stable during
  the run, CUDA unavailable/uninitialized. Includes FP32 low-rank math BEFORE
  layout transforms, exact target coverage, bad metadata/nonfinite rejection,
  actual CK CPU quantization, native safetensors save/reload and cancellation.
- The small fixture's INT8 values agree with whole-matrix CK quantization.
  Scales differ slightly for1/7-row chunks due to FP32 reduction order; both
  were verified against an independent FP64 Hadamard oracle and the derived
  dot-product rounding bound. No full-model bit-exactness or quality is asserted.
- Initial plain-pytest invocation lacked Core PYTHONPATH and collected no tests.
  Contract v1 exposed an incorrect largest-divisor128 group selection; corrected
  to the pinned native policy. v4 retained the overstrong bit-exact-scale failures;
  v5 tests the numerical contract above. Failure evidence is not relabelled pass.

## Continue

1. Reuse the finalized checkpoint and conversion receipt above. Preserve source
   shards and resumable parts; do not repeat the completed conversion.
2. Audit the completed model, native loader and actual GPU ConvRot dispatch.
   Tiny CPU success does not substitute for these gates. No original/BF16
   full-model GPU run is permitted.
3. Finish geometry/packed-reference/time-clock contracts and owned job lifecycle.
   Once the correct geometry weights are accessible, run the native shortest73
   frames with the fixed DMD grid (4 points/3 forwards), both shifts3,480-class
   references and768-class target. No silent resolution downgrade.
4. Complete P2 media/resource audits; include the new result in collective human
   review. Full front-end presets/editor are later P3/P4, not already implemented.

Only isolated development files changed. Final qualified code must be integrated
into the formal project's `h3_t8`, not the ambient unrelated node directory.
