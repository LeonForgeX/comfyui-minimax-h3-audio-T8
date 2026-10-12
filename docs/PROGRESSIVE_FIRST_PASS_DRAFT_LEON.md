# Leon fork: public first-pass drafts and finished-stage residency

This opt-in source extension adds capability `t8.progressive.first_pass_draft.v1`
for Leon v8.32. Existing sampler ports, output types and omitted-option numerical
behavior remain unchanged. The upstream Registry version stays 1.86.0.

## Public sampler contract

`MiniMaxH3ProgressiveSamplerEXPT8` appends optional forced-input ports:

- `draft_mode`: `disabled` (default), `save_or_reuse`, `reuse_only`, `save_new`.
- `draft_directory`: absolute directory owned by the caller.
- `draft_scope_json`: nonempty JSON project/pack/accepted-parent/media identity.
- `draft_id`: optional exact selected LOW cache SHA256; mismatch fails before sampling.
- `high_seed`: optional unsigned 64-bit override for HIGH video noise and HIGH sampler.
- `stage_release`: `disabled` (default) or `finished_stages`.

Omitting `high_seed` preserves the previous HIGH video noise seed `(seed+1)%2**64`
and previous native HIGH sampler seed. The default route does not create files or
change resource residency. LOW draft persistence requires CFG1, complete actual
LOW callback/forward evidence and an exclusive OS-backed directory lock.

A native Progressive LOW draft contains `clean_video` and **noisy** `audio_next`
float32 tensors in sampler coordinates, captured before the learned lift. It is
not a completed AV result, accepted segment, legacy `low_x0`, or HIGH sidecar.
Both `empty` and independently prepared `prepared_pair_exp` input routes retain
native source/mask/time/audio behavior. LOW initialization never comes from
resizing HIGH known-region masks.

The LOW key hashes the actual LOW model weights, LoRA bytes, authenticated patch
owners, actual LOW conditioning/source/mask, seed, full schedule/cut/geometry,
caller scope and implementation. HIGH weights/LoRA/seed and the selected learned
lifter are verified separately and do not invalidate LOW. Schedule or geometry
changes remain conservatively incompatible. Native long-video payload patches
are authenticated by their exact code and native original owner. Unknown opaque
LOW stacks can still sample; portable draft reuse is disabled. A Relay LOW draft
currently needs an additional dedicated identity adapter and is not portable.

`reuse_only` never silently reruns LOW. `save_new` never overwrites an existing
immutable draft: use a new owned directory/LOW seed or choose reuse. Changed or
corrupt receipts, bad SHA, missing noisy audio, invalid geometry/dtype/NaN and
path escapes produce real errors.

## Durable completion and reports

The atomic completion marker is `low-boundary-<cache_id>.json`; its only tensor
file is `low-boundary-<cache_id>-<uuid>.safetensors` in the same directory. Receipt
fields are `schema=1`, `capability`, `cache_id`, `contract`, `tensor_file`,
`tensor_sha256`, `tensor_size_bytes`, `low_report`, `report_sha256`, `created_at`.
`contract.scope` preserves caller identity, and `contract.plan` defines shapes.

`cache_id` and `report_sha256` use SHA256 of UTF-8 JSON from
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`. Safetensors compression is `none`. Orphan tensors never count
as complete, and incomplete LOW never publishes a receipt. Completed LOW remains
on disk when lift, HIGH sampling or later decoding fails.

The final sampler report includes `first_pass_draft` with capability, cache ID,
receipt path, bytes, integrity SHA, saved/reused status and historical LOW report.
Current replay callback/forward counts are LOW=0; historical LOW counts are never
relabeled as current execution. Catalog readers can inspect headers for metadata
and must verify tensor SHA/integrity before reuse or deletion.

## Registered integration helpers

`MiniMaxH3FirstPassFingerprintT8` consumes MODEL, positive CONDITIONING, AV LATENT,
SIGMAS, seed and `scope_json`, returning JSON with `fingerprint`, `contract`,
`portable_reuse`, `invalid_reason`. It does no sampling or weight mutation. Leon
can persist its legacy denoised LOW AV output under that identity while keeping
upscale/reconcile/second-pass math in existing T8/Core nodes.

`MiniMaxH3StageResidencyReleaseT8` accepts optional `model`, `model_hires`, `clip`,
`video_vae`, `audio_vae`, `vae`; at least one actual Core patcher is required. It
uses the existing finished-stage release transaction, preserves unrelated loaded
models and handles Dynamic Core's completed-node cleanup only for executed owned
stages. Native Progressive releases finished LOW before lift and finished HIGH
after sampling when selected. Callers can release conditioning models before
sampling and finished DiTs/native VAE before optional HyperVAE decode.

Resource reports contain before/after boundary snapshots, owned release counts
and preservation counts. They are not peak predictions, and do not guarantee
that arbitrary model/length/resolution combinations cannot exhaust VRAM.

## Validation boundary

`tests/test_progressive_first_pass_draft.py` exercises the registered public
sampler and helpers with real tiny Core H3/Euler CPU sampling. Learned weights
and media VAEs use labeled test doubles. It checks bitwise default/replay AV,
new HIGH LoRA/seed with zero LOW forwards, native context/source audio,
incompatible LOW rejection, failed-HIGH durable receipts, incomplete-LOW
nonpublication and unrelated residency preservation.

Full trained-model Windows GPU generation, visual quality, speed and peak VRAM
are NOT_RUN for this source extension. Existing accepted-picture-to-LOW seam
math, old examples, audio clocks and optional HyperVAE decode math are unchanged.

## 2026-10-12 native Bypass lifecycle compatibility

ComfyUI commit `d91ed5f5b7fa60fa18464c2ad7c80254da2f0f29` added
`original_force_forward` to native Bypass hooks on their first injection. The
field remains after eject. The previous exact four-field classifier then lost
native ownership and incorrectly reported `First-pass draft execution inputs
changed`, including after otherwise successful HIGH sampling.

The classifier now recognizes only the native four/five-field schemas and
plain boolean force-forward state. It binds the original flag through the
authenticated installed forward chain, including an active LOW clone sharing
the pending HIGH network. Extra fields, replaced owners, changed force state,
and changed base/adapter tensor bytes remain observable. Verification is not
disabled, and no sampler, audio, clock, sigma, or learned-upscaler math changes.

LOW Bypass stacks remain `nonportable_not_saved`; this fix does not make their
drafts portable. Plain LOW with HIGH-only Bypass can save LOW and retry a new
HIGH LoRA/seed. Start a fresh draft after this implementation update: an old
implementation identity is not force-reused.

With official Core d91 and Torch 2.8 CPU, the original six native Bypass cases
fail; the repaired full first-pass suite has 102 passed and 3 skipped. One skip
is a subprocess worker entry (its parent replay test executes); two require an
optional actual KJ checkout. Twelve added cases cover lifecycle, readonly
classification, shared clones, and genuine mutations. An independent real
Linear/LoRA CPU probe passes 30 checks, and the official pre-change four-field
lifecycle passes six checks. Tiny H3/Euler CPU sampling and labeled lifter test
doubles do not certify trained-model GPU output or arbitrary KJ/Sage stacks.
Windows GPU acceptance for this repair remains pending.

## 2026-10-12 review-time retry and byte-identity performance repair

This candidate supersedes the earlier LOW Bypass limitation above for the exact
source-authenticated native Core inference LoRA factory only. Its ordered
targets, strengths, compute weight bytes, configuration and source implementation
are bound to the LOW contract. Skip-injection, stale loaded flags, replaced live
methods, extra owner state and unknown stacks remain nonportable; their normal
generation is still allowed. Global stage/outpaint cache policies are unchanged.
Read-only native TAEH3 preview projection retains the same draft identity.
Inspection clone cleanup cannot eject the selected source's installed LoRA.

A completed LOW draft is available before Accept. Retrying HIGH preserves the
original LOW Take/seed and current material pack; current LOW forward count is
zero on replay. Only the user's subsequent Accept changes the canonical chain.
Old implementation-bound drafts are not migrated or force-reused; generate one
fresh LOW after updating this candidate.

Fresh full tensor checks retain their original SHA and bounded transfer sizes.
The CPU finite check uses exact FP16/BF16/FP8 storage bits, and SHA consumes a
read-only memory view without another bytes copy. Every boundary still checks
actual weight bytes. This does not change sampling, LoRA math, clocks, sigma,
learned lift, offload, image/audio output or quality settings.

The report separates `draft_bind_and_load_seconds`,
`draft_save_and_verify_seconds` and `draft_final_verify_seconds` from LOW, lift
and HIGH timing. `sampler_seconds` includes the final verification. These are
host wall-clock observations without forced CUDA synchronization.

With official Core d91/Torch 2.8 CPU, the complete first-pass suite has 122 passes
and 3 conditional skips. It covers saved/replayed native LoRA, zero LOW forwards,
shared/separate models, changed HIGH seed/model, active input preservation,
FP8 target lifecycle, native preview and genuine corruption/mutation rejection.
The focused byte-identity tests classify all half/FP8 storage patterns against
actual Torch and preserve original finite tensor SHA. CPU chunk benchmarks show
a faster check; trained-model RTX 5090 end-to-end speed and review acceptance
still require the user's real GPU result. No Registry release is made.


## Python 3.13 native Bypass draft authentication (2026-10-12)

CPython 3.13 adds inert class metadata `__firstlineno__` and
`__static_attributes__`. The draft adapter now validates those fields against
the loaded Core source and, when the interpreter emits them, the compiled
class-body constants. It still rejects foreign class members, changed live
code/defaults, non-native injection factories, wrong selections and corrupted
weights. The public/global cache identity policy is unchanged.

Nonportable generation skips the duplicate bind/read scan and the full scan
for a save that cannot publish a receipt. Fresh initial and final LOW/HIGH
weights, LoRA and selected execution identities remain checked. A nonportable
run publishes no draft; `reuse_only` still refuses to resample LOW. The report
adds fixed `nonportable_reason` and optional `model_identity_reason` codes,
without exposing arbitrary exception text. Sampling math, audio and NFE are
unchanged. Native CPU regressions and source-bound 3.13 metadata fixtures do
not substitute for the user's Windows/Python 3.13 GPU replay confirmation.
