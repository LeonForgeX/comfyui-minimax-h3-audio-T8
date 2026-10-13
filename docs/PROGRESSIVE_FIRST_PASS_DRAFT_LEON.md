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


## Stock Core LoRA loader metadata (2026-10-12)

The native Core `load_lora_for_models` path now preserves the safetensors string
map in a `lora_metadata` attachment. The prior classifier recognized only T8's
`t8_h3_lora_metadata` name, so adding an ordinary LoRA after native Bypass made
the otherwise verified LOW stack nonportable. Tiny native loader replay
reproduces this refusal; an isolated Bypass factory fixture missed it.

Bind the exact plain Core string map alongside the actual patch bytes, strength,
order and native injection proof. Invalid metadata values remain errors, and
unrelated executable attachments remain nonportable. The live model and its
metadata are not removed or changed. When native Bypass authentication succeeds
but another owner fails, retain that actual refusal instead of restarting the
generic classifier and hiding it behind the already authenticated injection.
Fixed report codes distinguish attachments, attention, wrappers and live hooks.

Tests load a real synthetic safetensors file through native Core LoRA loaders,
then exercise tiny H3/Euler save and HIGH-only retry, separate/shared models,
changed HIGH seed/LoRA, changed metadata, malformed metadata and unknown owners.
The KJ memory combination uses retrieved upstream source and an explicitly
labelled CPU Sage kernel double. These tests do not authenticate arbitrary KJ
preview wrappers, Sol variants, user workflows or trained-model GPU execution.
The user's latest full stack has not yet been identified; do not describe this
focused correction as confirmation that their complete workflow is repaired.


## Actual KJ Preview + SolAttnPatch workflow repair (2026-10-12)

The supplied workflow routes LOW through native Bypass, multiple ordinary
LoRAs, KJ memory-efficient Sage, Kijai SolAttnPatch and ModelPreviewOverrideKJ.
The previous metadata-only repair did not cover these complete owners. The
uploaded workflow, prompts, model filenames and material data are not stored
in this repository.

The draft-only adapter now inspects the source-bound KJ observer and Sol
selector, composed KJ forwards, native layout delegate and complete hook set.
Inspection does not remove previews, change forwards/hooks or change sampling
settings. The reviewed Sol scope is H3, no Morton reordering, no block probe,
no depth-specific gates, no earlier custom attention override, and exact KV
conditioning (with or without exact query rows), composed with KJ Sage.
Arbitrary replacements and different source revisions remain nonportable;
normal generation retains the selected stack. Global cache rules are unchanged.

Reviewed third-party sources: kijai/ComfyUI-KJNodes at
`d7641533d6634eb05f8570c980fce8fc8528dc7b` (preview and tiny VAE) and
kijai/ComfyUI-SolAttn_triton at `26d816ebd4f1e43a2c6e4d4759be3137f10a7a73`.
Source fingerprints normalize Git CRLF only and trailing whitespace at EOF.
Live function code, defaults, closure owners and selected settings are checked;
names or matching on-disk files alone do not authorize the adapter. Installed
kernel source/entry points and Triton version are bound to the LOW key.

For plain native readers, LOW save performs one fresh verification after the
small tensor write before receipt publication, avoiding the redundant scan
immediately before that write. Initial and final LOW/HIGH checks remain.
Effectful readers and continuation providers keep their prior pre/post checks.
No address/version-only weight cache, stale hash, forced portable flag or
synthetic draft receipt is introduced.

CPU tests load the actual third-party node factories and run native tiny H3
Euler with synthetic LoRA/tiny-decoder weights. KJ preview callbacks execute.
The Sol GPU kernel-contract boundary and Sage GPU kernel are explicitly
doubled in CPU mechanics tests; GPU Sol execution, trained-model picture/audio
quality, real Windows Python 3.13 and RTX 5090 end-to-end speed remain NOT_RUN.
The matching toolkit checks the registered KJ/Sol owners and installed source
files before asking for GPU work. It never queues a prompt. If sources differ,
return that small environment report before another generation.


Fresh draft state scans now hash independent, inert CPU weight tensors with at
most four workers when the candidate state is at least32MiB. CUDA tensor copies
remain on the caller's thread/stream, and the whole capture stays sequential
when any tensor/container has custom readers or unrecognized state. SHA schema,
finite checks, per-worker bounded chunks and fresh-boundary checks are unchanged.
No content identity is retained between calls.

## Python 3.13 KJ source fingerprint repair (2026-10-13)

The supplied Windows diagnostics still reported `model_stack_unverified` and
`saved_low=false` on the paired source. Python 3.13 changed `ast.dump()` to omit
empty optional fields by default. The same audited KJ Sage function consequently
had a different AST digest from Python 3.12 and failed the composed Sol/KJ draft
adapter. The prior 3.13 class-metadata fixture did not exercise a real 3.13
interpreter and missed this separate incompatibility.

On Python 3.13 and later, the shared KJ AST fingerprint now explicitly includes
empty fields. This reproduces the existing Python 3.12 representation and pinned
fingerprints; it does not accept new source revisions or bypass live-code checks.
KJ preview, Sol settings, LoRAs, native Bypass and all sampler math are retained.
Specific Sol/KJ owner failures now report the existing attention/memory reason
codes instead of the generic stack code.

A source inspection previously parsed and compiled the same KJ file once per H3
block. It now compiles each exact path/payload once within that inspection,
rereading source and verifying live code/globals for every block. The temporary
parse results are discarded at the end of each inspection; model identity and
weight scans remain fresh at each required boundary. This removes 49 redundant
compilations for a 50-block model without introducing a cross-run proof cache.

Reproduction and regression environment: actual CPython 3.13.11, ComfyUI
`8010b48d2c1be986613f8e8c761527d205bd756f`, Torch 2.9.1 **CPU**, Triton 3.5.1,
and the audited KJ/Sol revisions above. The original KJ fingerprint fails in this
environment; the corrected fingerprint matches the existing Python 3.12 digest.
The new integration test imports the actual Sol JITFunction/Autotuner owners
and uses the production kernel identity verifier without its previous test
double. Native tiny H3/Euler saves LOW and replays the same pack with zero LOW
and two HIGH forwards, matching a fresh CPU baseline bitwise. A replaced kernel
entry point still disables portable reuse. Sage computation and learned lift
remain labelled CPU doubles; Sol takes its existing CPU dense branch.

The affected suite, including all five required seam/cache modules, has 309
passing cases and one conditional worker-entry skip. This combines the full
batch with the focused rerun after correcting a new mutation fixture's source
filename; no production failure was suppressed. Python 3.12 fingerprint parity
was also checked against the actual reviewed source. Trained weights, CUDA
kernel launches, Windows execution and RTX 5090 end-to-end speed remain NOT_RUN.
The source/API environment check alone is not proof of a saved draft or a GPU
replay. The paired toolkit additionally fingerprints `nodes/ltxv_nodes.py`, the
file actually containing KJ H3 Sage execution.

After updating and restarting, create one fresh LOW draft: a previous run with
`saved_low=false` has no durable LOW receipt to recover, and old implementation
identities are not force-reused. Keep the current pack unaccepted when testing
HIGH-only retry; accepting it is not a prerequisite.

## Native INT8 validation cost repair (2026-10-13)

The user confirmed successful HIGH-only replay on the previous pair. Its latest
diagnostics show 191.022 seconds total, 21.562 seconds in HIGH, 101.128 seconds
in draft bind/load and 22.154 seconds in the final guard. Those two validation
boundaries consumed 123.282 seconds (64.54%). Initial save took 223.069 seconds,
including 72.198 seconds bind/load, 43.276 seconds save verification and
21.421 seconds final verification. These are reported user timings for the
previous pair, not a benchmark of this update.

The selected models use native INT8. Earlier cost fixtures used ordinary
floating-point modules; the read-only classifier rejected Core's native
mixed-precision Linear serializer and Kitchen QuantizedTensor, and retained the
conservative repeated captures. This is now reproduced with actual Core/Kitchen
INT8 storage, scales, metadata and nonzero LoRAs. A narrow source/live-code audit
recognizes these pure native readers. Unknown serializers, hooks, replacement
layout owners and custom tensor readers keep the original fallback.

For an explicit selected draft, the locked operation first checks the receipt,
tensor SHA, dimensions, finite values, size and completed LOW evidence. It then
computes current LOW/HIGH identities and compares the selected LOW contract
before returning anything to sampling. Changes during file loading still fail.
Public bind/load calls retain their separate fresh checks. No model hash or
source proof survives across validation boundaries. Final LOW/HIGH verification
and post-write verification before receipt publication remain mandatory.

For separate native LOW/HIGH models, full model captures fall from 12 to 6 on
initial save, and from 10 to 4 on exact selected replay. The 60% replay reduction
is a count of scans, not a promise of 60% less total GPU time. Source AST parsing
is bounded to once per source/symbol within each inspection, including a
64-layer regression; live code/defaults still check each reader.

CPU coverage uses real INT8 quantization, native H3/Euler, LoRA/Bypass loaders,
actual KJ/Sol/preview factories and real Triton JIT/Autotuner identity owners.
Same unaccepted pack replay is LOW=0/HIGH=2 and bitwise identical to fresh CPU
generation with the same HIGH seed. Mutated raw INT8 bytes, scales, LOW LoRA,
conditioning, receipt/tensor/report integrity and source owners are rejected.
Sage computation and learned lift remain labelled CPU doubles; no CUDA kernel,
trained-weight quality or end-to-end RTX 5090 speed is qualified by these tests.

Install the exact paired update and restart. This changes implementation identity:
create one fresh saved LOW before measuring HIGH-only replay on the same
unaccepted pack. Existing outputs and old receipts remain intact; old receipts
are not silently relabelled as compatible.

Affected regression batch: 299 passed, one conditional subprocess-worker entry
skip, including all five required seam/cache suites. The first new-test run
found a fixture trying to mutate a frozen dataclass; replacing that fixture's
scale descriptor retained the rejection assertion. No production failure was
suppressed. Workflow source bytes are unchanged; Core schema/serialization
validation uses explicitly synthetic catalog files and queues no generation.

## External Bypass LoRA receipt/catalog repair (2026-10-13 follow-up)

After the INT8 speed update was delivered, the user reported that connecting
external LoRA hides drafts and disables HIGH retry. A real Core/T8 CPU producer
reproduced this independently of portable-stack classification: native Bypass
LoRA on 258 H3 targets completed LOW/HIGH with saved_low=true, but wrote a
12,014,149-byte JSON receipt. Leon's catalog rejected it at its 512 KiB JSON
limit, leaving zero entries and unavailable_count=1. Prior one-target LoRA
fixtures never exercised a full-model receipt.

Core shares the loaded-key set across adapters. Exporting the complete native
Bypass contract repeats that set for every target. The T8 receipt now contains
its schema, selection/target counts and SHA256; the FULL freshly authenticated
contract is still included in the model content SHA before this summary is
returned. No LoRA target, selected strength, weight, source owner or loaded key
is excluded from validation. The same real producer's receipt becomes 53,728
bytes; its LOW tensor SHA is unchanged. A changed loaded key changes the digest,
and existing adapter/weight/owner mutation guards remain active.

The matching Leon update uses T8's 16 MiB bounded native-receipt budget for
reading, hashing, catalog comparison and selected-replay authorization. Legacy
drafts and UI JSON retain 512 KiB. Duplicate keys, excessive depth, nonfinite
values, symlinks, bad receipt/report/tensor hashes, wrong scope and missing
completed LOW evidence still fail. The original 12 MiB receipt now passes its
file-integrity read; this does not bypass implementation compatibility for
sampling or claim old drafts match the new T8 implementation.

Integration evidence: the actual new T8 receipt is listed by Leon's live catalog,
retains the original pack/Take/seed and passes its selected-replay authorization.
The many-target native T8 regression saves and replays at LOW=0/HIGH=2 with
bitwise CPU parity. Leon storage/API regression covers large compatible/stale
receipts and every corruption guard. CPU synthetic weights/lifter cannot
qualify trained GPU output; current GPU timing remains pending.
