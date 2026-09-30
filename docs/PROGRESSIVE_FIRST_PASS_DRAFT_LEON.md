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
