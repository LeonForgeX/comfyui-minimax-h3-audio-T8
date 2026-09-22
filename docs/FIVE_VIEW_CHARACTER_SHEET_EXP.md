# MiniMax H3 five-view character sheet (EXP)

This experiment turns one authorized reference image into five jointly sampled,
independently decoded still images. The five H3 latent positions are **image
slots**, not ordinary video frames. It does not modify or replace T8 Still,
Ref2VA video, Director, HyperFlow, or long-video generation.

## Sources and prerequisites

- Protocol: [author's contact-sheet node, fixed source](https://github.com/matlowai/ComfyUI-H3-ContactSheet/blob/3c8c8c66e69362e5abf9cf4c4e693baddab90ca3/__init__.py).
- Weight and limitations: [matlod H3 turnaround model card](https://huggingface.co/matlod/minimax-h3-turnaround).
- Tested weight: `minimax_h3_five_view_512_s1500.safetensors` from Hugging Face revision
  `791cceae539e19b297c08e26ff9d8246d231b89a`; 63,103,776 bytes,
  SHA-256 `9bc12ae97d2076b3449f8b1feca1fff52252576d608a84e4625497644f265d35`.
  Place it in ComfyUI `models/loras`. Model weights are not included in this source package.
- Use the H3 Ref2VA **pruned** base, H3 Qwen3-VL CLIP, and H3 video VAE. The example
  selects the installed INT8 ConvRot base, NVFP4/AWQ text encoder, and FP16 video VAE.
  The turnaround LoRA's 100 attention modules were attached in the actual test;
  this does not qualify other backbones or arbitrary LoRA stacks.
- Example frontend workflow:
  `examples/workflows/03-image-video-edit/2026-09-21_H3_Five_View_Character_Sheet_EXP.json`.
  Replace the placeholder image with a reference you have the right to use.

## Graph contract

`MiniMaxH3FiveViewConditioningEXPT8` uses one `<Picture 1>` reference,
aspect-preserving down-only resize, native MiniMax reference conditioning,
video latent `[1,24,5,S/16,S/16]`, and audio latent `[1,32,2,28]`.
The graph uses stock `LoraLoaderModelOnly` at strength 1.0 and stock
`res_multistep`/`simple` sampling for 28 steps, with the existing T8 sampling
setup used only to supply native H3 AV sigmas and sampler. The image slots
are denoised together in one pass.

`MiniMaxH3FiveViewDecodeEXPT8` rejects a latent lacking the dedicated protocol
marker. For each of exactly five slots it duplicates that one latent token into
a legal two-token VAE clip, decodes the clip separately, and keeps only the first
pixel frame. Outputs are a five-image batch and a horizontal strip. Passing the
five-slot latent through the ordinary Still or video decoder is not equivalent.
ComfyUI's stock `SamplerCustomAdvanced` preserves the protocol marker when it
copies the input latent to its sampled output (verified against the installed
Core source and by the run below).

Do not leave this turnaround LoRA connected when switching back to normal video
generation: the author documents degraded motion in that use. A five-view sheet
is a candidate reference asset; inspect it before adding views to a project's
shared reference library. No automatic import or silent Director setting change
is performed.

## Actual isolated validation, 2026-09-21

The test used an isolated ComfyUI 0.36.0 Core on port 8859 and project-bundled
public image `examples/workflows/26-h3-world/assets/h3_world_official_first_frame.png`.
It did not queue against the user's Core or mutate a Director project.

- Prompt ID: `3ba499d0-3b15-40ef-89f2-e1c107a10462`; Core history status
  `success`, no cached nodes, 57.04 s total.
- H3 Ref2VA pruned INT8 loaded with exactly 100 LoRA patches; 28/28 sampler steps
  completed and all five independent VAE decodes succeeded.
- Saved five distinct 512×512 RGB PNGs and one 2560×512 strip. Strip SHA-256:
  `21c7844951bceffc3c71f6c59255bab5f696922b401eacb3a351d4c52b0a9721`.
  Private run media and Core data remain local in the development tree under
  `G:/CodexHome/.codex/worktrees/t8-h3-final-20260920/artifacts/development/fiveview-20260921/isolated-core/output/MiniMaxH3_FiveView_EXP/`.
- During sampling one GPU observation was 15,574 MiB used / 536 MiB free on an
  RTX 4060 Ti 16 GB. This is a sampled observation, **not** a measured peak or
  a guarantee that other processes, references, sizes, or model combinations fit.
- The inspected result maintains the small game character and environment across
  the set with visible viewpoint/pose progression, but does not show a guaranteed
  standard-angle turnaround. This particular public test image is a wide scene
  with a small subject; subjective identity and angle quality on the user's
  intended characters remains unreviewed.
- The isolated service was stopped after the run; port 8859 no longer listens.

## Clear-back redo after user review, 2026-09-22

The earlier parking-garage input shows a small walking subject and produced
five motion-like views without a legible direct back. A new single full-body
front reference was prepared from the project-bundled subject. An isolated
Core on port 8878 ran the same installed `512_s1500` turnaround LoRA with
28-step `res_multistep`/`simple`; the first overly enumerated prompt
(`artifacts/development/fiveview-redo-a3725c2918a5/`) completed but packed
multiple figures into later slots, so it was **rejected visually**.

The second run simplified the instruction to one standing person per slot,
camera moving clockwise from direct front to direct back; independent seed
`26092206`. Its local evidence is
`artifacts/development/fiveview-redo-6c7f95fb42b8/` (prompt ID
`95bfb1c7-23dd-429b-9b16-bd429cbb9766`). Core status success, no cached
nodes, 28/28 steps, five separate 512² images plus strip; the owned Core was
stopped. The fifth image is visibly a full-body direct back with rear head,
shirt and legs and no face. The first two angles are close, so this is a
**clear-back sample, not a strict equal-angle or cross-subject guarantee**.
The native strip SHA-256 is
`369b048fd1867816835f29d119e306cc6d30f5ca84555dd4850f044a0dc81452`.

A separate, visually more regular five-angle image at
`artifacts/development/fiveview-corrected-20260922/five-view-turnaround-v2.png`
was generated with an image-generation tool; it is **not** a native five-view
LoRA result. The [author's model card](https://huggingface.co/matlod/minimax-h3-turnaround)
warns of under-rotation and lists 180°/360° spans as future v2 work; neither
prompt wording nor one successful back view should be marketed as guaranteed
180° performance. Both visual outputs and provenance are in
`artifacts/review/2026-09-22-0p6mp-fiveview/` for user inspection. No automatic
Director import was performed.

CPU tests verify the exact latent contract, reference token/latent payload,
separate two-token decode for each slot, protocol-marker guard, malformed inputs,
and frontend/API graph links. They are not substitutes for the actual GPU sample
or user review of five-view quality. The author's model card explicitly warns
that angle, pose and identity transfer can be imperfect; neither this example
nor a green graph promises a correct 90°/180° turn for every input.
