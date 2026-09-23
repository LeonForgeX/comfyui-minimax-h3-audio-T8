# Independent Ref2VA Progressive — Leon fork EXP

This extension builds on `feat/progressive-fl2va` (commit `24924e0`). The public
`MiniMaxH3ProgressiveSamplerT8` node adds `task=ref2va` at the end of its task list.
Use Ref2VA weights for both LOW and HIGH models and an empty native AV template.
The existing native Euler/AV schedule, learned video lift and audio trajectory
remain in use.

## Reference contract

- Native `minimax_refs` may contain images, silent videos, videos with reference
  audio, and standalone audio; native per-modality limits are 9/3/3 (15 blocks).
- Reference tensors, order, native spatial dimensions and time lengths are kept
  in both stages. Core's packed layout assigns each reference its own coordinates;
  reference canvases do not follow the generated video's LOW canvas.
- Text/vision token tags and embeddings are preserved. Reference video soundtracks
  remain paired with their video blocks. Standalone audio is conditioning, while
  the output contains newly generated joint audio.
- Shape, dtype, finite values, native time grids, metadata and reference counts
  are checked before sampling. At least one positive reference is required.
- First/last-frame anchors and initialized/source AV are outside this independent
  route. V2V/RV2V and continuation require their own source contracts.
- Existing user hooks remain attached with advisory warnings for unqualified
  combinations, following `PATCH_STACK_POLICY.md`.

## Leon integration and verification

Use `LeonForgeX/minimax-h3-director-Leon` branch
`feat/v8.28.0-progressive-ref2va`, select R2V and `progressive_exp`, disable Context,
and choose the intended local/global reference media. Restart ComfyUI after both
extensions are updated. Updating Leon alone cannot add a missing T8 task.

CPU tests cover every native reference kind, preserved tensors/order/tags, invalid
metadata, limits and the initialized/anchor boundaries. They exercise the native
schedule contract and conditioning preparation, without running trained H3
weights. Windows GPU generation, reference fidelity, output quality and peak VRAM
remain unverified for FL2VA/Ref2VA. Reference tokens remain present at LOW, so no
fixed speed or VRAM reduction is promised. This is a fork experiment; Registry
metadata and upstream release/version files are unchanged.
