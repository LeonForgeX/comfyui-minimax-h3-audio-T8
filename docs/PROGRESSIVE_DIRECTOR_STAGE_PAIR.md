# Public prepared progressive stage inputs (Leon fork, EXP)

This source extension stays on `feat/progressive-ref2va` / PR #2. It does not
change the upstream Registry version or publish a Registry release.

The public `MiniMaxH3ProgressiveSamplerEXPT8` node accepts optional inputs appended
after its existing ports: `input_mode`, `av_latent_low`, `positive_low`, and
`negative_low`. `input_mode` is a forced input to preserve existing positional
widget values. Omission retains `empty` and the qualified independent path.

`prepared_pair_exp` consumes HIGH AV/conditions from the existing inputs and LOW
AV/conditions from the three new inputs. It validates the native H3 spatial and
temporal grids, embeddings, motion keyframes, user references, audio context and
both masks before sampling. LOW source and mask are never inferred by resizing
the HIGH initialization. The learned lift, native Euler ladder and carried audio
sampler state remain the existing composed runtime. HIGH inpainting uses the
clean HIGH source, never the lifted prediction or noisy restart, as its anchor.

T2VA/I2VA/FL2VA/Ref2VA pairs can contain native motion conditioning and explicit
source-audio masks. The caller must prepare and authenticate those inputs and
install the existing long-video MODEL patch. Leon v8.29 does this through its
canonical accepted-parent, RGB-tail and raw-HIGH-sidecar contracts. Task labels
do not authorize an arbitrary parent, a source-video repaint algorithm or a
different model family. Existing patch owners remain preserved/delegated.

`initialized_av_exp` retains its previous strict task scope. The original
empty-AV path, private T8 continuation/checkpoint contracts, dual-pass workflow
recipes, examples, audio clocks and Registry version remain unchanged.

Tests distinguish actual tiny CPU Core H3/Euler execution from fake VAE/lifter
weights. No trained-model, Windows GPU, seam-quality, speed or VRAM claim is
made for the new paired entry.
