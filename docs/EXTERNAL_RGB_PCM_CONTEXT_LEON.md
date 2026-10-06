# External RGB/PCM context for Leon branches

This source-only optional API prepares native H3 continuation context from the
actual selected delivery picture and original PCM. It does not sample, create
an accepted Director parent, recover the original latent, or certify quality.
Registry version 1.86.0, node IDs, existing workflows and sampler mathematics
remain unchanged. The installed package explicitly loads
`h3_t8.external_rgb_pcm_context` for runtime adapter discovery.

## Public API

```python
source = capture_external_rgb_pcm_source(
    root, relative_path, media_sha256,
    start_frame=0, end_frame=cut_frame, context_frames=22,
    audio_policy='reencode_stereo_pcm_context',
    source_binding_json='{}',
    audio_relative_path=None, audio_sha256=None)

low_context, high_context, report = prepare_external_rgb_pcm_contexts(
    source, video_vae, audio_vae,
    low_width=low_width, low_height=low_height,
    width=width, height=height, context_audio='video_and_audio')

identity = validate_external_rgb_pcm_context(
    high_context, source=source, video_vae=video_vae, audio_vae=audio_vae)

# Paired Leon v8.39.1 guard: fresh source/tensor checks for both contexts,
# one producer scan per inert native VAE within this call only.
low_identity, high_identity = validate_external_rgb_pcm_contexts(
    (low_context, high_context), source=source, video_vae=video_vae, audio_vae=audio_vae)

high_latent, prefix_report = apply_external_high_prefix(
    high_latent, high_context, source=source, preserve_existing_mask=False)
```

The half-open retained prefix is `[start_frame, end_frame)`. The decoded motion
tail is exactly `[end_frame - context_frames, end_frame)`, with a supported
5/22/39-frame context. Both canvases use existing `core.resize_image` then the
connected video VAE. LOW is smaller than HIGH in both dimensions; each canvas
dimension is a positive multiple of 32. Existing native H3 latent geometry is
validated before returning context.

Provide `audio_relative_path` and `audio_sha256` together to bind a separate
immutable original soundtrack, such as Leon's lossless stereo `source_audio.wav`.
Without them the selected video container is the explicit audio authority.
Refined video audio is never substituted for a separately bound original.
Actual stereo PCM is decoded and encoded **once**, then the exact same audio
tensor is shared by LOW and HIGH. `video_only` uses an explicitly unused native
placeholder and does not call the audio VAE.

## Provenance and conditioning

Each context is an `ExternalRGBPCMContext` dict subclass with
`origin='external_rgb_pcm_reencoded'` at the top level and in metadata. Native
conditioning uses local coordinate `segment_index=1` and local source coordinate
0 solely to activate the native motion-reference geometry. These coordinates
do not identify a canonical candidate or accepted parent. Metadata has no
accepted candidate or chain ID, and reports explicitly set
`original_native_latent_available=False` and `has_sampling_ancestor=False`.

The adapter retains its actual public branch segment/candidate index 0 and
existing prompt/media mapping. It validates the typed external context before
and after calling existing `long_video.build_long_video_conditioning` with
local coordinate 1, then supplies the independently prepared LOW/HIGH pair to
the existing `prepared_pair_exp` sampler path. Validate again at execution
and checkpoint boundaries. The caller must independently revalidate its
current selected delivery, branch record and retained prefix; a frozen source
cannot establish that an editor selection is still current.

The HIGH helper seeds only the video prefix and adds the native full-shape
prefix mask. Free video regions, the audio tensor and existing audio mask are
preserved. It never routes through the accepted-parent HIGH helper. When an
explicit caller redraw path has already seeded the exact re-encoded prefix,
`preserve_existing_mask=True` authenticates the prefix and retains that entire
AV object and soft redraw mask. Conflicting prefix ownership or mismatched
geometry fails rather than silently overwriting another mask.

Trim the generated context prefix once for branch delivery. Keep the original
retained RGB/PCM prefix in the caller's editorial assembly; it is not a native
accepted sampling ancestor. Later generated branch candidates may establish
their own real native lineage through the existing acceptance path.

## Clocks and binding

Input video must be exactly one zero-origin uninterrupted 24fps CFR stream with
square pixels, a fixed canvas and no declared rotation. A new material must
explicitly convert unsupported clocks or geometry; this helper never guesses
the source phase. An audio authority must have one actual uninterrupted stereo
stream whose sample clock covers the selected tail. Mono is not duplicated.

At native 32kHz, frame boundaries are often fractional samples. Each endpoint
is `round(Fraction(absolute_frame * sample_rate, 24))`, using ties to even. The
slice length is **rounded end minus rounded start**, not rounded duration. A
22-frame tail at frames 30..52 contains 29,333 samples, whereas 31..53 contains
29,334. The original source phase and actual PTS are retained in the JSON report.
Existing native H3 audio-overhang and conditioning time-grid rules remain in
use after encoding.

Bindings include the complete source video/original audio SHA256, relative
contained paths, prefix/tail intervals, decoded clocks, implementation bytes,
actual encoded tensor content and connected producer identities. Source bytes,
context geometry/tensors and known native VAE content/method selections are
revalidated. Known native VAEs use the existing producer identity adapter.
Unknown wrappers remain process-local/nonportable; opaque callable internals
are not universally certified and are delegated under the existing patch-stack
policy rather than blocked for being unqualified combinations.

## Validation scope

Focused CPU tests use real encoded MP4/WAV input, native conditioning/masks and
the pinned Core tiny H3 Euler path. A separate probe executes reduced random
native video/audio VAE encoders through public Core VAE wrappers and checks
native weight binding. Other learned VAE/lifter fixtures are explicitly test
doubles. No pretrained weights are loaded. The existing native CPU CI gate explicitly collects
`tests/test_progressive_external_rgb_pcm_context.py` alongside its previous
contracts; its triggers, permissions and dependency installation are unchanged.
These checks cannot establish pretrained model quality, full
delivery seam quality, CUDA compatibility, VRAM use or speed. Reports retain
`gpu_acceptance='NOT_RUN'` and `quality_qualified=False`; full generated video
and audio still need the project's existing delivery and human-review gates.
