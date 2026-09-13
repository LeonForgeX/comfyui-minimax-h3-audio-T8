# v1.79.5 — Topaz delivery hardening

## Fixed

- Manual controls are now filtered against the selected model definition. Configurable models receive their declared controls; fixed-preset Artemis/Gaia/NXL models no longer fail because unsupported sliders were sent. The report records ignored controls.
- Weight preflight now follows each official backend's declared network filename templates instead of assuming every file contains the selected output scale. This fixes false "not downloaded" errors for Rhea 1x/2x, Nyx XL, and shared Theia network files.
- The output container is selected before GPU inference. AAC/MP3/AC3/EAC3/ALAC or silent sources use MP4; other unprimed packet-copy audio uses MKV. Non-AAC priming/padding that has no qualified lossless mapping is rejected before model work with an AAC-conversion instruction.
- Only qualified 8-bit SDR pixel formats enter the H.264 route. HDR and 10/12/16-bit sources are refused instead of being silently converted to 8-bit.
- ComfyUI now receives monotonic progress for model frames and the subsequent media-audit stages, so low GPU usage during verification is no longer indistinguishable from a hang.

## Preserved behavior

- The default remains direct CQ16 H.264 NVENC delivery with packet-copied audio and no downstream SaveVideo.
- No automatic model download or license modification was added. Topaz does not need to stay open after the selected model is installed and licensed.
- On the qualification machine, all 14 regular dropdown models at their officially declared scales and all 5 interpolation dropdown models were prepared through the licensed official engine for the 1024x512 route. These local commercial weights are not redistributed and do not constitute visual-quality approval for every model.
- Source files are never overwritten; failed candidates remain task-local and are not published as outputs.

## Remaining limits

- The official Apollo `apo-8` weight was downloaded through the licensed Topaz engine and completed a real one-second 24-to-48fps run through the production worker with `download=0`. Duration, AAC packet bytes, decoded PCM and strict decode passed; visual quality is still human-pending, and other FI models are not qualified.
- Starlight remains paused after the official Neuroserver license checkout failure.
- 10-bit/HDR output, VFR, interlaced media, rotation metadata, multi-GPU selection and a combined upscale-plus-interpolation node are outside this release.
