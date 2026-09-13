# v1.79.4 — Topaz practical delivery, controls and interpolation

## Fixed

- Regular Topaz enhancement now defaults to a single high-quality H.264 NVENC MP4 output (p5/HQ/CQ16) instead of a 16-bit lossless frame master. A 15-second 720×1282 Iris2x run had produced a 12.9GB MOV and then exhausted the F drive when a downstream SaveVideo tried to write it again. The new node output is already the final saved file; do not append SaveVideo.
- `lossless_master` remains an explicit advanced audit/archive option and is clearly labeled as potentially huge.
- Delivery disk telemetry uses a conservative H.264 working estimate and a 256MiB runtime stop floor; the lossless route keeps its separate RGB48 estimate and 1GiB floor.

## Controls

- Added `model_defaults`, `auto_estimate` and `manual` parameter modes.
- Manual controls expose antialias/deblur, noise reduction, detail recovery, dehalo, sharpen, compression recovery, pre-noise, output grain/size, color correction and source blend. Advanced JSON remains only as an override/forward-compatibility escape hatch.
- Added model-use guidance to the upscale selector and workflow note.
- Added an advanced extra-engine-instance control. It defaults to `0`: on the same local 60-frame Iris3 pilot, `1` only reduced elapsed time from about 11.5 to 10.5 seconds while increasing resource use, so 16GB GPUs should keep `0` unless separately qualified.
- Topaz does not need to stay open. It must be installed and licensed, and the selected model must first be downloaded through the official application.

## Frame interpolation

- Added a separate `MiniMaxH3TopazFrameInterpolationEXPT8` node and example workflow. It uses official `tvai_fi` for 2x/4x FPS conversion, preserves duration/dimensions, packet-copies the source audio, and directly saves H.264 MP4.
- Apollo, Apollo Fast, Chronos, Chronos Fast and Aion IDs are offered with use guidance. Weights remain user-supplied through the official Topaz installation; this machine has no FI weights installed, so actual FI visual quality is not claimed by this release.

## Evidence and limits

- Real Iris3 2x / 60-frame test: 1440×2564 H.264 NVENC, approximately 11.5 seconds and 2.9MB. This proves the compact encoder path and official model invocation, not universal speed or visual quality.
- Existing short Topaz quality reviews remain applicable. Starlight remains separate and paused.
