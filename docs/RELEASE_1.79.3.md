# v1.79.3 — Official Topaz preflight and model-selection fixes

## Changes

- Removed the fixed 12GiB free-VRAM startup requirement from the official Topaz route. The external engine still receives `vram_fraction`, and the controller still validates telemetry and stops its owned process under critical or sustained resource pressure.
- Changed the RGB48 uncompressed disk calculation from a blocking requirement to an advisory audit record. PNG/FFV1 output is compressed, so the raw-frame upper bound is not the actual master size. The owned process still stops before output-disk free space falls below 1GiB.
- Added optional `output_directory` support for storing the lossless task master on another absolute local path. A blank value keeps the existing ComfyUI output location.
- Changed `model_id` from free text to a dropdown of common official enhancement definitions. Advanced `custom_model_id` overrides the dropdown for future official definitions not yet listed.
- Reworded the shared isolated-process exception from “FI task” to “task” so Topaz failures are not mislabeled as frame interpolation failures.

## Compatibility

The node ID, required connection order, existing `iris-3` default, fixed-scale behavior and old workflow values remain valid. New fields are optional and appended to the existing widget sequence. Models and weights are not downloaded automatically; a selected model still must be prepared through the user's official Topaz installation.

## Scope

These changes fix preflight and interface behavior. They do not claim that every clip fits every GPU or disk, do not enable interpolation, and do not qualify 4x or Starlight. Existing Iris short-clip mechanical and human-review evidence remains the applicable quality scope.
