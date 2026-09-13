# v1.79.6 — Topaz non-Starlight completion

## Completed

- All 14 regular enhancement models in the node dropdown and all 5 frame-interpolation models completed real runs through the official production worker with runtime downloading disabled. Output publication, strict decode, frame/timeline checks and applicable packet/PCM audio checks passed.
- Both Topaz nodes expose an advanced GPU index. Existing workflows keep GPU 0 because the new widget is appended, preserving prior positional values.
- A shipped workflow now connects the regular enhanced file directly into the independent interpolation node. The two owned tasks remain serial and each publishes its own audited output.
- `delivery_hevc_main10` is an explicit SDR-only profile for 10-bit sources. It uses P010 and HEVC Main10 instead of silently reducing the source to 8-bit. The default `delivery_h264` remains 8-bit only.
- The audit-only `lossless_master` path now validates its real 16-bit RGB output rather than incorrectly applying the delivery 8-bit guard.

## Local model preparation

The qualification machine has the official weights needed by every regular and interpolation dropdown entry for the tested 1024×512 route. They were prepared through the licensed official engine and remain under the user's Topaz model-data directory. The node continues to set `download=0`; no proprietary program, weight or license file is bundled or uploaded.

## Boundaries

- Main10 means 10-bit SDR preservation, not HDR support. PQ/HLG HDR remains rejected.
- VFR, interlaced material, non-square pixels and rotation metadata still require explicit preprocessing. The node does not silently alter timing or geometry.
- Mechanical completion does not claim that every model is visually superior on every source. Select models by material and review the result.
- Starlight remains paused and excluded from this release after the official Neuroserver license checkout failure. No alternate distribution or license bypass is used.
