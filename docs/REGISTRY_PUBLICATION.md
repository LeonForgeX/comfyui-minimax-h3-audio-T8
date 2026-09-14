# Comfy Registry publication gate

The official publish action proving that an archive upload completed is not evidence that users can
install that version. Registry security scanning may leave an uploaded version `Flagged` or `Banned`,
while the public node endpoint continues to expose an older Active version as `latest_version`.

The release workflow therefore performs a second, authoritative check after upload:

- the exact version from `pyproject.toml` must exist in the node's Registry version list;
- its status must be `NodeVersionStatusActive`;
- the public node endpoint must expose that exact version as `latest_version`.

`Flagged` and `Banned` fail the workflow immediately. Temporary propagation delay is retried for a
bounded period. The gate never republishes, changes Registry state or bypasses review.

## v1.80.0 incident

Version 1.80.0 uploaded successfully on 2026-09-14 but was marked `Flagged`; the public latest version
therefore remained 1.47.0. The local report is
[T8mars/comfyui-minimax-h3-audio-T8#19](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/19)
and the manual Registry review request is
[Comfy-Org/registry-backend#233](https://github.com/Comfy-Org/registry-backend/issues/233).

Future Registry archives built from this revision exclude the bundled VRetouchEr research adapter and
pinned upstream source. Those modules have no registered node or released workflow, and the upstream
source contains an unused SpyNet fallback downloader. They remain in the GitHub source tree for
reproducibility but are not part of the future installable Registry runtime package. The already-uploaded
v1.80.0 archive is unchanged and still requires manual Registry review. Registered Topaz, DLSS, FFmpeg,
long-video and compatibility features remain packaged; their fixed-argument worker processes and audited
compatibility compilation must be reviewed rather than removed or disguised merely to evade scanner
patterns.
