"""CPU-only abrupt-exit fixture; does not load an H3 model or touch user processes."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from comfy.cli_args import args

args.cpu = True

import conftest  # noqa: E402,F401
from test_video_outpaint_sampling_runtime import _execution  # noqa: E402
from h3_audio_t8_pkg.video_outpaint_sampling_runtime import sample_prepared_outpaint_windows  # noqa: E402


def abrupt_exit(_):
    os._exit(43)


if __name__ == "__main__":
    sample_prepared_outpaint_windows(**_execution(Path(sys.argv[1])), progress=abrupt_exit)
    raise RuntimeError("abrupt-exit fixture failed to reach its checkpoint")
