"""Find an already installed FFmpeg without changing PATH or downloading binaries."""

import os
from pathlib import Path
import shutil
import sys

import folder_paths


def _portable_candidates():
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    core = Path(folder_paths.base_path)
    runtime = Path(sys.executable).parent
    roots = dict.fromkeys((core, core.parent, runtime, runtime.parent))
    for root in roots:
        for relative in (Path(executable), Path("ffmpeg") / executable, Path("ffmpeg/bin") / executable):
            yield root / relative


def resolve_ffmpeg():
    explicit = os.environ.get("T8_FFMPEG_PATH")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise RuntimeError("T8_FFMPEG_PATH 必须指向可执行的 FFmpeg 文件，请修正后重试")
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in _portable_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    except (ImportError, RuntimeError, OSError):
        pass
    raise RuntimeError(
        "未找到 FFmpeg：已检查 PATH、ComfyUI／整合包的 ffmpeg/bin 和 imageio-ffmpeg。"
        "请安装 FFmpeg，或将 T8_FFMPEG_PATH 设置为可执行文件完整路径后重启 ComfyUI。"
    )
