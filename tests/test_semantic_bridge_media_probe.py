"""Decoder diagnostics retain the failing process exit code, even without stderr."""
import json
from types import SimpleNamespace

import pytest

from tools import run_semantic_bridge_workflow_probe as probe


@pytest.mark.parametrize("selection", ["0:v:0", "0:a:0"])
@pytest.mark.parametrize("code,stderr", [(3221225477, ""), (1, "invalid packet")])
def test_strict_decode_reports_stream_and_actual_exit(tmp_path, monkeypatch, selection, code, stderr):
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(b"not real video; subprocess double only")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=json.dumps({"streams": [
                dict(codec_type="video", width=64, height=64, nb_read_frames="73", avg_frame_rate="24/1"),
                dict(codec_type="audio", duration="3.042")]}))
        failed = command[command.index("-map") + 1] == selection
        return SimpleNamespace(returncode=code if failed else 0, stderr=stderr if failed else "")
    monkeypatch.setattr(probe.subprocess, "run", run)
    with pytest.raises(RuntimeError) as error:
        probe.audit_video(path, tmp_path, [64, 64, 73, "24/1"])
    assert f"{selection}, exit={code}" in str(error.value)
    assert (stderr or "decoder returned no stderr") in str(error.value)
    assert calls[-1][0] == "ffmpeg" and "-xerror" in calls[-1]
