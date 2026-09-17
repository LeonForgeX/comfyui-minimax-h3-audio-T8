"""Packaged stream worker entry/cleanup using CPU-only model/runtime substitutes."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tests.test_tao_worker_finalization import worker_case  # noqa: F401


@pytest.mark.parametrize("fault", [None, "generation", "postflight", "dispose"])
def test_stream_worker_entry_metadata_and_owned_disposal(
    worker_case,  # noqa: F811
    monkeypatch, fault
):
    original, state, root = worker_case
    path = Path(__file__).parents[1] / "h3_t8/prepared_backend/taomate_stream_worker.py"
    spec = importlib.util.spec_from_file_location(
        "public_stream_worker_entry_test", path
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "sha", original.sha)
    state["fault"] = fault
    devices = []
    model_type = sys.modules["taomate_h3.model.dit"].MiniMaxH3DiT

    def empty(model, **kwargs):
        device = kwargs["device"]
        assert device in ("cpu", "meta")
        devices.append(device)
        if device == "meta" and fault == "dispose":
            raise RuntimeError("synthetic model disposal failure")

    monkeypatch.setattr(model_type, "to_empty", empty)
    prepared = tuple(dict(prompt=f"fixture prompt {i}") for i in range(2))
    inputs = ModuleType("taomate_stream_inputs")
    inputs.load_stream_inputs = lambda *args: (object(), prepared)
    monkeypatch.setitem(sys.modules, "taomate_stream_inputs", inputs)
    runtime = SimpleNamespace(
        _cache=object(), request_owner=SimpleNamespace(state="ready")
    )
    monkeypatch.setattr(
        sys.modules["taomate_local_runtime"],
        "make_local_runtime",
        lambda *a, **k: runtime,
    )
    execution = ModuleType("taomate_stream_execution")

    def execute(*args, **kwargs):
        runtime._cache = None
        runtime.request_owner.state = "closed"
        if fault == "generation":
            raise InterruptedError("synthetic stream generation failure")
        return (torch.zeros(1, 24, 72, 30, 54), torch.zeros(2, 32, 405)), dict(
            timing=dict(
                requests=2,
                native_frames=243,
                published_frames=240,
                fps=24,
                video_latents=72,
                audio_latents=405,
                native_samples=324000,
                published_samples=320000,
                sample_rate=32000,
            ),
            requests=[],
        )

    execution.execute_stream = execute
    monkeypatch.setitem(sys.modules, "taomate_stream_execution", execution)
    request_path = root / "request.json"
    request = json.loads(request_path.read_text())
    request["schema"] = "t8-taomate-prepared-stream-v1"
    request_path.write_text(json.dumps(request))
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            worker.main()
        report = json.loads((root / "report.json").read_text())
        assert report["status"] == "failed"
    else:
        worker.main()
        report = json.loads((root / "report.json").read_text())
        assert report["status"] == "native_Tao_stream_latents_pass"
        assert report["postflight_pass"] and report["owned_cache_released"]
        with safe_open(
            root / "tao-stream-normalized-latents.safetensors", framework="pt"
        ) as source:
            assert source.metadata()["schema"] == "t8-taomate-stream-latents-v1"
            assert json.loads(source.metadata()["timing"])["requests"] == 2
        assert state["alive_at_postflight"] == {"model": False, "pipeline": False}
    assert devices == ["cpu", "meta"]
    assert runtime._cache is None and runtime.request_owner.state == "closed"
    assert torch.cuda.is_initialized() == state["cuda_initialized_before"]
