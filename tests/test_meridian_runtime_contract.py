"""Production boundaries with real CPU upstream math; no learned geometry/GPU claim."""

from pathlib import Path
import os

import pytest
import torch

from h3_audio_t8_pkg import meridian_runtime as r
from h3_audio_t8_pkg.meridian_plan import preset_plan
from h3_audio_t8_pkg.meridian_sources import source_module, omega_imports
from h3_audio_t8_pkg.meridian_generate import fixed_assets


def test_material_xor_before_any_external_import():
    with pytest.raises(ValueError, match="exactly one"):
        r.prepare({})
    with pytest.raises(ValueError, match="exactly one"):
        r.prepare({}, torch.zeros(1, 1, 1, 3), object())


def test_gauge_empty_and_negative_depth_never_promoted():
    geometry = dict(
        depth=torch.zeros(1, 512, 512),
        keep=torch.zeros(1, 512, 512, dtype=torch.bool),
        intr=torch.eye(3)[None],
    )
    with pytest.raises(ValueError, match="usable depth"):
        r.gauge(geometry, [0, 0, 1280, 1280, 1.0])
    geometry["depth"].fill_(-1)
    geometry["keep"].fill_(True)
    with pytest.raises(ValueError):
        r.gauge(geometry, [0, 0, 1280, 1280, 1.0])


def test_gauge_depth_units_and_target_pixel_are_finite():
    geometry = dict(
        depth=torch.full((1, 512, 512), 2.0),
        keep=torch.ones(1, 512, 512, dtype=torch.bool),
        intr=torch.tensor([[[450.0, 0, 256], [0, 450.0, 256], [0, 0, 1]]]),
    )
    zm, pivot = r.gauge(geometry, [0, 0, 1280, 1280, 1.0])
    assert zm == 2 and pivot == [0.0, 0.0, 1.0]


def test_omega_import_missing_source_does_not_change_path(tmp_path):
    import sys

    before = sys.path.copy()
    with pytest.raises(ValueError):
        with omega_imports(tmp_path):
            pass
    assert sys.path == before


def test_unknown_source_module_rejected_without_import(tmp_path):
    with pytest.raises(ValueError):
        source_module(tmp_path, "h3")


def test_assets_length_validation_not_upper_frame_limit(tmp_path):
    runtime = dict(source=str(tmp_path))
    with pytest.raises(ValueError, match="paired assets"):
        fixed_assets(runtime, 872)
    with pytest.raises(ValueError, match="17k"):
        fixed_assets(runtime, 869)


def test_independent_space_keys_delegate_actual_upstream_curve():
    # This installed pinned math is tested separately from any model/component loader.
    import folder_paths

    upstream = Path(
        os.environ.get(
            "MERIDIAN_SOURCE_DIR",
            str(Path(folder_paths.models_dir) / "meridian/source"),
        )
    )
    path = source_module(upstream, "path")
    material = dict(
        identity="known", start=0, end=100, kind="video", pivot=[0.0, 0.0, 1.0]
    )
    p = preset_plan(material, 73)
    a = path.plan_path([dict(k, src=0) for k in p["camera_keys"]], 73, 2.0)[0]
    p["time_keys"].insert(1, dict(t=20, src=20))
    b = path.plan_path([dict(k, src=0) for k in p["camera_keys"]], 73, 2.0)[0]
    assert torch.equal(torch.from_numpy(a), torch.from_numpy(b))


@pytest.mark.parametrize(
    "exception",
    [RuntimeError("early resource failure"), InterruptedError("ordinary cancel")],
)
def test_generate_entry_receipt_keeps_early_errors(tmp_path, monkeypatch, exception):
    import json
    from h3_audio_t8_pkg import meridian_generate as g

    def fail(*args):
        raise exception

    monkeypatch.setattr(g, "_generate", fail)
    with pytest.raises(type(exception)):
        g.generate({}, output_directory=str(tmp_path))
    receipt = json.loads(next(tmp_path.glob("Meridian_*.json")).read_text())
    expected = "interrupted" if isinstance(exception, InterruptedError) else "failed"
    assert (
        receipt["status"] == expected
        and receipt["current_stage"] == "resource_verification"
    )
    assert type(exception).__name__ in receipt["error"] and receipt["wall_seconds"] >= 0
    assert not list(tmp_path.glob("*.mp4"))


def test_generate_wrapper_preserves_returned_delivery_and_terminal(
    tmp_path, monkeypatch
):
    import json
    from h3_audio_t8_pkg import meridian_generate as g
    from h3_audio_t8_pkg.meridian_checkpoint_io import atomic_json

    video = object()

    def success(prepared, seed, audio_mode, delivery):
        record = dict(
            status="complete_full_AV_decoded_not_human", marker="unchanged_inner_report"
        )
        atomic_json(delivery["report_path"], record, replace_existing=True)
        return video, str(delivery["path"]), json.dumps(record)

    monkeypatch.setattr(g, "_generate", success)
    returned, path, raw = g.generate({}, output_directory=str(tmp_path))
    assert returned is video and path.startswith(str(tmp_path))
    record = json.loads(raw)
    assert record["status"] == "complete_full_AV_decoded_not_human"
    assert (
        record["marker"] == "unchanged_inner_report"
        and record["current_stage"] == "complete"
    )


def test_external_vae_delegated_without_loading_or_mutation():
    from h3_audio_t8_pkg.meridian_generate import own_vae

    foreign = object()
    assert own_vae(dict(video_vae=foreign)) is foreign


def test_owned_vae_cancel_during_load_releases_only_owned_patcher(monkeypatch):
    from types import SimpleNamespace
    import comfy.sd
    import comfy.utils
    import comfy.model_patcher
    import comfy.model_management as mm
    from h3_audio_t8_pkg import meridian_generate as g

    released = []
    vae = SimpleNamespace(
        latent_channels=24,
        device="cpu",
        first_stage_model=SimpleNamespace(to=lambda **kwargs: None),
    )
    patcher = SimpleNamespace(
        offload_device="cpu",
        unpatch_model=lambda device: released.append(("owned", device)),
    )
    monkeypatch.setattr(g, "check", lambda: None)
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda path: {})
    monkeypatch.setattr(comfy.sd, "VAE", lambda **kwargs: vae)
    monkeypatch.setattr(
        comfy.model_patcher, "ModelPatcher", lambda *args, **kwargs: patcher
    )

    def cancel(models, **kwargs):
        assert models == [patcher]
        raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "load_models_gpu", cancel)
    with pytest.raises(mm.InterruptProcessingException):
        g.own_vae(dict(video_vae=None, vae=dict(path="owned-file")))
    assert released == [("owned", "cpu")]


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("legacy_sys_api", [False, True])
def test_owned_omega_original_failure_not_masked_by_cleanup(
    tmp_path, monkeypatch, cancel, cleanup_failure, legacy_sys_api
):
    """CPU boundary double only; actual StageStore and ordinary Core exception."""
    from contextlib import nullcontext
    import json
    import sys
    from types import SimpleNamespace
    import comfy.model_patcher
    import comfy.model_management as mm

    original = mm.InterruptProcessingException() if cancel else RuntimeError("Omega forward failed")
    released = []

    class Model:
        def eval(self):
            return self

        def load_state_dict(self, weights, strict):
            assert weights == {} and strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def reconstruct(*args):
        raise original

    def release(device):
        released.append(str(device))
        if cleanup_failure:
            raise RuntimeError("Omega offload failed")

    patcher = SimpleNamespace(unpatch_model=release)
    monkeypatch.setitem(sys.modules, "vggt_omega.models", SimpleNamespace(VGGTOmega=Model))
    monkeypatch.setattr(r, "check", lambda: None)
    monkeypatch.setattr(r, "omega_imports", lambda *args: nullcontext())
    monkeypatch.setattr(r, "source_module", lambda *args: SimpleNamespace(reconstruct=reconstruct, to_input=lambda x: x))
    monkeypatch.setattr(r, "file_identity", lambda path: dict(path=path, bytes=0, sha256="0" * 64))
    monkeypatch.setattr(r, "roi", lambda full, geo: (full, [0, 0, 8, 8, 1.], [8, 8], [8, 8], [0, 0, 8, 8]))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(comfy.model_patcher, "ModelPatcher", lambda *args, **kwargs: patcher)
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(mm, "load_models_gpu", lambda models, **kwargs: None)
    runtime = dict(source="fixture", omega_repo="fixture", omega=dict(path="owned-file"),
        omega_sources={}, source_files=dict(geometry="0" * 64), implementation={},
        budget=0, cache_root=str(tmp_path))
    with monkeypatch.context() as compatibility:
        # Simulates the missing3.11 API on this installed Python, not a3.10 runtime.
        if legacy_sys_api:
            compatibility.delattr(sys, "exception", raising=False)
        with pytest.raises(type(original)) as caught:
            r.prepare(runtime, image=torch.ones(1, 8, 8, 3))
    assert caught.value is original and released == ["cpu"]
    notes = getattr(original, "__notes__", [])
    assert any("Owned Omega cleanup also failed" in note for note in notes) == cleanup_failure
    record = json.loads(next(tmp_path.glob("geometry-*.uncached-*.json")).read_bytes())
    assert record["status"] == ("interrupted" if cancel else "failed")
    assert type(original).__name__ in record["error"]
