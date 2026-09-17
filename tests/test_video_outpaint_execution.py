from types import SimpleNamespace

import pytest
import torch
from comfy.model_base import MiniMaxH3, ModelType
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg.video_outpaint_conditioning import prepare_outpaint_conditioning
from h3_audio_t8_pkg.video_outpaint_identity import native_stock_model_identity, value_identity
from h3_audio_t8_pkg.video_outpaint_execution import sample_verified_outpaint
from h3_audio_t8_pkg.video_outpaint_audio_runtime import prepare_outpaint_audio_cache
from h3_audio_t8_pkg.video_outpaint_source_runtime import prepare_outpaint_source_cache
from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from test_video_outpaint_media import _clip
from test_video_outpaint_source_runtime import StatefulVAE
from test_video_outpaint_sampling import _backend


class TinyNative(MiniMaxH3):
    """Native MODEL type with tiny test weights; no learned diffusion is run."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.diffusion_model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.diffusion_model.weight.fill_(1)
            self.diffusion_model.bias.zero_()
        self.model_config = SimpleNamespace(unet_config={"test_width": 2})
        self.model_type = ModelType.FLOW_AV
        self.manual_cast_dtype = None


def _model():
    return ModelPatcher(TinyNative(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


class TinyClip:
    def __init__(self):
        self.calls = []
        self.offset = 0

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text):
        self.calls.append(text)
        return [[torch.full((1, max(1, len(text)), 16), float(len(text)+self.offset)),
                 {"pooled_output": None, "minimax_token_tags": torch.zeros(1, max(1, len(text)), dtype=torch.long),
                  "test_tuple": ("preserved", 1.0)}]]


def test_actual_model_state_config_and_patch_guards():
    model = _model()
    first = native_stock_model_identity(model)
    assert first == native_stock_model_identity(_model())
    with torch.no_grad():
        model.model.diffusion_model.weight[0, 0] += 0.1
    assert native_stock_model_identity(model)["sha256"] != first["sha256"]
    model = _model()
    model.model.model_config.unet_config["test_width"] = 3
    assert native_stock_model_identity(model)["sha256"] != first["sha256"]
    model.patches["weight"] = "unverified LoRA"
    assert native_stock_model_identity(model)['portable_cache_reuse'] is False
    with pytest.raises(ValueError, match="materializing"):
        value_identity(torch.ones(3, 4).T)


def test_model_runtime_forward_replacement_is_not_silently_trusted():
    model = _model()
    model.model.diffusion_model.forward = lambda x: x
    first = native_stock_model_identity(model)
    assert first['portable_cache_reuse'] is False
    assert first['sha256'] != native_stock_model_identity(model)['sha256']


def _plan():
    return build_outpaint_plan(source_sha256="a"*64, width=32, height=32, frame_count=90,
                              aspect="custom", left=32, right=32, cut_frames=(45,))


def test_conditioning_is_shot_local_frozen_and_rechecked_from_bytes(tmp_path):
    clip = TinyClip()
    provider = prepare_outpaint_conditioning(clip, ["first shot", "second shot"], _plan(), tmp_path)
    assert clip.calls == ["first shot", "second shot"]
    first = provider(0, 0)
    first[0][0].zero_()
    assert torch.all(provider(0, 0)[0][0] == 10)
    assert provider(1, 0)[0][1]["test_tuple"] == ("preserved", 1.0)
    assert provider(1, 0)[0][1]["minimax_token_tags"].dtype == torch.long
    assert prepare_outpaint_conditioning(clip, ["first shot", "second shot"], _plan(), tmp_path).verify() == provider.verify()
    clip.offset = 1
    with pytest.raises(ValueError, match="CLIP outputs/prompts differ"):
        prepare_outpaint_conditioning(clip, ["first shot", "second shot"], _plan(), tmp_path)
    assert torch.all(provider(0, 0)[0][0] == 10)
    record = provider.data["shots"][0]
    (tmp_path / f"conditioning-{record['sha256']}.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        provider(0, 0)


def test_cancel_prompt_preparation_does_not_publish_partial_shots(tmp_path):
    def cancel(_):
        raise RuntimeError("cancel prompt encoding")

    with pytest.raises(RuntimeError, match="cancel"):
        prepare_outpaint_conditioning(TinyClip(), "scene", _plan(), tmp_path, progress=cancel)
    assert not (tmp_path / "outpaint_conditioning.json").exists()
    assert prepare_outpaint_conditioning(TinyClip(), "scene", _plan(), tmp_path).verify()


def test_empty_prompt_remains_available_like_native_reference(tmp_path):
    clip = TinyClip()
    provider = prepare_outpaint_conditioning(clip, "", _plan(), tmp_path)
    assert clip.calls == ["", ""]
    assert provider(0, 0)[0][0].shape == (1, 1, 16)


def _chain(tmp_path):
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=90, sound=False)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=32, height=32, frame_count=90,
                              aspect="custom", left=32, right=32, top=32, bottom=32, window_frames=73)
    video, _ = prepare_outpaint_source_cache(StatefulVAE(), inspection, plan, tmp_path / "source-cache")
    audio, _ = prepare_outpaint_audio_cache(None, inspection, plan, tmp_path / "audio-cache")
    text = prepare_outpaint_conditioning(TinyClip(), "test scene", plan, tmp_path / "text-cache")
    return dict(model=_model(), conditioning=text, source_store=video, audio=audio,
                cache_root=tmp_path / "sampling", sample_function=_backend)


def test_verified_execution_actual_provider_chain_can_resume(tmp_path):
    inputs = _chain(tmp_path)

    def cancel(_):
        raise RuntimeError("cancel sampled window")

    with pytest.raises(RuntimeError, match="cancel"):
        sample_verified_outpaint(**inputs, progress=cancel)
    store, report = sample_verified_outpaint(**inputs, resume=True)
    assert report["all_windows_sampled"] and report["windows_sampled_this_call"] == 1
    assert report["resume_from"] == 1 and not report["caller_supplied_identity_trusted"]
    assert not report["generated_video_complete"]
    assert len(store.snapshot()["committed"]) == 2


def test_mid_sample_model_mutation_cannot_commit_a_window(tmp_path):
    inputs = _chain(tmp_path)

    def mutate(*a, **kw):
        with torch.no_grad():
            inputs["model"].model.diffusion_model.weight += 1
        return _backend(*a, **kw)

    inputs["sample_function"] = mutate
    with pytest.raises(ValueError, match="actual MODEL"):
        sample_verified_outpaint(**inputs)
    import json
    state = json.loads((tmp_path / "sampling/outpaint_windows.json").read_text())
    assert state["status"] == "interrupted" and state["committed"] == []
