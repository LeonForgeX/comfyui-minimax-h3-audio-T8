from __future__ import annotations

import json

import pytest
import torch

import h3_audio_t8_pkg.prompt_relay_advanced as prompt_relay_module
from comfy.model_patcher import ModelPatcher
from h3_audio_t8_pkg.long_video import LONG_VIDEO_CONDITIONING_KEY, LONG_VIDEO_SCHEMA
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY,
    PROMPT_RELAY_PAYLOAD_KEY,
    build_prompt_relay_plan,
)
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
    build_prompt_relay_long_video_conditioning,
    project_prompt_relay_plan_to_long_video_window,
    configure_long_video_window_text,
    WINDOW_TEXT_POLICY_KEY,
)
from helpers import FakeAudioVAE, FakeVideoVAE


class _ByteHF:
    def __init__(self):
        self.byte_decoder = {chr(0x100 + value): value for value in range(256)}

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return chr(0x100 + int(token_id))


class _ByteInner:
    def __init__(self):
        self.tokenizer = _ByteHF()

    @staticmethod
    def tokenize_with_weights(text, **_kwargs):
        return [[(int(value), 1.0) for value in text.encode("utf-8")]]


class _OuterTokenizer:
    def __init__(self):
        self.qwen3vl_32b = _ByteInner()


class NativeLikeFakeClip:
    def __init__(self):
        self.tokenizer = _OuterTokenizer()

    @staticmethod
    def tokenize(prompt, **kwargs):
        prefix_count = 0
        if kwargs.get("images"):
            prefix_count += 2 * len(kwargs["images"])
        if kwargs.get("minimax_ref_items"):
            prefix_count += 2 * len(kwargs["minimax_ref_items"])
        return {
            "qwen3vl_32b": [[
                *[(1000 + index, 1.0) for index in range(prefix_count)],
                *[(int(value), 1.0) for value in prompt.encode("utf-8")],
            ]]
        }

    @staticmethod
    def encode_from_tokens_scheduled(tokens):
        entries = tokens["qwen3vl_32b"][0]
        tags = torch.tensor(
            [0 if int(entry[0]) >= 1000 else 1 for entry in entries],
            dtype=torch.long,
        )
        return [[torch.zeros((1, len(entries), 8)), {"minimax_token_tags": tags}]]


class MiniMaxH3Model(torch.nn.Module):
    pass


class _NativeH3Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniMaxH3Model()

    def extra_conds(self, **_kwargs):
        return {}


_NativeH3Base.extra_conds.__module__ = "comfy.model_base"


def _model_patcher():
    return ModelPatcher(
        _NativeH3Base(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def _allow_fixture_core_contract(monkeypatch):
    monkeypatch.setattr(prompt_relay_module, "_source_sha256", lambda _source: "fixture")
    monkeypatch.setattr(prompt_relay_module, "ATTENTION_FORWARD_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "PACKED_LAYOUT_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "TOKENIZER_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "EXTRA_CONDS_SHA256S", {"fixture"})


def _global_plan(length=345):
    return build_prompt_relay_plan(
        global_prompt="同一个人物、场景与连续声音",
        local_prompts="人物抬手并响起钟声\n人物奔跑，只有脚步声\n镜头拉远并响起雷声",
        length=length,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )[0]


def test_projection_uses_accepted_start_minus_context_and_preserves_global_sigma():
    source = _global_plan()
    projected, prompt, report = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=1,
        length=124,
        context_frames=22,
        timeline_start_seconds=124 / 24,
        timeline_end_seconds=226 / 24,
    )
    parsed = json.loads(report)
    assert prompt == source["compiled_prompt"]
    assert projected["frame_count"] == 124
    assert projected["long_video_projection"]["render_start_frame"] == 102
    assert parsed["render_window_frames"] == [102, 226]
    assert parsed["accepted_window_frames"] == [124, 226]
    assert parsed["render_active_event_indices"] == [1, 2]
    assert parsed["accepted_active_event_indices"] == [2]

    coordinate_shift = (5 / 3) * 102
    for original, local in zip(source["events"], projected["events"], strict=True):
        assert local["midpoint"] + coordinate_shift == pytest.approx(original["midpoint"])
        assert local["window"] == original["window"]
        assert local["sigma"] == original["sigma"]
    # Event 1 crosses into the context head. It must not be clamped to local frame 0
    # and re-estimated as a short event.
    assert projected["events"][0]["start_frame"] == -102
    assert projected["events"][0]["end_frame_exclusive"] == 13


def test_projection_fails_closed_on_wrong_grid_or_global_duration():
    source = _global_plan(124)
    with pytest.raises(ValueError, match="context_frames=0"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 124, 22, 0, 124 / 24)
    with pytest.raises(ValueError, match=r"17n\+5"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 123, 0, 0, 123 / 24)
    with pytest.raises(ValueError, match="exceeds the global"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 141, 0, 0, 130 / 24)


def test_combined_segment_zero_installs_both_scoped_contracts(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _global_plan(124)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=0,
        length=124,
        context_frames=0,
        timeline_start_seconds=0,
        timeline_end_seconds=124 / 24,
    )
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": True,
        "chain_id": "relay-long-video",
        "target_segment_index": 0,
    }
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context=context,
        prompt_relay_plan=projected,
        segment_index=0,
        context_frames=0,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    patched, conditioning, latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    parsed = json.loads(report)
    assert metadata[LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert PROMPT_RELAY_BINDING_KEY in metadata
    assert metadata["model_conds"][PROMPT_RELAY_PAYLOAD_KEY].cond == (
        metadata[PROMPT_RELAY_BINDING_KEY]["binding_hash"]
    )
    assert getattr(
        patched.get_model_object("extra_conds"),
        "_t8_long_video_patch_version",
        None,
    ) == 1
    assert patched.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)[
        "projected_plan_hash"
    ] == projected["plan_hash"]
    assert parsed["status"] == "applied_exp"
    assert parsed["audio_mode"] == "native"
    assert json.loads(parsed["stable_conditioning_report"])["audio_mode"] == "native"
    assert parsed["render_window_frames"] == [0, 124]
    assert parsed["dense_s_by_s_mask_created"] is False
    assert latent["samples"].is_nested


@pytest.mark.parametrize(
    ("local_prompts", "expected_status"),
    [
        ("", "passthrough_no_events_long_video_only"),
        ("One global-length local event.", "passthrough_single_event_long_video_only"),
    ],
)
def test_zero_or_one_event_bypasses_only_relay_but_keeps_long_video_patch(
    monkeypatch,
    local_prompts,
    expected_status,
):
    _allow_fixture_core_contract(monkeypatch)
    source = build_prompt_relay_plan(
        global_prompt="One stable global scene.",
        local_prompts=local_prompts,
        length=124,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )[0]
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=0,
        length=124,
        context_frames=0,
        timeline_start_seconds=0,
        timeline_end_seconds=124 / 24,
    )
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context={
            "schema": LONG_VIDEO_SCHEMA,
            "empty": True,
            "chain_id": "relay-long-video-bypass",
            "target_segment_index": 0,
        },
        prompt_relay_plan=projected,
        segment_index=0,
        context_frames=0,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    patched, conditioning, _latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    parsed = json.loads(report)
    assert metadata[LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert PROMPT_RELAY_BINDING_KEY not in metadata
    assert PROMPT_RELAY_PAYLOAD_KEY not in metadata.get("model_conds", {})
    assert getattr(
        patched.get_model_object("extra_conds"),
        "_t8_long_video_patch_version",
        None,
    ) == 1
    assert patched.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY) is None
    assert parsed["status"] == expected_status
    assert parsed["attention_patch_installed"] is False
    assert parsed["event_count"] == (1 if local_prompts else 0)


def test_combined_continuation_binds_repaired_motion_context_layout(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _global_plan(345)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=1,
        length=124,
        context_frames=22,
        timeline_start_seconds=124 / 24,
        timeline_end_seconds=226 / 24,
    )
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": torch.zeros((1, 24, 12, 8, 8)),
        "audio_tail": torch.zeros((1, 32, 2, 65)),
        "metadata": {
            "source_segment_index": 0,
            "target_segment_index": 1,
            "max_context_frames": 39,
            "audio_overhang": 1 / 3,
        },
    }
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context=context,
        prompt_relay_plan=projected,
        segment_index=1,
        context_frames=22,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="auto",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    _patched, conditioning, _latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    binding = metadata[PROMPT_RELAY_BINDING_KEY]
    parsed = json.loads(report)
    assert parsed["resolved_task"] == "i2va-motion"
    assert parsed["context_frames"] == 22
    assert parsed["render_window_frames"] == [102, 226]
    assert binding["keyframe_count"] == 7
    assert binding["reference_block_count"] == 1
    assert binding["layout_contract"]["segments"][-2][2] == "audio"
    assert binding["layout_contract"]["segments"][-1][2] == "video"


def test_combined_continuation_recovers_known_live_long_video_patch(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source_model = _model_patcher()
    base_model = source_model.model

    from h3_audio_t8_pkg.long_video import patch_long_video_model

    segment_zero_model = patch_long_video_model(source_model)
    live_patch = segment_zero_model.get_model_object("extra_conds")
    base_model.extra_conds = live_patch
    try:
        source = _global_plan(345)
        projected, *_ = project_prompt_relay_plan_to_long_video_window(
            source,
            segment_index=1,
            length=124,
            context_frames=22,
            timeline_start_seconds=124 / 24,
            timeline_end_seconds=226 / 24,
        )
        context = {
            "schema": LONG_VIDEO_SCHEMA,
            "empty": False,
            "video_tail": torch.zeros((1, 24, 12, 8, 8)),
            "audio_tail": torch.zeros((1, 32, 2, 65)),
            "metadata": {
                "source_segment_index": 0,
                "target_segment_index": 1,
                "max_context_frames": 39,
                "audio_overhang": 1 / 3,
            },
        }
        result = build_prompt_relay_long_video_conditioning(
            model=source_model,
            clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            context=context,
            prompt_relay_plan=projected,
            segment_index=1,
            context_frames=22,
            context_audio="video_and_audio",
            width=128,
            height=128,
            length=124,
            task_type="auto",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            execution_mode="apply_exp",
            query_chunk_rows=64,
        )
        patched = result[0]
        assert "extra_conds" in patched.object_patches
        normalized_patch = patched.get_model_object("extra_conds")
        assert getattr(
            normalized_patch,
            "_t8_long_video_patch_version",
            None,
        ) == 1
    finally:
        del base_model.extra_conds


def test_conditioning_rejects_a_projected_plan_from_another_segment():
    source = _global_plan(345)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source, 1, 124, 22, 124 / 24, 226 / 24
    )
    with pytest.raises(ValueError, match="does not match Conditioning inputs"):
        build_prompt_relay_long_video_conditioning(
            model=object(),
            clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            context={},
            prompt_relay_plan=projected,
            segment_index=2,
            context_frames=22,
            context_audio="video_only",
            width=128,
            height=128,
            length=124,
            task_type="auto",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            execution_mode="report_only",
            query_chunk_rows=64,
        )


def test_long_video_prompt_relay_nodes_are_append_only_advanced_nodes():
    schemas = [node.define_schema() for node in PROMPT_RELAY_LONG_VIDEO_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3PromptRelayLongVideoPlanT8Advanced",
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced",
    ]
    assert all(schema.is_experimental is True for schema in schemas)
    assert all(schema.category == "T8/MiniMax H3/Long Video/Experimental" for schema in schemas)
    conditioning_inputs = {item.id: item for item in schemas[1].inputs}
    assert conditioning_inputs["execution_mode"].default == "report_only"
    assert conditioning_inputs["audio_mode"].default == "native"


def _window_text_plan():
    return build_prompt_relay_plan(
        global_prompt="Same scene and identity. Quiet room tone.",
        local_prompts="FIRST台词\nSECOND台词\nListen silently.",
        length=192, timing_mode="frames", time_ranges="0-123\n124-183\n184-191",
        math_profile="paper_v1", epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )[0]


def test_window_text_default_is_exact_plan_identity_and_can_be_disabled():
    source = _window_text_plan()
    default, _ = configure_long_video_window_text(source)
    assert default is source
    enabled, _ = configure_long_video_window_text(source, "accepted_window_text_exp")
    assert enabled is not source and WINDOW_TEXT_POLICY_KEY not in source
    restored, _ = configure_long_video_window_text(enabled, "preserve_all")
    assert restored == source
    assert enabled[WINDOW_TEXT_POLICY_KEY] == "accepted_window_text_exp"


@pytest.mark.parametrize(("index", "length", "context", "start", "end", "kept"), [
    (0, 124, 0, 0, 124, [1]),
    (1, 90, 22, 124, 192, [2, 3]),
])
def test_explicit_window_text_omits_complete_and_future_keys_with_exact_spans(
    index, length, context, start, end, kept,
):
    source = _window_text_plan()
    source_hash = source["plan_hash"]
    enabled, _ = configure_long_video_window_text(source, "accepted_window_text_exp")
    old, *_ = project_prompt_relay_plan_to_long_video_window(source, index, length, context, start/24, end/24)
    actual, prompt, report = project_prompt_relay_plan_to_long_video_window(enabled, index, length, context, start/24, end/24)
    assert [e["event_index"] for e in old["events"]] == [1, 2, 3]
    assert [e["event_index"] for e in actual["events"]] == kept
    assert source["plan_hash"] == source_hash and source["compiled_prompt"] == old["compiled_prompt"]
    for e in actual["events"]:
        before = next(k for k in old["events"] if k["event_index"] == e["event_index"])
        assert {k:v for k,v in e.items() if not k.startswith("prompt_char_")} == {
            k:v for k,v in before.items() if not k.startswith("prompt_char_")}
        assert prompt[e["prompt_char_start"]:e["prompt_char_end"]] == f"Event {e['event_index']}: {e['local_prompt']}"
    parsed = json.loads(report)
    assert parsed["text_event_indices"] == kept and parsed["event_count_kept"] == len(kept)
    assert ("FIRST台词" in prompt) == (1 in kept)
    assert ("SECOND台词" in prompt) == (2 in kept)


def test_window_text_retains_crossing_event_original_width_sigma_and_absolute_time():
    source = _global_plan(345)
    enabled, _ = configure_long_video_window_text(source, "accepted_window_text_exp")
    old, *_ = project_prompt_relay_plan_to_long_video_window(source, 1, 124, 22, 124/24, 226/24)
    actual, *_ = project_prompt_relay_plan_to_long_video_window(enabled, 1, 124, 22, 124/24, 226/24)
    assert [e["event_index"] for e in actual["events"]] == [2]
    before = old["events"][1]
    after = actual["events"][0]
    for key in ("start_frame", "end_frame_exclusive", "midpoint", "window", "sigma", "start_coord", "end_coord"):
        assert after[key] == before[key]


@pytest.mark.parametrize("bad", ["invented", "changed_hash", "projected"])
def test_window_text_policy_fails_closed_before_conditioning(bad):
    source = _window_text_plan()
    if bad == "changed_hash":
        source["global_prompt"] += "changed"
        match = "hash mismatch"
    elif bad == "projected":
        source, *_ = project_prompt_relay_plan_to_long_video_window(source, 0, 124, 0, 0, 124/24)
        match = "before projecting"
    else:
        match = "Unknown"
    with pytest.raises(ValueError, match=match):
        configure_long_video_window_text(source, "invented" if bad == "invented" else "accepted_window_text_exp")


@pytest.mark.parametrize('policy', ['accepted_window_text_exp', 'dialogue_start_owner_exp'])
def test_window_text_single_event_native_conditioning_keeps_long_patch_and_context(monkeypatch, policy):
    _allow_fixture_core_contract(monkeypatch)
    source = _global_plan(345)
    if policy == 'dialogue_start_owner_exp':
        source = build_prompt_relay_plan(
            global_prompt='Same person and scene.',
            local_prompts='S1 looks up.\nS1 smiles. <d>[Mandarin]只说一次</d>\nS1 listens.',
            length=345, timing_mode='auto_equal', time_ranges='', math_profile='paper_v1',
            epsilon=.1, allow_gaps=False, allow_overlaps=False)[0]
    enabled, _ = configure_long_video_window_text(source, policy)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(enabled, 1, 124, 22, 124/24, 226/24)
    context = {"schema": LONG_VIDEO_SCHEMA, "empty": False,
               "video_tail": torch.zeros((1,24,12,8,8)), "audio_tail": torch.zeros((1,32,2,65)),
               "metadata": {"source_segment_index":0, "target_segment_index":1,
                            "max_context_frames":39, "audio_overhang":1/3}}
    original_audio = context["audio_tail"].clone()
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        context=context, prompt_relay_plan=projected, segment_index=1, context_frames=22,
        context_audio="video_and_audio", width=128, height=128, length=124, task_type="auto",
        audio_mode="native", audio_denoise_strength=.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", execution_mode="apply_exp", query_chunk_rows=64)
    patched, conditioning, latent, audio, prompt, media, report = result
    parsed = json.loads(report)
    assert parsed["status"] == "passthrough_single_event_long_video_only"
    assert parsed["text_event_indices"] == [2]
    assert parsed["context_frames"] == 22 and parsed["audio_mode"] == "native"
    assert conditioning[0][1][LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert getattr(patched.get_model_object("extra_conds"), "_t8_long_video_patch_version", None) == 1
    assert torch.equal(context["audio_tail"], original_audio)
    assert prompt == projected["compiled_prompt"] and latent["samples"].is_nested
    if policy == 'dialogue_start_owner_exp':
        assert 'S1 smiles.' in prompt and '只说一次' not in prompt
        assert parsed['omitted_dialogue_event_indices'] == [2]


def test_dialogue_owner_zero_active_events_keeps_global_and_no_dialogue():
    source = build_prompt_relay_plan(
        global_prompt='Fixed room.', local_prompts='<d>[Mandarin]未来台词</d>\nVisual.',
        length=345, timing_mode='frames', time_ranges='226-305\n306-344',
        math_profile='paper_v1', epsilon=.1, allow_gaps=True, allow_overlaps=False)[0]
    owned, _ = configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    projected, prompt, report = project_prompt_relay_plan_to_long_video_window(owned, 0, 124, 0, 0, 124/24)
    assert projected['events'] == [] and prompt == 'Global scene: Fixed room.'
    assert json.loads(report)['dialogue_owner_event_indices'] == []


def test_window_text_independent_node_default_preserves_old_schemas():
    from h3_audio_t8_pkg.nodes_prompt_relay_window_text_exp import MiniMaxH3PromptRelayWindowTextEXPT8
    schema = MiniMaxH3PromptRelayWindowTextEXPT8.define_schema()
    assert schema.node_id == "MiniMaxH3PromptRelayWindowTextEXPT8" and schema.is_experimental
    assert next(x for x in schema.inputs if x.id == "text_policy").default == "preserve_all"
    source = _window_text_plan()
    assert MiniMaxH3PromptRelayWindowTextEXPT8.execute(prompt_relay_plan=source).result[0] is source


def _dialogue_owner_plan(*, global_prompt="Stable theatre and identity.",
                         local=None, ranges="0-72\n73-145\n146-191"):
    return build_prompt_relay_plan(
        global_prompt=global_prompt,
        local_prompts=local or ("S1 looks up. <d>[Mandarin]你终于回来了。</d>\n"
                               "S1 smiles gently. <d>[Mandarin]这次别再走了。</d>\n"
                               "S1 listens; no new speech."),
        length=192, timing_mode="frames", time_ranges=ranges,
        math_profile="paper_v1", epsilon=.1, allow_gaps=False, allow_overlaps=False,
    )[0]


def test_dialogue_start_owner_keeps_crossing_visual_event_but_not_restarted_line():
    import copy
    source = _dialogue_owner_plan()
    frozen = copy.deepcopy(source)
    window, _ = configure_long_video_window_text(source, 'accepted_window_text_exp')
    owned, _ = configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    assert owned['plan_hash'] != window['plan_hash']
    assert configure_long_video_window_text(owned, 'preserve_all')[0] == source
    first, first_prompt, first_report = project_prompt_relay_plan_to_long_video_window(
        owned, 0, 124, 0, 0, 124/24)
    second, prompt, report = project_prompt_relay_plan_to_long_video_window(
        owned, 1, 90, 22, 124/24, 192/24)
    old, *_ = project_prompt_relay_plan_to_long_video_window(window, 1, 90, 22, 124/24, 192/24)
    assert '你终于回来了' in first_prompt and '这次别再走了' in first_prompt
    assert '你终于回来了' not in prompt and '这次别再走了' not in prompt
    assert 'S1 smiles gently.' in prompt and 'S1 listens;' in prompt
    assert [e['event_index'] for e in second['events']] == [2, 3]
    for actual, expected in zip(second['events'], old['events']):
        excluded = {'local_prompt', 'prompt_char_start', 'prompt_char_end'}
        assert {k:v for k,v in actual.items() if k not in excluded} == {
            k:v for k,v in expected.items() if k not in excluded}
        assert prompt[actual['prompt_char_start']:actual['prompt_char_end']] == (
            f"Event {actual['event_index']}: {actual['local_prompt']}")
    assert json.loads(first_report)['dialogue_owner_event_indices'] == [1, 2]
    assert json.loads(report)['dialogue_owner_event_indices'] == []
    assert json.loads(report)['omitted_dialogue_event_indices'] == [2]
    assert first['global_plan_hash'] == owned['plan_hash'] and source == frozen


def test_dialogue_event_start_exactly_at_seam_belongs_only_to_next_window():
    source = _dialogue_owner_plan(ranges='0-123\n124-183\n184-191')
    owned, _ = configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    _, before, _ = project_prompt_relay_plan_to_long_video_window(owned, 0, 124, 0, 0, 124/24)
    _, after, report = project_prompt_relay_plan_to_long_video_window(owned, 1, 90, 22, 124/24, 192/24)
    assert '这次别再走了' not in before and '这次别再走了' in after
    assert json.loads(report)['dialogue_owner_event_indices'] == [2]
    assert json.loads(report)['omitted_dialogue_event_indices'] == []


def test_multiple_dialogue_blocks_unicode_and_visual_text_survive_only_outside_tags():
    source = _dialogue_owner_plan(local='Listen.\n顔🙂 <d>[Japanese]甲</d> 眨眼 <d>[Mandarin]乙</d> 点头\n静静微笑')
    owned, _ = configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    _, prompt, report = project_prompt_relay_plan_to_long_video_window(owned, 1, 90, 22, 124/24, 192/24)
    assert '顔🙂 ' in prompt and ' 眨眼 ' in prompt and ' 点头' in prompt
    assert '甲' not in prompt and '乙' not in prompt and '<d>' not in prompt
    assert json.loads(report)['omitted_dialogue_event_indices'] == [2]


@pytest.mark.parametrize('bad', ['<d>missing', '</d>stray', '<d>a<d>b</d></d>',
                               '<D>uppercase</D>', '<d x="1">attributes</d>', '<d'])
def test_dialogue_owner_rejects_malformed_tags_only_in_explicit_policy(bad):
    source = _dialogue_owner_plan(local=bad + '\nVisual.\nListen.')
    assert configure_long_video_window_text(source)[0] is source
    assert configure_long_video_window_text(source, 'accepted_window_text_exp')[0]
    with pytest.raises(ValueError, match='Dialogue owner'):
        configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    # A valid signed marker cannot bypass the check at projection time.
    source[WINDOW_TEXT_POLICY_KEY] = 'dialogue_start_owner_exp'
    source.pop('plan_hash')
    source['plan_hash'] = prompt_relay_module._sha256_json(source)
    with pytest.raises(ValueError, match='Dialogue owner'):
        project_prompt_relay_plan_to_long_video_window(source, 0, 124, 0, 0, 124/24)


def test_dialogue_owner_rejects_global_tagged_line_but_preserves_old_policies():
    source = _dialogue_owner_plan(global_prompt='S1 says <d>[Mandarin]一次</d>')
    assert configure_long_video_window_text(source)[0] is source
    with pytest.raises(ValueError, match='Global'):
        configure_long_video_window_text(source, 'dialogue_start_owner_exp')


def test_dialogue_owner_many_windows_and_untagged_speech_not_claimed_controlled():
    source = build_prompt_relay_plan(
        global_prompt='Fixed room.', local_prompts='Visual <d>[Mandarin]唯一</d> continues.\nuntagged speech',
        length=345, timing_mode='frames', time_ranges='0-305\n306-344',
        math_profile='paper_v1', epsilon=.1, allow_gaps=False, allow_overlaps=False)[0]
    owned, _ = configure_long_video_window_text(source, 'dialogue_start_owner_exp')
    for index, start, end, length, context in [(0, 0, 124, 124, 0), (1, 124, 226, 124, 22),
                                             (2, 226, 328, 124, 22), (3, 328, 345, 39, 22)]:
        _, prompt, _ = project_prompt_relay_plan_to_long_video_window(owned, index, length, context, start/24, end/24)
        assert ('唯一' in prompt) == (index == 0)
        assert ('untagged speech' in prompt) == (index >= 2)
