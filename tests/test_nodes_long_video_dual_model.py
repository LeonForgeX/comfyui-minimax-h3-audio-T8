from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8
from h3_audio_t8_pkg.nodes_long_video_in_node_loop_effects_advanced import MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced


def test_new_schema_is_independent_and_does_not_mutate_old_defaults():
    old_before = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
    schema = MiniMaxH3DualModelLongVideoEXPT8.define_schema()
    ids = [item.id for item in schema.inputs]
    assert len(ids) == len(set(ids))
    assert ids[:2] == ["model_pass1", "model_pass2"]
    assert "model" not in ids and "long_video_sampling_plan" not in ids
    fields = {item.id: item for item in schema.inputs}
    assert fields["coarse_steps"].default == fields["refine_steps"].default == 4
    assert fields["low_width"].default == 512 and fields["width"].default == 1024
    assert fields["second_audio_strength"].default == 0.
    assert schema.is_output_node and schema.is_experimental
    old_after = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
    assert [(item.id, getattr(item, "default", None)) for item in old_before.inputs] == [
        (item.id, getattr(item, "default", None)) for item in old_after.inputs]


def test_native_tokenizer_identity_covers_vocabulary_padding_and_clip_options():
    from types import SimpleNamespace
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from h3_audio_t8_pkg.long_video_dual_tokenizer import tokenizer_identity
    clip = SimpleNamespace(tokenizer=MiniMaxH3Tokenizer(), tokenizer_options={}, layer_idx=None,
                           use_clip_schedule=False, apply_hooks_to_conds=None)
    first = tokenizer_identity(clip)
    clip.tokenizer.qwen3vl_32b.max_length -= 1
    assert tokenizer_identity(clip) != first
    clip.tokenizer.qwen3vl_32b.max_length += 1
    assert tokenizer_identity(clip) == first
    clip.tokenizer.qwen3vl_32b.tokenizer.add_tokens(['special_identity_probe'])
    assert tokenizer_identity(clip) != first


def test_plain_and_relay_workflows_keep_independent_lora_edges():
    import json
    from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
    from tools.build_dual_model_workflows import build_prompt, build_workflow
    from tools.audit_progressive_workflows import audit_candidate
    # Minimal loader schemas isolate builder contracts. The separate live Core
    # qualification checks real installed enums/types without test path aliases.
    info = {
        'UNETLoader': {'input': {'required': {'unet_name': ['STRING', {}], 'weight_dtype': ['STRING', {}]}},
                       'output': ['MODEL'], 'output_name': ['MODEL']},
        'VAELoader': {'input': {'required': {'vae_name': ['STRING', {}]}}, 'output': ['VAE'], 'output_name': ['VAE']},
        'CLIPLoader': {'input': {'required': {key: ['STRING', {}] for key in ('clip_name', 'type', 'device')}},
                       'output': ['CLIP'], 'output_name': ['CLIP']}}
    for cls in (MiniMaxH3DualModelLongVideoEXPT8, MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                MiniMaxH3PromptRelayPlanT8Advanced):
        info[cls.define_schema().node_id] = cls.GET_NODE_INFO_V1()
    info = json.loads(json.dumps(info))
    for relay in (False, True):
        prompt = build_prompt(info, relay=relay)
        workflow = build_workflow(info, relay=relay)
        audit = audit_candidate(prompt, workflow, info)
        assert audit['nodes'] == (8 if relay else 7)
        assert prompt['8']['inputs']['model_pass1'] == ['2', 0]
        assert prompt['8']['inputs']['model_pass2'] == ['3', 0]
        assert prompt['2'] is not prompt['3']
        assert prompt['8']['inputs']['eav_mode'] == 'disabled'
        assert prompt['8']['inputs']['total_duration_seconds'] == 8.
        assert prompt['8']['inputs']['video_context_mode'] == 'high_native_mask_exp'
        assert workflow['extra']['dual_model_delivery_status'] == 'unreleased_scoped_8s_human_accepted_not_universal_quality'
        if relay:
            assert prompt['7']['inputs']['length'] == 193
            assert '<d>' not in prompt['7']['inputs']['global_prompt']
            assert prompt['7']['inputs']['local_prompts'].count('<d>') == 1
        else:
            assert '<d>' not in prompt['8']['inputs']['global_prompt']
