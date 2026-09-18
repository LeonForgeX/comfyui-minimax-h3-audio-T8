"""One owned continuation-window text intervention, never a new full-loop receipt.

The first segment is read-only evidence, not adopted into a new accepted chain.
Only the known post-owner event2 text changes. Reference audio, native AV tails,
event timing, joint route and the Stock20 sampling functions remain unchanged.
"""
import argparse
import asyncio
import copy
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import uuid


PARENT = 'native-voice-dialogue-owner'
OWNER_NOTE = (' [The dialogue already started in an earlier segment; '
              'continue the performance without restarting or repeating that line.]')
BEFORE = 'S1 speaks gently with relief and restrained joy: ' + OWNER_NOTE
AFTER = ('S1 maintains a gentle expression of relief and restrained joy, looks at '
         'the camera, and listens silently with closed lips. No additional speech.')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_parent(first, full, processes):
    """Require the real terminal20+20 evidence, not intent or a lockfile."""
    for receipt in processes:
        p = receipt.get('process', {})
        if (receipt.get('status') != 'owned_worker_exit0_cleanup0'
                or p.get('status') != 'complete' or p.get('exit_code') != 0
                or p.get('active_after_cleanup') != 0
                or p.get('job_assigned_before_task') is not True):
            raise ValueError('Both parent owned Jobs must be terminal and cleaned up')
    if (first.get('status') != 'controlled_native_voice_window_first_committed_cancelled_not_complete8s'
            or first.get('actual_forwards') != 20 or first.get('accepted_count') != 1
            or full.get('status') != 'actual_native_reference_single_model_two_segment20_8s_complete_not_human'
            or full.get('actual_forwards') != 20 or full.get('all_jobs_actual_forwards') != 40
            or full.get('completed_segments') != 2 or full.get('discarded_timeout_partial_forwards') != 0
            or full.get('reused_first_segment') != first.get('first_segment')
            or full.get('reused_first_segment_bytes_unchanged') is not True
            or full.get('assets_and_Core_unchanged') is not True
            or any(t.get('window_text_policy') != 'dialogue_start_owner_exp'
                   or t.get('sources_unchanged') is not True for t in (first, full))
            or first.get('production_source_sha256') != full.get('production_source_sha256')):
        raise ValueError('Only the exact completed owner20+20 parent qualifies as input evidence')
    return copy.deepcopy(first['first_segment'])


def visual_only_crossing_text(projected, validate_plan, hash_plan):
    """Fixed artifact-only intervention: no NLP rewriting of arbitrary user text."""
    source = validate_plan(projected)
    projection = source.get('long_video_projection', {})
    if (source.get('long_video_window_text_policy') != 'dialogue_start_owner_exp'
            or source.get('query_route') != 'joint_av_exp'
            or projection.get('accepted_start_frame') != 124
            or projection.get('accepted_end_frame_exclusive') != 192
            or projection.get('segment_index') != 1 or projection.get('context_frames') != 22
            or projection.get('dialogue_owner_event_indices') != []
            or projection.get('omitted_dialogue_event_indices') != [2]
            or [e['event_index'] for e in source['events']] != [2, 3]
            or source['events'][0]['local_prompt'] != BEFORE
            or 'Clear newly generated Mandarin dialogue' not in source['global_prompt']):
        raise ValueError('Intervene only in the known second owner window and its residual speech instruction')
    result = copy.deepcopy(source)
    result.pop('plan_hash')
    result['events'][0]['local_prompt'] = AFTER
    compiled = 'Global scene: ' + result['global_prompt']
    for event in result['events']:
        compiled += '\n'
        event['prompt_char_start'] = len(compiled)
        compiled += f"Event {event['event_index']}: {event['local_prompt']}"
        event['prompt_char_end'] = len(compiled)
    result['compiled_prompt'] = compiled
    # The changed compiled text already has a separate authenticated identity.
    # Keep every projection field/coordinate unchanged; no forged source receipt.
    result['plan_hash'] = hash_plan(result)
    validate_plan(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    core = project.parents[1]
    evidence = project / 'artifacts/five-track-development-20260918'
    if args.output.exists() or not args.output.resolve().is_relative_to(evidence):
        raise ValueError('Use a fresh task-owned evidence output')
    previous = evidence / (PARENT + '-gpu-v1')
    terminal_paths = [previous / 'terminal.json', evidence / (PARENT + '-gpu-v2/terminal.json')]
    process_paths = [evidence / (PARENT + f'-process-v{i}.json') for i in (1, 2)]
    first, full = [json.loads(p.read_text(encoding='utf8')) for p in terminal_paths]
    entry = validate_parent(first, full, [json.loads(p.read_text(encoding='utf8')) for p in process_paths])
    recipe_path = previous / 'executed.api.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf8'))
    from native_voice_owned_resume import owned_resume_chain_root, _validated_fixed_recipe_and_first
    root = owned_resume_chain_root(previous, recipe['8']['inputs']['chain_id'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if len(manifest.get('segments', [])) != 2 or manifest['segments'][0] != entry:
        raise ValueError('Preserve the real completed parent manifest without changing it')
    _validated_fixed_recipe_and_first(recipe, {**manifest, 'segments': [entry]}, relay_owner='13')
    if recipe.get('13') != dict(class_type='MiniMaxH3PromptRelayWindowTextEXPT8',
                               inputs=dict(prompt_relay_plan=['12', 0], text_policy='dialogue_start_owner_exp')):
        raise ValueError('Preserve the fixed parent text-policy owner')
    inputs = recipe['8']['inputs']
    frozen = {str(p.relative_to(project)): digest(p) for p in (project / 'h3_t8').rglob('*.py')}
    if frozen != first['production_source_sha256']:
        raise ValueError('Production implementation changed since the parent probe')
    bound = {str(p): digest(p) for p in terminal_paths + process_paths + [recipe_path, manifest_path]}
    for name in ('video', 'context'):
        path = (root / entry[name + '_path']).resolve(strict=True)
        if not path.is_relative_to((root / 'accepted').resolve(strict=True)) or digest(path) != entry[name + '_sha256']:
            raise ValueError('Parent accepted input escaped or changed')
        bound[str(path)] = entry[name + '_sha256']
    binding_path = evidence / 'native-voice-ablation-gpu-v1/terminal.json'
    binding = json.loads(binding_path.read_text(encoding='utf8'))
    bound[str(binding_path)] = digest(binding_path)
    assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
    assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
    args.output.mkdir(parents=True)
    result = dict(status='incomplete', published=False, human_qualified=False,
                  full_loop_qualified=False, resume_qualified=False, source_audio_mux=False,
                  user_frontend_changed=False, spatial_sampling_tiles=False,
                  hypothesis='Residual positive speech instruction may invite unrequested/reference-like words after tagged dialogue removal',
                  global_Mandarin_instruction_already_present=True,
                  change='only post-projection event2 text; reference audio and native AV carry-over retained',
                  parent_evidence_SHA256=bound, production_source_sha256=frozen,
                  scope='segment1 only: render90/context22/delivery68; no first-segment sampling, accepted-chain write or full8s claim')
    sys.path.insert(0, str(project / 'h3_t8/prepared_backend'))
    from resource_guard import SerialProbeLease, NvmlResourceReader
    try:
        with SerialProbeLease(project / 'artifacts/acceleration-research-20260909/serial-gpu.lock'), NvmlResourceReader() as reader:
            result['resource_before'] = reader.sample()
            sys.path[:0] = [str(core), str(project)]
            sys.argv = ['owned-voice-continuation-text', '--disable-pinned-memory', '--reserve-vram', '6', '--preview-method', 'none']
            import comfy.options
            comfy.options.enable_args_parsing()
            import torch
            import nodes
            import execution
            import folder_paths
            import comfy.model_patcher
            import comfy.model_management as mm
            from comfy.ldm.minimax.model import MiniMaxH3Model
            torch.set_num_threads(2)
            assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == reader.uuid.removeprefix('GPU-').lower()
            for name, setter in [('output', folder_paths.set_output_directory), ('temp', folder_paths.set_temp_directory), ('user', folder_paths.set_user_directory)]:
                path = args.output / name
                path.mkdir()
                setter(str(path))
            spec = importlib.util.spec_from_file_location('owned_continuation', project / '__init__.py', submodule_search_locations=[str(project)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)
            class Server:
                client_id = 'owned-continuation'
                last_node_id = None
                def send_sync(self, event, data, sid=None):
                    if str(event) in {'execution_error', 'execution_interrupted', 'execution_success'}:
                        with (args.output / 'events.jsonl').open('a', encoding='utf8') as stream:
                            stream.write(json.dumps(dict(event=str(event), data=data), default=str) + '\n')

            async def run():
                for name in ('nodes_custom_sampler.py', 'nodes_audio.py', 'nodes_video.py'):
                    assert await nodes.load_custom_node(str(core / 'comfy_extras' / name), module_parent='comfy_extras')
                for cls in await package.comfy_entrypoint().get_node_list():
                    nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
                valid = await execution.validate_prompt(str(uuid.uuid4()), recipe, None)
                assert valid[0] and not valid[3], str(valid)
                executor = execution.PromptExecutor(Server(), cache_args={'ram': 16., 'ram_inactive': 16., 'lru': 0})
                await executor.execute_async(recipe, str(uuid.uuid4()), {'client_id': 'owned-continuation'}, ['2', '3', '4', '5', '6', '11', '13'])
                assert executor.success, str(executor.status_messages)
                async def value(key):
                    return (await executor.caches.outputs.get(key)).outputs[0][0]
                clip, video_vae, audio_vae, picture, recording, model, plan = [await value(k) for k in ('2', '3', '4', '5', '6', '11', '13')]
                for vae in (video_vae, audio_vae):
                    vae.first_stage_model.to(dtype=vae.vae_dtype)
                    vae.patcher = comfy.model_patcher.ModelPatcher(vae.first_stage_model, load_device=vae.device, offload_device=mm.vae_offload_device())
                    vae.disable_offload = True
                result['VAE_policy'] = 'own_native_static_patcher_original_compute_dtype_original_chunks_no_Core_edits'
                from owned_continuation.long_video_delivery import _load_accepted_context_file, _write_mp4_atomic, _tensor_sha256
                from owned_continuation.prompt_relay_advanced import _validate_plan, _sha256_json
                from owned_continuation.prompt_relay_long_video_advanced import project_prompt_relay_plan_to_long_video_window
                from owned_continuation.long_video_in_node_loop_effects_advanced import _bind_loop_relay_conditioning, _sample_prepared_segment
                from owned_continuation.long_video import patch_long_video_model
                from owned_continuation.sampling import setup_dual_clock_sampling
                from owned_continuation.audio_ops import decode_av_latent, trim_av_output
                # PromptExecutor runs node bodies under inference_mode. The
                # explicit segment call must use that same context, otherwise
                # cached inference-only CLIP parameters enter an autograd path.
                bind = torch.inference_mode()(_bind_loop_relay_conditioning)
                sample = torch.inference_mode()(_sample_prepared_segment)
                decode = torch.inference_mode()(decode_av_latent)
                schedule = torch.inference_mode()(setup_dual_clock_sampling)
                context, context_report = _load_accepted_context_file(root / entry['context_path'], inputs['chain_id'], 0, 1)
                tails = {k: _tensor_sha256(context[k]) for k in ('video_tail', 'audio_tail')}
                before, _, projection_report = project_prompt_relay_plan_to_long_video_window(plan, 1, 90, 22, 124/24, 192/24)
                after = visual_only_crossing_text(before, _validate_plan, _sha256_json)
                for name, p in [('before', before), ('after', after)]:
                    (args.output / (name + '.projected-plan.json')).write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding='utf8')
                result.update(context=context_report, unchanged_AV_tail_tensor_SHA256=tails,
                              original_projection=json.loads(projection_report), before_prompt=before['compiled_prompt'],
                              after_prompt=after['compiled_prompt'], before_plan_hash=before['plan_hash'], after_plan_hash=after['plan_hash'])
                started = time.monotonic()
                result['phase'] = 'conditioning'
                segment_model, positive, latent, mux, prompt, media_map, relay_report = bind(
                    model=model, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
                    context=context, prompt_relay_plan=after, segment_index=1, context_frames=22,
                    context_audio=inputs['context_audio'], width=512, height=768, length=90,
                    task_type=inputs['task_type'], audio_mode=inputs['audio_mode'],
                    audio_denoise_strength=inputs['audio_denoise_strength'], add_source_as_reference=False,
                    prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size=inputs['ref_image_size'],
                    reference_video_policy=inputs['reference_video_policy'], execution_mode='apply_exp',
                    query_chunk_rows=inputs['query_chunk_rows'], ref_images={'ref_image_0': picture},
                    ref_audios={'ref_audio_0': recording})
                assert mux is None and prompt == after['compiled_prompt']
                result.update(media_map=json.loads(media_map), relay_report=json.loads(relay_report))
                assert result['media_map']['audios'] == {'1': 'ref_audio_1'}
                segment_model = patch_long_video_model(segment_model)
                sampled_model, sampler, sigmas = schedule(segment_model, latent, 20, 12., 3., 'dual_clock_euler', 'native_flow')
                result['sigmas'] = sigmas.tolist()
                calls, original = [], MiniMaxH3Model.forward
                @wraps(original)
                def counted(self, *items, **kwargs):
                    output = original(self, *items, **kwargs)
                    calls.append(len(calls) + 1)
                    (args.output / 'forward-progress.json').write_text(json.dumps({'actual_forwards': len(calls)}), encoding='utf8')
                    return output
                MiniMaxH3Model.forward = counted
                try:
                    result['phase'] = 'sampling'
                    sampled = sample(sampled_model, positive, latent, sampler=sampler, sigmas=sigmas, seed=20260919, segment_index=1)
                finally:
                    MiniMaxH3Model.forward = original
                assert len(calls) == 20
                result['phase'] = 'native_AV_decode'
                frames, generated_audio, _, _ = decode(sampled, video_vae, audio_vae)
                # Original loop context-removal boundaries, not discretionary speech trimming.
                frames, generated_audio, trim = trim_av_output(frames, 22/24, 68/24, generated_audio, 24.)
                assert frames.shape[0] == 68 and all(_tensor_sha256(context[k]) == sha for k, sha in tails.items())
                path = args.output / 'continuation68.mp4'
                result['phase'] = 'native_encode'
                written = _write_mp4_atomic(path, frames, generated_audio, fps=24,
                    target_audio_samples=round(192 * generated_audio['sample_rate']/24) - round(124 * generated_audio['sample_rate']/24), bit_depth=8, crf=18)
                result.update(actual_forwards=len(calls), seconds=time.monotonic()-started, seed=20260919,
                              native_context_trim=json.loads(trim), encode_report=written,
                              AV_context_tensors_unchanged=True, media=dict(path=str(path), sha256=digest(path)))
            asyncio.run(run())
            result['resource_after'] = reader.sample()
        assert all(digest(path) == sha for path, sha in bound.items())
        assert all(digest(project / name) == sha for name, sha in frozen.items())
        assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
        assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
        result.update(status='actual_owned_continuation20_text_intervention_complete_not_full8s_not_human',
                      parent_evidence_and_implementation_unchanged=True, Core_and_assets_unchanged=True, phase='complete')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'terminal.json').write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k in ('status', 'actual_forwards', 'seconds', 'error', 'media', 'full_loop_qualified', 'published')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
