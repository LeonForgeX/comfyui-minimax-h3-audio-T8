"""Owned8s/two-segment native reference probe, not human voice certification."""
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
import uuid


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def with_query_route(graph, route):
    """Explicit experimental graph connection; never change old defaults or sampling."""
    if route not in ('video_only_paper', 'joint_av_exp'):
        raise ValueError('Unknown query route')
    result = copy.deepcopy(graph)
    if route == 'video_only_paper':
        return result
    if '12' in result or result['8']['inputs'].get('prompt_relay_plan') != ['7', 0]:
        raise ValueError('Expected the fixed qualification recipe; preserve foreign connections')
    result['12'] = dict(class_type='MiniMaxH3PromptRelayQueryRouteT8Advanced',
                        inputs=dict(prompt_relay_plan=['7', 0], query_route=route))
    result['8']['inputs']['prompt_relay_plan'] = ['12', 0]
    return result


def prepared_probe_graph(graph, *, single_model=False):
    """Add computation chunks to a known graph without replacing foreign owners."""
    result = copy.deepcopy(graph)
    target = result['8']['inputs']
    expected = ('MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced' if single_model
                else 'MiniMaxH3DualModelLongVideoEXPT8')
    if result['8']['class_type'] != expected:
        raise ValueError('Recipe must match the explicitly selected probe mode')
    chunk_nodes = {
        '10': {'class_type': 'MiniMaxH3LowVRAMAttentionT8Advanced',
               'inputs': {'model': ['1', 0], 'head_chunks': 4}},
        '11': {'class_type': 'MiniMaxH3ChunkFeedForwardT8Advanced',
               'inputs': {'model': ['10', 0], 'chunks': 2, 'seq_threshold': 4096}},
    }
    if any(key in result and result[key] != value for key, value in chunk_nodes.items()):
        raise ValueError('Preserve foreign chunk-node ownership')
    pins = ('model',) if single_model else ('model_pass1', 'model_pass2')
    if any(target.get(pin) not in (['1', 0], ['11', 0]) for pin in pins):
        raise ValueError('Preserve foreign MODEL connections')
    result.update(chunk_nodes)
    target.update({pin: ['11', 0] for pin in pins})
    return result


def with_window_text_graph(graph, text_policy='accepted_window_text_exp'):
    """One explicit text-policy connection, leaving all timeline/sampling values intact."""
    if text_policy not in ('accepted_window_text_exp', 'dialogue_start_owner_exp'):
        raise ValueError('Explicit known experimental text policy required')
    result = copy.deepcopy(graph)
    node = dict(class_type='MiniMaxH3PromptRelayWindowTextEXPT8',
                inputs=dict(prompt_relay_plan=['12', 0], text_policy=text_policy))
    if (result.get('12') != dict(class_type='MiniMaxH3PromptRelayQueryRouteT8Advanced',
                                inputs=dict(prompt_relay_plan=['7', 0], query_route='joint_av_exp'))
            or result['8']['inputs'].get('prompt_relay_plan') not in (['12', 0], ['13', 0])
            or ('13' in result and result['13'] != node)):
        raise ValueError('Preserve foreign text-policy connections')
    result['13'] = node
    result['8']['inputs']['prompt_relay_plan'] = ['13', 0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-route', choices=['video_only_paper', 'joint_av_exp'], default='video_only_paper')
    parser.add_argument('--single-model', action='store_true',
                        help='Qualify the existing native20 single-model loop; no learned upscale or HIGH pass')
    resume_selection = parser.add_mutually_exclusive_group()
    resume_selection.add_argument('--resume-single-from', type=Path,
                        help='Only the cleaned-up timed-out native-voice-long-gpu-v1 owned output; preserve its completed first segment')
    resume_selection.add_argument('--resume-window-from', type=Path,
                        help='Only the cleaned-up controlled native-voice-window-gpu-v1 first-segment probe')
    parser.add_argument('--window-text', action='store_true')
    parser.add_argument('--text-policy', choices=['accepted_window_text_exp', 'dialogue_start_owner_exp'],
                        default='accepted_window_text_exp')
    parser.add_argument('--stop-after-first', action='store_true',
                        help='Own-process cancellation after committing segment0, before segment1; no discarded partial work')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    core = project.parents[1]
    if ((args.window_text or args.stop_after_first or args.resume_window_from)
            and (not args.window_text or not args.single_model or args.query_route != 'joint_av_exp')
            or args.stop_after_first and (args.resume_single_from or args.resume_window_from)):
        raise ValueError('Window probe requires explicit single-model joint route; controlled stop only on fresh runs')
    if args.window_text and args.resume_single_from:
        raise ValueError('Do not attach window-text claims to the old all-key timed-out probe')
    if args.text_policy != 'accepted_window_text_exp' and not args.window_text:
        raise ValueError('A dialogue owner policy requires an explicit window-text probe')
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Fresh owned output required')
    args.output.mkdir(parents=True)
    result = dict(status='incomplete', human_qualified=False, voice_identity_qualified=False,
        published=False, user_service_changed=False, spatial_sampling_tiles=False)
    frozen = {str(p.relative_to(project)): digest(p) for p in (project / 'h3_t8').rglob('*.py')}
    resume_graph, accepted_files, binding = None, {}, None
    media_root = args.output
    window_resuming = args.resume_window_from is not None
    if window_resuming:
        previous = args.resume_window_from.resolve(strict=True)
        probe_name = ('native-voice-dialogue-owner' if args.text_policy == 'dialogue_start_owner_exp'
                      else 'native-voice-window')
        if (previous != project / ('artifacts/five-track-development-20260918/' + probe_name + '-gpu-v1')
                or args.recipe.resolve(strict=True) != previous / 'executed.api.json'):
            raise ValueError('Only resume the exact controlled task-owned window probe')
        from native_voice_owned_resume import owned_resume_chain_root, prepare_owned_window_resume
        raw = json.loads(args.recipe.read_text(encoding='utf8'))
        chain_root = owned_resume_chain_root(previous, raw['8']['inputs']['chain_id'])
        manifest = json.loads((chain_root / 'manifest.json').read_text(encoding='utf8'))
        prior_terminal = json.loads((previous / 'terminal.json').read_text(encoding='utf8'))
        process = json.loads((previous.parent / (probe_name + '-process-v1.json')).read_text(encoding='utf8'))
        resume_graph, first = prepare_owned_window_resume(raw, manifest, process, prior_terminal,
                                                         text_policy=args.text_policy)
        if frozen != prior_terminal['production_source_sha256']:
            raise ValueError('Production implementation changed since the controlled first segment')
        binding = json.loads((previous.parent / 'native-voice-ablation-gpu-v1/terminal.json').read_text(encoding='utf8'))
        assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
        assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
        for key in ('video', 'context'):
            path = (chain_root / first[key + '_path']).resolve(strict=True)
            if not path.is_relative_to((chain_root / 'accepted').resolve(strict=True)) or digest(path) != first[key + '_sha256']:
                raise ValueError('Accepted window-probe files escaped or changed')
            accepted_files[str(path)] = first[key + '_sha256']
        media_root = previous / 'output'
        result.update(resumed_accepted_segments_before=1, reused_first_segment=first,
                      prior_job_actual_forwards=20, production_source_sha256=frozen,
                      window_text_policy=args.text_policy,
                      prior_job_controlled_stop=True)
    if args.resume_single_from:
        previous = args.resume_single_from.resolve(strict=True)
        expected_previous = project / 'artifacts/five-track-development-20260918/native-voice-long-gpu-v1'
        if (previous != expected_previous or not args.single_model or args.query_route != 'joint_av_exp'
                or args.recipe.resolve(strict=True) != previous / 'executed.api.json'):
            raise ValueError('Only resume this exact single-model owned test, without recipe changes')
        from native_voice_owned_resume import owned_resume_chain_root, prepare_owned_single_resume
        raw = json.loads(args.recipe.read_text(encoding='utf8'))
        chain_root = owned_resume_chain_root(previous, raw['8']['inputs']['chain_id'])
        manifest = json.loads((chain_root / 'manifest.json').read_text(encoding='utf8'))
        process = json.loads((previous.parent / 'native-voice-long-process-v1.json').read_text(encoding='utf8'))
        resume_graph, first = prepare_owned_single_resume(raw, manifest, process)
        prior = json.loads((previous.parent / 'five-track-cpu-final-v11/sources-before.json').read_text(encoding='utf8'))
        if any(prior.get(name.replace('\\', '/')) != value for name, value in frozen.items()):
            raise ValueError('Production implementation changed since the owned first segment')
        binding = json.loads((previous.parent / 'native-voice-ablation-gpu-v1/terminal.json').read_text(encoding='utf8'))
        assert binding['assets_and_Core_unchanged']
        assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
        assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
        for key in ('video', 'context'):
            path = (chain_root / first[key + '_path']).resolve(strict=True)
            if not path.is_relative_to((chain_root / 'accepted').resolve(strict=True)) or digest(path) != first[key + '_sha256']:
                raise ValueError('Accepted first-segment files escaped or changed')
            accepted_files[str(path)] = first[key + '_sha256']
        media_root = previous / 'output'
        result.update(resumed_accepted_segments_before=1, reused_first_segment=first,
                      prior_job_actual_forwards=json.loads((previous / 'forward-progress.json').read_text(encoding='utf8'))['actual_forwards'],
                      assets_and_Core_binding='preexisting_native-voice-ablation-gpu-v1_checked_again_not_first_job_specific_Core_snapshot')
    if args.window_text and binding is None:
        binding = json.loads((project / 'artifacts/five-track-development-20260918/native-voice-ablation-gpu-v1/terminal.json').read_text(encoding='utf8'))
        assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
        assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
        result['assets_and_Core_binding'] = 'existing_ablation_binding_checked_before_and_after_window_probe_not_original_job_specific_snapshot'
    sys.path.insert(0, str(project / 'h3_t8/prepared_backend'))
    from resource_guard import SerialProbeLease, NvmlResourceReader
    try:
        with SerialProbeLease(project / 'artifacts/acceleration-research-20260909/serial-gpu.lock'), NvmlResourceReader() as reader:
            sys.path[:0] = [str(core), str(project)]
            sys.argv = ['owned-voice-dual-short', '--disable-pinned-memory', '--reserve-vram', '6', '--preview-method', 'none']
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
            for kind, setter in [('output', folder_paths.set_output_directory), ('temp', folder_paths.set_temp_directory), ('user', folder_paths.set_user_directory)]:
                path = media_root if kind == 'output' and resume_graph is not None else args.output / kind
                path.mkdir(exist_ok=resume_graph is not None and kind == 'output')
                setter(str(path))
            result['resource_before'] = reader.sample()
            spec = importlib.util.spec_from_file_location('owned_voice_dual', project / '__init__.py', submodule_search_locations=[str(project)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)
            class Server:
                client_id = 'owned-voice-dual'
                last_node_id = None
                def send_sync(self, event, data, sid=None):
                    if str(event) in {'execution_error', 'execution_interrupted', 'executed', 'execution_success'}:
                        with (args.output / 'events.jsonl').open('a', encoding='utf8') as stream:
                            stream.write(json.dumps(dict(event=str(event), data=data), default=str) + '\n')

            async def run():
                for name in ('nodes_custom_sampler.py', 'nodes_audio.py', 'nodes_video.py'):
                    assert await nodes.load_custom_node(str(core / 'comfy_extras' / name), module_parent='comfy_extras')
                for cls in await package.comfy_entrypoint().get_node_list():
                    nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
                raw = json.loads(args.recipe.read_text(encoding='utf8'))
                if resume_graph is not None:
                    graph = resume_graph
                else:
                    route_graph = (raw if args.window_text and '12' in raw else with_query_route(raw, args.query_route))
                    if args.window_text:
                        route_graph = with_window_text_graph(route_graph, text_policy=args.text_policy)
                    graph = prepared_probe_graph(route_graph, single_model=args.single_model)
                if args.window_text:
                    result.update(window_text_policy=args.text_policy, production_source_sha256=frozen)
                result['query_route'] = args.query_route
                result['single_model'] = args.single_model
                target = graph['8']['inputs']
                if args.single_model:
                    assert target['steps'] == 20 and target['shift_video'] == 12. and target['shift_audio'] == 3.
                    assert target['sampler_name'] == 'dual_clock_euler' and target['scheduler'] == 'native_flow'
                    assert not any(name in target for name in ('model_pass1', 'model_pass2', 'upscaler_model'))
                else:
                    assert target['coarse_steps'] == 20 and target['refine_steps'] == 4
                assert target['audio_mode'] == 'native' and target['task_type'] == 'Ref2VA'
                assert not any(name in target for name in ('drive_audio', 'final_audio'))
                suffix = 'long' if args.single_model else 'dual'
                if resume_graph is None:
                    target.update(chain_id='owned_voice_' + suffix + '_' + uuid.uuid4().hex[:16],
                                  filename_prefix='Voice_ref_native20_8s' if args.single_model else 'Voice_ref_native20plus4_8s')
                assert target['total_duration_seconds'] == 8. and target['context_frames'] == 22
                assert [target['width'], target['height']] == [512, 768]
                validation = await execution.validate_prompt(str(uuid.uuid4()), graph, None)
                assert validation[0] and not validation[3] and set(validation[2]) == {'8', '9'}, str(validation)
                (args.output / 'executed.api.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf8')
                executor = execution.PromptExecutor(Server(), cache_args={'ram': 16., 'ram_inactive': 16., 'lru': 0})
                # Keep original compute dtype/math, using a static native patcher
                # for these own VAEs to avoid the observed dynamic pre_norm residency bug.
                await executor.execute_async(graph, str(uuid.uuid4()), {'client_id': 'owned-voice-dual'}, ['3', '4'])
                assert executor.success, str(executor.status_messages)
                for key in ('3', '4'):
                    vae = (await executor.caches.outputs.get(key)).outputs[0][0]
                    vae.first_stage_model.to(dtype=vae.vae_dtype)
                    vae.patcher = comfy.model_patcher.ModelPatcher(vae.first_stage_model,
                        load_device=vae.device, offload_device=mm.vae_offload_device())
                    vae.disable_offload = True
                result['VAE_policy'] = 'own_native_static_patcher_original_compute_dtype_original_chunks_no_Core_edits'
                from owned_voice_dual.preview_execution_context import CONTEXT
                calls = []
                original = MiniMaxH3Model.forward
                @wraps(original)
                def counted(self, *items, **kwargs):
                    value = original(self, *items, **kwargs)
                    calls.append(dict(CONTEXT.get() or {}))
                    (args.output / 'forward-progress.json').write_text(json.dumps({'actual_forwards': len(calls), 'last': calls[-1]}), encoding='utf8')
                    return value
                MiniMaxH3Model.forward = counted
                planned_stop = False
                if args.stop_after_first:
                    loop_module = sys.modules['owned_voice_dual.long_video_in_node_loop_effects_advanced']
                    original_accept = loop_module.accept_long_video_candidate
                    @wraps(original_accept)
                    def accept_then_stop(*items, **kwargs):
                        nonlocal planned_stop
                        value = original_accept(*items, **kwargs)
                        if value[1]:
                            first_manifest = json.loads(Path(value[2]).read_text(encoding='utf8'))
                            assert first_manifest['chain_id'] == target['chain_id'] and len(first_manifest['segments']) == 1
                            planned_stop = True
                            # This is an isolated Job with no queue/UI/server; never a user Core.
                            mm.interrupt_current_processing(True)
                        return value
                    loop_module.accept_long_video_candidate = accept_then_stop
                start = time.monotonic()
                try:
                    await executor.execute_async(graph, str(uuid.uuid4()), {'client_id': 'owned-voice-dual'}, ['8', '9'])
                    if args.stop_after_first:
                        root = media_root / 'output/minimax_h3_t8_long_video' / target['chain_id']
                        state = json.loads((root / 'in_node_loop_effects_state.json').read_text(encoding='utf8'))
                        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
                        assert planned_stop and not executor.success and state['status'] == 'interrupted'
                        assert len(calls) == 20 and len(manifest['segments']) == 1
                        result.update(actual_forwards=20, accepted_count=1, controlled_cancel=True,
                                      first_segment=manifest['segments'][0], manifest_path=str(root / 'manifest.json'),
                                      seconds=time.monotonic()-start)
                        return graph
                    assert executor.success, str(executor.status_messages)
                    result.update(actual_forwards=len(calls), forward_contexts=calls, seconds=time.monotonic()-start)
                    assert len(calls) == (20 if resume_graph is not None else 40 if args.single_model else 48), str(result['actual_forwards'])
                    cached = await executor.caches.outputs.get('8')
                    path = Path(cached.outputs[1][0]).resolve(strict=True)
                    assert path.is_relative_to(media_root.resolve())
                    result.update(completed_segments=cached.outputs[3][0], loop_status=cached.outputs[4][0],
                        report=json.loads(cached.outputs[5][0]), media=dict(path=str(path), sha256=digest(path)))
                    assert result['completed_segments'] == 2
                    import av
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        assert [stream.width, stream.height] == [512, 768] and stream.codec_context.name == 'h264'
                        assert sum(1 for _ in container.decode(video=0)) == 192
                    with av.open(str(path)) as container:
                        assert sum(1 for _ in container.decode(audio=0)) > 0
                finally:
                    MiniMaxH3Model.forward = original
                    if args.stop_after_first:
                        loop_module.accept_long_video_candidate = original_accept
                        mm.interrupt_current_processing(False)
                return graph
            asyncio.run(run())
            result['resource_after'] = reader.sample()
        assert all(digest(project / name) == value for name, value in frozen.items())
        if args.window_text:
            assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
            assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
            result['assets_and_Core_unchanged'] = True
        if resume_graph is not None:
            assert all(digest(path) == sha for path, sha in accepted_files.items())
            assert all(digest(core / name) == sha for name, sha in binding['Core_sources'].items())
            assert all(digest(row['path']) == row['sha256'] for row in binding['bound_assets'].values())
            result.update(reused_first_segment_bytes_unchanged=True, assets_and_Core_unchanged=True,
                          all_jobs_actual_forwards=result['prior_job_actual_forwards'] + result['actual_forwards'],
                          completed_selected_segment_configured_forwards=40,
                          discarded_timeout_partial_forwards=result['prior_job_actual_forwards'] - 20)
        result.update(status=('controlled_native_voice_window_first_committed_cancelled_not_complete8s'
                              if args.stop_after_first else 'actual_native_reference_single_model_two_segment20_8s_complete_not_human'
                              if args.single_model else 'actual_native_reference_two_segment20plus4_8s_complete_not_human'),
                      sources_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'terminal.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')
        print(json.dumps({k: v for k, v in result.items() if k not in {'report', 'forward_contexts'}}))


if __name__ == '__main__':
    main()
