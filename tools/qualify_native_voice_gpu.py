"""Two minimal native Ref2VA voice/emotion clips; reference is never delivery."""
import argparse
import asyncio
from copy import deepcopy
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


VOICE_BINDING = ' S1 uses the vocal identity/timbre reference in <Audio 1>; the reference is not the target dialogue.'


def without_voice_reference(graph):
    """One fixed qualification recipe ablation, not a general workflow rewrite."""
    graph = deepcopy(graph)
    inputs = graph['8']['inputs']
    if inputs.get('ref_audios.ref_audio_0') != ['7', 0] or '7' not in graph:
        raise ValueError('Expected the independently connected reference recording')
    prompt = inputs['prompt']
    if prompt.count(VOICE_BINDING) != 1:
        raise ValueError('Expected exactly one known voice-reference binding')
    inputs['prompt'] = prompt.replace(VOICE_BINDING, '')
    if '<Audio' in inputs['prompt']:
        raise ValueError('An unbound audio tag would remain')
    del inputs['ref_audios.ref_audio_0']
    del graph['7']
    for node in graph.values():
        if any(isinstance(value, list) and len(value) == 2 and value[0] == '7'
               for value in node['inputs'].values()):
            raise ValueError('Reference recording is also connected elsewhere')
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-ablation', action='store_true',
                        help='One neutral WITH-reference and one neutral WITHOUT-reference; do not repeat emotion')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    core = project.parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Fresh owned output required')
    args.output.mkdir(parents=True)
    result = dict(status='incomplete', human_qualified=False, voice_identity_qualified=False,
                  emotion_qualified=False, published=False, user_service_changed=False)
    sys.path.insert(0, str(project / 'h3_t8/prepared_backend'))
    from resource_guard import SerialProbeLease, NvmlResourceReader
    frozen = {str(p.relative_to(project)): digest(p) for p in (project / 'h3_t8').rglob('*.py')}
    result['source_hashes'] = frozen
    try:
        with SerialProbeLease(project / 'artifacts/acceleration-research-20260909/serial-gpu.lock'), NvmlResourceReader() as reader:
            result['resource_before'] = reader.sample()
            sys.path[:0] = [str(core), str(project)]
            sys.argv = ['owned-native-voice', '--disable-pinned-memory', '--reserve-vram', '6', '--preview-method', 'none']
            import comfy.options
            comfy.options.enable_args_parsing()
            import torch
            import nodes
            import execution
            import folder_paths
            from comfy.ldm.minimax.model import MiniMaxH3Model
            torch.set_num_threads(2)
            assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == reader.uuid.removeprefix('GPU-').lower()
            folder_paths.set_output_directory(str(args.output / 'output'))
            folder_paths.set_temp_directory(str(args.output / 'temp'))
            spec = importlib.util.spec_from_file_location('owned_voice_plugin', project / '__init__.py', submodule_search_locations=[str(project)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)
            class Server:
                client_id = None
                last_node_id = None
                def send_sync(self, event, data, sid=None):
                    with (args.output / 'events.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(dict(event=str(event), data=data), default=str) + '\n')

            async def run():
                for name in ('nodes_custom_sampler.py', 'nodes_audio.py', 'nodes_video.py'):
                    assert await nodes.load_custom_node(str(core / 'comfy_extras' / name), module_parent='comfy_extras')
                for cls in await package.comfy_entrypoint().get_node_list():
                    nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
                executor = execution.PromptExecutor(Server(), cache_args={'ram': 16., 'ram_inactive': 16., 'lru': 0})
                if args.reference_ablation:
                    neutral = json.loads((args.recipes / '2026-09-18_T8_voice_neutral_EXP.api.json').read_text(encoding='utf8'))
                    assets = {
                        'diffusion': folder_paths.get_full_path_or_raise('diffusion_models', neutral['1']['inputs']['unet_name']),
                        'clip': folder_paths.get_full_path_or_raise('text_encoders', neutral['3']['inputs']['clip_name']),
                        'video_vae': folder_paths.get_full_path_or_raise('vae', neutral['4']['inputs']['vae_name']),
                        'audio_vae': folder_paths.get_full_path_or_raise('vae', neutral['5']['inputs']['vae_name']),
                        'picture': str(Path(folder_paths.get_input_directory()) / neutral['6']['inputs']['image']),
                        'reference': str(Path(folder_paths.get_input_directory()) / neutral['7']['inputs']['audio']),
                    }
                    result['bound_assets'] = {name: dict(path=str(path), bytes=Path(path).stat().st_size, sha256=digest(path))
                                              for name, path in assets.items()}
                    core_sources = ['comfy/ldm/minimax/model.py', 'comfy/ops.py', 'comfy/model_patcher.py',
                                    'comfy/lora.py', 'comfy/text_encoders/minimax.py', 'execution.py']
                    result['Core_sources'] = {name: digest(core / name) for name in core_sources}
                completed = []
                native_audio_condition_counts = []
                original = MiniMaxH3Model.forward
                @wraps(original)
                def counted(self, *items, **kwargs):
                    if args.reference_ablation:
                        payload = kwargs.get('minimax_payload') or {}
                        count = len(payload.get('cond_audio_latents', []))
                        assert count == int(has_reference), 'Actual native audio reference differs from the ablation'
                        native_audio_condition_counts.append(count)
                    value = original(self, *items, **kwargs)
                    completed.append(time.time())
                    return value
                MiniMaxH3Model.forward = counted
                cases = []
                try:
                    routes = ('voice_reference_with', 'voice_reference_without') if args.reference_ablation else ('voice_neutral', 'voice_emotion')
                    for route in routes:
                        source_route = 'voice_neutral' if args.reference_ablation else route
                        path = args.recipes / ('2026-09-18_T8_' + source_route + '_EXP.api.json')
                        graph = json.loads(path.read_text(encoding='utf8'))
                        has_reference = route != 'voice_reference_without'
                        if not has_reference:
                            graph = without_voice_reference(graph)
                        assert graph['1']['inputs']['unet_name'] == 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
                        assert graph['8']['inputs']['audio_mode'] == 'native' and graph['8']['inputs']['task_type'] == 'Ref2VA'
                        assert graph['9']['inputs']['steps'] == 20 and '2' not in graph
                        assert graph['15']['inputs']['audio'] == ['14', 1]
                        assert all(name not in graph['8']['inputs'] for name in ('drive_audio', 'final_audio'))
                        graph['16']['inputs']['filename_prefix'] = route
                        execution_graph_hash = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                        (args.output / (route + '.executed.api.json')).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding='utf8')
                        validation = await execution.validate_prompt(str(uuid.uuid4()), graph, None)
                        assert validation[0] and not validation[3] and set(validation[2]) == {'16', '17'}, str(validation)
                        start, before = time.monotonic(), len(completed)
                        await executor.execute_async(graph, str(uuid.uuid4()), {'client_id': 'owned-voice'}, ['16', '17'])
                        events = [event for event, _ in executor.status_messages]
                        if 'execution_success' not in events or 'execution_error' in events:
                            errors = [{k: v for k, v in data.items() if k in {'node_id', 'exception_message', 'exception_type', 'traceback'}} for event, data in executor.status_messages if event == 'execution_error']
                            raise RuntimeError(str(errors))
                        assert len(completed)-before == 20, f'Expected20 actual forwards; got {len(completed)-before}'
                        condition = await executor.caches.outputs.get('8')
                        media = json.loads(condition.outputs[4][0])
                        # The socket is zero-indexed, but conditioning's public
                        # labels are one-indexed. Do not change production mapping.
                        assert media['audios'] == ({'1': 'ref_audio_1'} if has_reference else {}) and media['source_audio_ordinal'] is None, str(media)
                        assert media['pictures'] == {'1': 'ref_image_1'}, str(media)
                        assert condition.outputs[2][0] is None
                        assert 'audio_mode=native' in condition.outputs[5][0] and 'frames=73 ' in condition.outputs[5][0]
                        decode = await executor.caches.outputs.get('14')
                        generated = decode.outputs[1][0]
                        reference = (await executor.caches.outputs.get('7')).outputs[0][0] if has_reference else None
                        wave = generated['waveform']
                        assert torch.isfinite(wave).all() and generated['sample_rate'] > 0 and wave.numel() > 0
                        assert reference is None or wave is not reference['waveform']
                        item = dict(route=route, recipe_sha256=digest(path), actual_forwards=20,
                            executed_graph_sha256=execution_graph_hash, voice_reference_connected=has_reference,
                            seconds=time.monotonic()-start, media_map=media, condition_report=condition.outputs[5][0],
                            source_reference_only=has_reference, delivered_generated_audio=True,
                            original_reference_samples=reference['waveform'].shape[-1] if reference else None, generated_samples=wave.shape[-1],
                            generated_sample_rate=generated['sample_rate'], generated_rms=float(wave.square().mean().sqrt()),
                            generated_peak=float(wave.abs().max()), history=executor.history_result,
                            cross_language_reference_experimental=has_reference)
                        if args.reference_ablation:
                            item['actual_native_audio_condition_counts'] = native_audio_condition_counts[before:]
                            assert item['actual_native_audio_condition_counts'] == [int(has_reference)] * 20
                        cases.append(item)
                        (args.output / (route + '.json')).write_text(json.dumps(item, indent=2, default=str), encoding='utf8')
                finally:
                    MiniMaxH3Model.forward = original
                return cases
            result['cases'] = asyncio.run(run())
            import av
            clips = []
            for path in (args.output / 'output').glob('*.mp4'):
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    assert [stream.width, stream.height] == [512, 768] and stream.codec_context.name == 'h264'
                    frames = sum(1 for _ in container.decode(video=0))
                with av.open(str(path)) as container:
                    blocks = sum(1 for _ in container.decode(audio=0))
                assert frames == 73 and blocks > 0
                clips.append(dict(path=str(path), sha256=digest(path), frames=frames, audio_blocks=blocks))
            assert len(clips) == 2
            result.update(media=clips, resource_after=reader.sample())
            if args.reference_ablation:
                assert all(digest(row['path']) == row['sha256'] for row in result['bound_assets'].values())
                assert all(digest(core / name) == value for name, value in result['Core_sources'].items())
                result['assets_and_Core_unchanged'] = True
        assert all(digest(project / name) == value for name, value in frozen.items())
        result.update(status=('actual_native_Ref2VA_reference_ablation_two73frame_generated_AV_complete_not_human'
                              if args.reference_ablation else 'actual_native_Ref2VA_two73frame_generated_AV_complete_not_voice_or_emotion_human_qualified'), sources_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'terminal.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}), flush=True)


if __name__ == '__main__':
    main()
