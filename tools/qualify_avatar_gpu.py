"""Approved two73-frame clips plus cancellation/recovery, isolated actual Core."""
import argparse
import asyncio
from functools import wraps
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256')
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--with-preview', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Fresh owned artifact directory required')
    args.output.mkdir(parents=True)
    result = dict(status='incomplete', human_qualified=False, spatial_tiling=False,
                  user_service_changed=False, published=False)
    sys.path[:0] = [str(project / 'h3_t8/prepared_backend')]
    from resource_guard import SerialProbeLease, NvmlResourceReader
    try:
        frozen = {str(p.relative_to(project)): digest(p) for p in (project / 'h3_t8').rglob('*.py')}
        result['source_hashes'] = frozen
        with SerialProbeLease(project / 'artifacts/acceleration-research-20260909/serial-gpu.lock'), NvmlResourceReader() as reader:
            result['resource_before'] = reader.sample()
            sys.path[:0] = [str(args.core), str(project)]
            sys.argv = ['owned-avatar-gpu', '--disable-pinned-memory', '--reserve-vram', '6', '--preview-method', 'none']
            result['isolated_Core_reserve_vram_GiB'] = 6
            import comfy.options
            comfy.options.enable_args_parsing()
            import torch
            import nodes
            import execution
            import folder_paths
            import comfy.model_management as mm
            from comfy.ldm.minimax.model import MiniMaxH3Model
            torch.set_num_threads(2)
            assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == reader.uuid.removeprefix('GPU-').lower()
            folder_paths.set_output_directory(str(args.output / 'output'))
            folder_paths.set_temp_directory(str(args.output / 'temp'))
            spec = importlib.util.spec_from_file_location('owned_avatar_plugin', project / '__init__.py', submodule_search_locations=[str(project)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)

            class Server:
                client_id = None
                last_node_id = None
                preview_cancel_armed = False
                preview_events = []
                def send_sync(self, event, data, sid=None):
                    with (args.output / 'events.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(dict(event=str(event), data=data), default=str) + '\n')
                    if str(event) == 't8-taeh3-sampling-preview':
                        self.preview_events.append(dict(client=sid, **{k: v for k, v in data.items() if k != 'frames'}))
                        if data['kind'] == 'frames' and self.preview_cancel_armed:
                            self.preview_cancel_armed = False
                            mm.interrupt_current_processing(True)

            async def run():
                for name in ('nodes_custom_sampler.py', 'nodes_audio.py', 'nodes_video.py'):
                    assert await nodes.load_custom_node(str(args.core / 'comfy_extras' / name), module_parent='comfy_extras')
                for cls in await package.comfy_entrypoint().get_node_list():
                    nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
                recipe = json.loads(args.recipe.read_text(encoding='utf8'))
                assert recipe['18']['inputs']['ensure_minimum_context'] is False, '73f probe must not expand minimum context'
                server = Server()
                executor = execution.PromptExecutor(server, cache_args={'ram': 16., 'ram_inactive': 16., 'lru': 0})
                counter = []
                original_forward = MiniMaxH3Model.forward
                @wraps(original_forward)
                def counted(self, *items, **kwargs):
                    output = original_forward(self, *items, **kwargs)
                    value = items[0]
                    shapes = [list(t.shape) for t in value] if isinstance(value, (tuple, list)) else list(value.shape)
                    counter.append(dict(shape=shapes, completed=time.time()))
                    return output
                MiniMaxH3Model.forward = counted
                cases = []

                async def execute(name, graph, expected, interrupted=False):
                    validation = await execution.validate_prompt(str(uuid.uuid4()), graph, None)
                    if not validation[0]:
                        raise RuntimeError(json.dumps(validation, default=str))
                    start, before = time.monotonic(), len(counter)
                    await executor.execute_async(graph, str(uuid.uuid4()), {'client_id': 'owned-probe'}, ['16', '17'])
                    events = [e for e, _ in executor.status_messages]
                    success = 'execution_success' in events and 'execution_error' not in events
                    item = dict(name=name, success=success, seconds=time.monotonic()-start,
                        forwards=len(counter)-before, validation=validation, resource_after=reader.sample(),
                        status_events=events, executor_success=executor.success)
                    if interrupted:
                        valid_count = 1 <= item['forwards'] < 8 if name == 'cancel_after_actual_preview' else item['forwards'] == 1
                        if success or 'execution_interrupted' not in events or not valid_count:
                            raise RuntimeError('Actual cancellation contract failed: ' + json.dumps(item))
                        assert await executor.caches.outputs.get('16') is None
                        item['interrupted_without_output'] = True
                    else:
                        if not success or item['forwards'] != expected:
                            errors = [{k: v for k, v in data.items() if k in {'node_id', 'exception_message', 'exception_type', 'traceback'}} for event, data in executor.status_messages if event == 'execution_error']
                            raise RuntimeError('Actual generation failed: ' + json.dumps(item) + str(errors))
                        item['history'] = executor.history_result
                        if name.startswith('avatar_progressive'):
                            cached = await executor.caches.outputs.get('11')
                            item['avatar_report'] = json.loads(cached.outputs[1][0])
                            assert item['avatar_report']['counts']['actual_forwards'] == {'low': 4, 'high': 4}
                        # Source input, locked latent and explicit delivered source recording.
                        from owned_avatar_plugin.core import nested_av_parts
                        condition = await executor.caches.outputs.get('8')
                        sampled = await executor.caches.outputs.get('13' if name == 'native8_control' else '11')
                        source_audio = nested_av_parts(condition.outputs[1][0])[1]
                        final_audio = nested_av_parts(sampled.outputs[0][0])[1]
                        item['source_audio_latent_max_abs_diff'] = float((source_audio-final_audio).abs().max())
                        if item['source_audio_latent_max_abs_diff'] > 1e-5:
                            raise RuntimeError('Source audio anchor changed')
                    cases.append(item)
                    (args.output / (name + '.json')).write_text(json.dumps(item, indent=2, default=str), encoding='utf8')

                try:
                    baseline = deepcopy(recipe)
                    baseline['11'] = {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 20260918}}
                    baseline['12'] = {'class_type': 'BasicGuider', 'inputs': {'model': ['9', 0], 'conditioning': ['8', 0]}}
                    baseline['13'] = {'class_type': 'SamplerCustomAdvanced', 'inputs': {
                        'noise': ['11', 0], 'guider': ['12', 0], 'sampler': ['9', 1], 'sigmas': ['9', 2], 'latent_image': ['8', 1]}}
                    baseline['14']['inputs']['av_latent'] = ['13', 0]
                    baseline['16']['inputs']['filename_prefix'] = 'native8_control'
                    if args.control_root:
                        previous = json.loads((args.control_root / 'terminal.json').read_text(encoding='utf8'))
                        assert previous['source_hashes'] == frozen
                        control = json.loads((args.control_root / 'native8_control.json').read_text(encoding='utf8'))
                        assert control['success'] and control['forwards'] == 8
                        cases.append(control)
                        result['control_reused_from'] = str(args.control_root)
                        result['control_resource_policy_note'] = 'Control used3GiB; current6GiB limits weight residency only. Sampling/model/seed/recording unchanged; not a matched performance comparison.'
                    else:
                        await execute('native8_control', baseline, 8)
                    # Cancellation hits after one actual LOW forward. Patch only this
                    # owned process's entry callback; real InterruptProcessingException.
                    from owned_avatar_plugin import nodes_avatar_progressive as entry
                    original = entry.sample_avatar_progressive
                    def cancelled(**kwargs):
                        def stop(*items):
                            raise mm.InterruptProcessingException()
                        kwargs['callback'] = stop
                        return original(**kwargs)
                    entry.sample_avatar_progressive = cancelled
                    cancel_graph = deepcopy(recipe)
                    cancel_graph['16']['inputs']['filename_prefix'] = 'cancelled_must_not_exist'
                    try:
                        await execute('cancel_low', cancel_graph, 0, interrupted=True)
                    finally:
                        entry.sample_avatar_progressive = original
                        nodes.interrupt_processing(False)
                    recovered = deepcopy(recipe)
                    recovered['16']['inputs']['filename_prefix'] = 'avatar_progressive_after_cancel'
                    await execute('avatar_progressive_after_cancel', recovered, 8)
                    if args.with_preview:
                        from server import PromptServer
                        from owned_avatar_plugin.taeh3_sampling_preview import EVENT
                        assert EVENT == 't8-taeh3-sampling-preview'
                        old_server = getattr(PromptServer, 'instance', None)
                        cached = await executor.caches.outputs.get('11')
                        before_preview = [part.clone() for part in cached.outputs[0][0]['samples'].unbind()]
                        preview_graph = deepcopy(recovered)
                        preview_graph['101'] = dict(class_type='MiniMaxH3TAEH3SamplingPreviewEXPT8', inputs=dict(
                            model=['9', 0], enabled=True, checkpoint='taeh3.safetensors', phase='low',
                            update_every_steps=2, min_interval_ms=0, max_resolution=128,
                            latent_prefix='7', frames=12, fps=12, jpeg_quality=75))
                        preview_graph['11']['inputs']['model'] = ['101', 0]
                        preview_graph['16']['inputs']['filename_prefix'] = 'cancel_preview_must_not_exist'
                        PromptServer.instance = server
                        server.preview_cancel_armed = True
                        try:
                            await execute('cancel_after_actual_preview', preview_graph, 0, interrupted=True)
                            assert not server.preview_cancel_armed, 'Cancellation must follow an actual decoded preview'
                            nodes.interrupt_processing(False)
                            preview_graph['16']['inputs']['filename_prefix'] = 'avatar_progressive_preview_after_cancel'
                            await execute('avatar_progressive_preview_after_cancel', preview_graph, 8)
                            cached = await executor.caches.outputs.get('11')
                            after_preview = cached.outputs[0][0]['samples'].unbind()
                            differences = [float((a-b).abs().max()) for a, b in zip(before_preview, after_preview)]
                            assert differences == [0., 0.], f'Read-only preview changed final AV: {differences}'
                            assert all(item['client'] == 'owned-probe' for item in server.preview_events)
                            assert all(item['phase'] == 'low' for item in server.preview_events)
                            assert not any(item['kind'] == 'unavailable' for item in server.preview_events)
                            result['taeh3'] = dict(final_AV_max_abs_differences=differences,
                                actual_private_events=server.preview_events, real_checkpoint=str(args.core / 'models/vae_approx/taeh3.safetensors'),
                                fake_transport_not_live_browser=True, read_only_GPU_qualified=True)
                        finally:
                            PromptServer.instance = old_server
                            nodes.interrupt_processing(False)
                finally:
                    MiniMaxH3Model.forward = original_forward
                return cases
            result['cases'] = asyncio.run(run())
            # Decode every frame/audio block; no sparse-thumbnail qualification.
            import av
            media = []
            media_paths = list((args.output / 'output').rglob('*.mp4'))
            if args.control_root:
                media_paths += list((args.control_root / 'output').rglob('native8_control*.mp4'))
            for path in sorted(media_paths):
                with av.open(str(path)) as c:
                    vs = c.streams.video[0]
                    dims = [vs.width, vs.height]
                    frames = sum(1 for _ in c.decode(video=0))
                with av.open(str(path)) as c:
                    blocks = sum(1 for _ in c.decode(audio=0))
                assert dims == [512, 768] and frames == 73 and blocks > 0, f'Unexpected media: {path}: {dims}/{frames}/{blocks}'
                media.append(dict(path=str(path), sha256=digest(path), frames=frames, dimensions=dims, audio_blocks=blocks))
            assert len(media) == (3 if args.with_preview else 2)
            result['media'] = media
            result['resource_after'] = reader.sample()
        assert all(digest(project / name) == value for name, value in frozen.items())
        result.update(status='actual_native8_vs_avatar4plus4_full_AV_and_cancel_recovery_pass_not_human', sources_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'terminal.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')
        print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, default=str), flush=True)


if __name__ == '__main__':
    main()
