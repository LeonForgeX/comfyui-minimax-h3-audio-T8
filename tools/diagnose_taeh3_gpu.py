"""Same cached inputs: OFF/OFF/ON, real eight-forward Avatar and bounded traces."""
import argparse
from functools import wraps
import asyncio
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


def completion_status(args):
    if args.first_forward_repeat_only:
        return 'same_loaded_three_native_forwards_partial_cancel_diagnostic_complete_not_clip_qualification'
    if args.single_route:
        return 'one_fresh_actual_call_complete_diagnostic_not_quality_qualification'
    if args.residency_diagnostic or args.residency_all_casts:
        return 'two_OFF_actual_calls_complete_diagnostic_not_quality_qualification'
    return 'three_actual_calls_complete_diagnostic_not_quality_qualification'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--first-forward-repeat-only', action='store_true')
    mode.add_argument('--single-route', choices=['off1', 'on'])
    mode.add_argument('--residency-diagnostic', action='store_true',
                        help='Two OFF runs; read-only first-forward arguments and selected effective weights')
    mode.add_argument('--residency-all-casts', action='store_true',
                      help='Two OFF runs; read-only first-forward arguments and every observed Core weight cast')
    args = parser.parse_args()
    residency = args.residency_diagnostic or args.residency_all_casts
    project = Path(__file__).resolve().parents[1]
    core = project.parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Fresh owned output required')
    args.output.mkdir(parents=True)
    result = dict(status='incomplete', human_qualified=False, published=False, user_service_changed=False)
    frozen = {str(p.relative_to(project)): digest(p) for p in (project / 'h3_t8').rglob('*.py')}
    result['source_hashes'] = frozen
    sys.path.insert(0, str(project / 'h3_t8/prepared_backend'))
    from resource_guard import SerialProbeLease, NvmlResourceReader
    try:
        with SerialProbeLease(project / 'artifacts/acceleration-research-20260909/serial-gpu.lock'), NvmlResourceReader() as reader:
            sys.path[:0] = [str(core), str(project)]
            sys.argv = ['owned-taeh3-diagnostic', '--disable-pinned-memory', '--reserve-vram', '6', '--preview-method', 'none']
            import comfy.options
            comfy.options.enable_args_parsing()
            import torch
            import nodes
            import execution
            import folder_paths
            import comfy.model_management as mm
            from server import PromptServer
            from comfy_execution.utils import CurrentNodeContext
            from comfy.ldm.minimax.model import MiniMaxH3Model
            import comfy.ops
            torch.set_num_threads(2)
            assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == reader.uuid.removeprefix('GPU-').lower()
            folder_paths.set_output_directory(str(args.output / 'output'))
            folder_paths.set_temp_directory(str(args.output / 'temp'))
            spec = importlib.util.spec_from_file_location('owned_taeh3_diagnostic', project / '__init__.py', submodule_search_locations=[str(project)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)

            class Server:
                client_id = 'owned-diagnostic'
                last_node_id = None
                def send_sync(self, event, data, sid=None):
                    row = dict(event=str(event), client=sid, data={k: v for k, v in data.items() if k != 'frames'})
                    with (args.output / 'events.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(row, default=str) + '\n')

            async def prepare():
                for name in ('nodes_custom_sampler.py', 'nodes_audio.py', 'nodes_video.py'):
                    assert await nodes.load_custom_node(str(core / 'comfy_extras' / name), module_parent='comfy_extras')
                for cls in await package.comfy_entrypoint().get_node_list():
                    nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
                graph = json.loads(args.recipe.read_text(encoding='utf8'))
                check = await execution.validate_prompt(str(uuid.uuid4()), graph, None)
                assert check[0] and not check[3] and set(check[2]) == {'16', '17'}, str(check)
                executor = execution.PromptExecutor(Server(), cache_args={'ram': 16., 'ram_inactive': 16., 'lru': 0})
                await executor.execute_async(graph, str(uuid.uuid4()), {'client_id': 'owned-diagnostic'}, ['9', '10', '17'])
                assert executor.success, str(executor.status_messages)
                settings = {}
                for key, value in graph['11']['inputs'].items():
                    if isinstance(value, list):
                        cache = await executor.caches.outputs.get(value[0])
                        settings[key] = cache.outputs[value[1]][0]
                    else:
                        settings[key] = value
                return settings, executor

            settings, executor = asyncio.run(prepare())
            from owned_taeh3_diagnostic import taeh3_sampling_preview as preview
            from owned_taeh3_diagnostic.avatar_progressive_entry import sample_avatar_progressive
            from owned_taeh3_diagnostic.preview_execution_context import CONTEXT
            records, current = {}, []
            original_forward, original_decoder = MiniMaxH3Model.forward, preview.TinyDecoder
            original_cast = comfy.ops.cast_bias_weight
            residency_modules, captured_weights = {}, {}
            capturing = False

            def fingerprint(value):
                """Content only; never serialize callable execution identity as a portable cache key."""
                if isinstance(value, torch.Tensor):
                    if hasattr(value, '_qdata'):
                        return dict(layout=str(value._layout_cls), qdata=fingerprint(value._qdata),
                                    params={k: fingerprint(v) for k, v in vars(value._params).items()})
                    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value))
                if isinstance(value, dict):
                    return {str(k): fingerprint(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [fingerprint(v) for v in value]
                if value is None or isinstance(value, (bool, int, float, str)):
                    return value
                if callable(value):
                    return dict(callable_type=f'{type(value).__module__}.{type(value).__qualname__}',
                                execution_identity_not_compared=True)
                if type(value).__name__ == 'PackedLayout':
                    return dict(type='PackedLayout', content=fingerprint(vars(value)))
                return dict(type=f'{type(value).__module__}.{type(value).__qualname__}', opaque=True)

            @wraps(original_cast)
            def traced_cast(module, *items, **kwargs):
                output = original_cast(module, *items, **kwargs)
                if capturing and id(module) in residency_modules:
                    captured_weights[residency_modules[id(module)]] = fingerprint(output[:2])
                return output

            def parts(value):
                return [item.detach().cpu().clone() for item in value]

            def tensor_hash(value):
                return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

            def rng():
                return [tensor_hash(torch.get_rng_state()), tensor_hash(torch.cuda.get_rng_state())]

            @wraps(original_forward)
            def counted(self, *items, **kwargs):
                nonlocal capturing
                before = parts(items[0])
                payload = kwargs.get('minimax_payload', {}) or {}
                context = kwargs.get('context', items[2] if len(items) > 2 else None)
                row = dict(phase=(CONTEXT.get() or {}).get('phase'), input=before,
                    context_sha256=tensor_hash(context) if isinstance(context, torch.Tensor) else None,
                    reference_video_sha256=[tensor_hash(t) for t in payload.get('cond_video_latents', [])],
                    reference_audio_sha256=[tensor_hash(t) for t in payload.get('cond_audio_latents', [])])
                if residency and not current:
                    patch_keys = {str(key[0] if isinstance(key, tuple) else key) for key in settings['model'].patches}
                    candidates = [(name, module) for name, module in self.named_modules()
                                  if 'diffusion_model.' + name + '.weight' in patch_keys
                                  and getattr(module, 'quant_format', None) == 'int8_tensorwise']
                    assert len(candidates) >= 6, 'Expected real LoRA-patched modules'
                    selected = candidates[:2] + candidates[len(candidates)//2:len(candidates)//2+2] + candidates[-2:]
                    if args.residency_all_casts:
                        selected = [(name, module) for name, module in self.named_modules()
                                    if hasattr(module, 'comfy_cast_weights') and isinstance(getattr(module, 'weight', None), torch.Tensor)]
                    residency_modules.clear()
                    residency_modules.update((id(module), name) for name, module in selected)
                    captured_weights.clear()
                    row['first_forward_arguments'] = fingerprint(dict(items=items, kwargs=kwargs))
                    row['selected_modules'] = [name for name, _ in selected]
                    result['model_patcher_type'] = type(settings['model']).__qualname__
                    row['selected_host_weights'] = ({} if args.residency_all_casts else
                                                    {name: fingerprint(module.weight) for name, module in selected})
                    capturing = True
                if args.first_forward_repeat_only:
                    repeats = []
                    for index in range(3):
                        state = rng()
                        output = original_forward(self, *items, **kwargs)
                        repeats.append(parts(output))
                        result.setdefault('immediate_rng', []).append(dict(index=index, unchanged=state == rng()))
                        assert all(torch.equal(a, b) for a, b in zip(before, parts(items[0]))), 'Native input mutated'
                    result['same_loaded_native_forward'] = dict(actual_forwards=3,
                        repeat_max_abs_difference=[differences(repeats[0], other) for other in repeats[1:]],
                        same_call_arguments=True, sampling_callback_calls=0, decoder_calls=0,
                        qualification='Three immediate native forwards before any sampling callback; no completed clip')
                    raise mm.InterruptProcessingException()
                try:
                    output = original_forward(self, *items, **kwargs)
                finally:
                    capturing = False
                if residency and not current:
                    if not args.residency_all_casts:
                        assert set(captured_weights) == set(residency_modules.values()), 'Selected casts not actually observed'
                    else:
                        assert len(captured_weights) >= 250, 'Expected whole H3 cast observation'
                    row['unobserved_cast_modules'] = sorted(set(residency_modules.values()) - set(captured_weights))
                    row['selected_effective_weights'] = dict(captured_weights)
                row['output'] = parts(output)
                current.append(row)
                return output

            class TracedDecoder(original_decoder):
                def __init__(self, *items, **kwargs):
                    before = rng()
                    super().__init__(*items, **kwargs)
                    result.setdefault('decoder_rng', []).append(dict(operation='construct', unchanged=before == rng()))
                def decode(self, value):
                    before = rng()
                    input_hash = tensor_hash(value)
                    output = super().decode(value)
                    result.setdefault('decoder_rng', []).append(dict(operation='decode', unchanged=before == rng(), input_unchanged=input_hash == tensor_hash(value)))
                    return output

            def differences(a, b):
                return [float((x-y).abs().max()) for x, y in zip(a, b)]

            old_server = getattr(PromptServer, 'instance', None)
            MiniMaxH3Model.forward, preview.TinyDecoder = counted, TracedDecoder
            if residency:
                comfy.ops.cast_bias_weight = traced_cast
            PromptServer.instance = Server()
            try:
                routes = (('off1', 'off2') if residency else
                          ((args.single_route,) if args.single_route else (('off1',) if args.first_forward_repeat_only else ('off1', 'off2', 'on'))))
                for route in routes:
                    current.clear()
                    options = dict(settings)
                    if route == 'on':
                        config = preview.PreviewSettings(str(core / 'models/vae_approx/taeh3.safetensors'), min_interval_ms=0, max_resolution=128)
                        options['model'] = preview.attach_preview(options['model'], config, '101')
                    start = time.monotonic()
                    try:
                        with CurrentNodeContext(str(uuid.uuid4()), '11'):
                            latent, report = sample_avatar_progressive(**options)
                    except mm.InterruptProcessingException:
                        if not args.first_forward_repeat_only:
                            raise
                        assert result['same_loaded_native_forward']['actual_forwards'] == 3
                        result['diagnostic_cancel_propagated'] = True
                        break
                    final = parts(latent['samples'].unbind())
                    assert len(current) == 8 and json.loads(report)['counts']['actual_forwards'] == {'low': 4, 'high': 4}
                    records[route] = dict(final=final, trace=list(current))
                    torch.save(records[route], args.output / (route + '.pt'))
                    result.setdefault('cases', []).append(dict(route=route, seconds=time.monotonic()-start, actual_forwards=8, report=json.loads(report)))
                    print(json.dumps(dict(completed=route, forwards=8)), flush=True)
                comparisons = (('off_repeat', 'off1', 'off2'),) if residency else (('off_repeat', 'off1', 'off2'), ('preview', 'off2', 'on'))
                for label, first, second in (() if args.first_forward_repeat_only or args.single_route else comparisons):
                    a, b = records[first], records[second]
                    result[label] = dict(final_max_abs_difference=differences(a['final'], b['final']),
                        steps=[dict(phase=x['phase'], input_max_abs_difference=differences(x['input'], y['input']),
                            output_max_abs_difference=differences(x['output'], y['output']),
                            context_same=x['context_sha256'] == y['context_sha256'],
                            refs_same=x['reference_video_sha256'] == y['reference_video_sha256'] and x['reference_audio_sha256'] == y['reference_audio_sha256'])
                            for x, y in zip(a['trace'], b['trace'])])
                    if residency:
                        x, y = a['trace'][0], b['trace'][0]
                        result['residency_diagnostic'] = dict(
                            selected_modules=x['selected_modules'],
                            same_module_selection=x['selected_modules'] == y['selected_modules'],
                            first_forward_arguments_same=x['first_forward_arguments'] == y['first_forward_arguments'],
                            host_weights_same=(None if args.residency_all_casts else
                                               x['selected_host_weights'] == y['selected_host_weights']),
                            host_weight_fingerprint_scope=('not_measured' if args.residency_all_casts else 'six_selected_modules'),
                            effective_weights_same=x['selected_effective_weights'] == y['selected_effective_weights'],
                            observed_cast_counts=[len(x['selected_effective_weights']), len(y['selected_effective_weights'])],
                            unobserved_cast_modules=[x['unobserved_cast_modules'], y['unobserved_cast_modules']],
                            first_forward_arguments=[x['first_forward_arguments'], y['first_forward_arguments']],
                            host_weights=[x['selected_host_weights'], y['selected_host_weights']],
                            effective_weights=[x['selected_effective_weights'], y['selected_effective_weights']],
                            scope=('Every observed Core cast during first LOW forward; missing casts explicit; no quality qualification'
                                   if args.residency_all_casts else 'Six selected real LoRA casts during first LOW forward; read-only hashes, not all model weights or quality'))
            finally:
                MiniMaxH3Model.forward, preview.TinyDecoder = original_forward, original_decoder
                comfy.ops.cast_bias_weight = original_cast
                PromptServer.instance = old_server
            result['resource_after'] = reader.sample()
            del executor
        assert all(digest(project / name) == value for name, value in frozen.items())
        result.update(status=completion_status(args), sources_unchanged=True,
                      completed_sampling_calls=len(result.get('cases', [])))
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'terminal.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')
        print(json.dumps({k: v for k, v in result.items() if k not in {'source_hashes', 'cases'}}), flush=True)


if __name__ == '__main__':
    main()
