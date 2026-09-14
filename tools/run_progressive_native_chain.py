"""One isolated full-trained 0.4MP/8s progressive chain. CPU API audit by default.

No retries, downloads, production deployment or quality claims. GPU execution
requires --mode gpu; all resource and process ownership guards remain enabled.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402
from build_dual_model_workflows import SCENE_PROMPT, RELAY_EVENTS  # noqa: E402


ASSETS = {
    'diffusion_models': ['minimax_h3_fl2va_int8_convrot.safetensors'],
    'text_encoders': ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
    'vae': ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_audio_vae_fp32.safetensors'],
    'loras': ['minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors'],
    'latent_upscale_models': ['minimax_h3_latent_upscaler_3d_fp16.safetensors'],
}


def build_graph(chain_id, backend='kj-memory', relay=True, eav='disabled', tst='disabled'):
    if backend not in {'kj-memory', 'pytorch', 'sol'} or eav not in {'disabled', 'report_only', 'apply_exp'} or tst not in {'disabled', 'report_only', 'apply_exp'}:
        raise ValueError('Unknown explicit probe configuration')
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': ASSETS['diffusion_models'][0], 'weight_dtype': 'default'}},
        '2': {'class_type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced', 'inputs': {
            'model': ['1', 0], 'lora_name': ASSETS['loras'][0], 'strength_model': 1.}},
        '3': {'class_type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced', 'inputs': {
            'model': ['1', 0], 'lora_name': ASSETS['loras'][0], 'strength_model': 1.}},
        '4': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': ASSETS['text_encoders'][0], 'type': 'minimax', 'device': 'default'}},
        '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': ASSETS['vae'][0]}},
        '6': {'class_type': 'VAELoader', 'inputs': {'vae_name': ASSETS['vae'][1]}},
        '8': {'class_type': 'T8ProgressiveNativeChainProbe', 'inputs': {
            'model': ['21', 0], 'model_hires': ['22', 0], 'clip': ['4', 0],
            'video_vae': ['5', 0], 'audio_vae': ['6', 0], 'chain_id': chain_id,
            'global_prompt': SCENE_PROMPT, 'upscaler_model': ASSETS['latent_upscale_models'][0],
            'eav_mode': eav, 'tst_mode': tst}},
        '9': {'class_type': 'PreviewAny', 'inputs': {'source': ['8', 1]}},
        '10': {'class_type': 'PreviewAny', 'inputs': {'source': ['8', 0]}},
    }
    for node_id, source in [('21', '2'), ('22', '3')]:
        if backend == 'kj-memory':
            graph[node_id] = {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch', 'inputs': {'model': [source, 0]}}
        elif backend == 'sol':
            graph[node_id] = {'class_type': 'SolAttentionPatch', 'inputs': {'model': [source, 0],
                'enabled': True, 'tau': .5, 'min_tokens': 4096, 'strict': True,
                'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}}
        else:
            graph[node_id] = {'class_type': 'ModelAttentionBackend', 'inputs': {'model': [source, 0], 'attention': 'pytorch attention'}}
    if relay:
        graph['7'] = {'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced', 'inputs': {
            'global_prompt': SCENE_PROMPT, 'local_prompts': RELAY_EVENTS, 'length': 193,
            'timing_mode': 'percent', 'time_ranges': '0-15\n15-40\n40-75\n75-100',
            'math_profile': 'paper_v1', 'epsilon': .1, 'allow_gaps': False, 'allow_overlaps': False}}
        graph['11'] = {'class_type': 'MiniMaxH3PromptRelayQueryRouteT8Advanced',
                       'inputs': {'prompt_relay_plan': ['7', 0], 'query_route': 'joint_av_exp'}}
        graph['8']['inputs']['prompt_relay_plan'] = ['11', 0]
    return graph


def asset_paths(core):
    result = []
    for category, names in ASSETS.items():
        for name in names:
            path = (core / 'models' / category / name).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(core / 'models'):
                raise ValueError('Expected a local model file within Core/models')
            result.append(path)
    return result


def startup_resources(reader, guard, *, prior_preflight=None):
    sample = reader.sample()
    if prior_preflight is not None:
        if (sample.get('gpu_uuid') != prior_preflight.get('gpu_uuid')
                or sample.get('monotonic', 0) <= prior_preflight.get('monotonic', 0)):
            raise RuntimeError('Startup resource guard: device or clock changed after preflight')
    reason = guard.observe(sample, startup=True)
    if reason:
        raise RuntimeError('Startup resource guard: ' + reason)
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--interface', choices=['probe', 'public'], default='probe')
    parser.add_argument('--port', type=int, default=8217)
    parser.add_argument('--backend', choices=['kj-memory', 'pytorch', 'sol'], default='kj-memory')
    parser.add_argument('--no-relay', action='store_true')
    parser.add_argument('--eav', choices=['disabled', 'report_only', 'apply_exp'], default='disabled')
    parser.add_argument('--tst', choices=['disabled', 'report_only', 'apply_exp'], default='disabled')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    core, root = args.core.resolve(strict=True), args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a fresh directory within this development worktree artifacts')
    files = asset_paths(core)
    graph = build_graph('progressive_' + root.name, args.backend, not args.no_relay, args.eav, args.tst)
    transport.CORE, transport.PROJECT = core, project
    base_command = transport.server_command

    def command(*values):
        result = base_command(*values)
        at = result.index('--whitelist-custom-nodes') + 1
        result[at:at] = ['progressive_native_probe_extension',
                        'ComfyUI-sol-attn' if args.backend == 'sol' else 'ComfyUI-KJNodes']
        return result + ['--disable-pinned-memory']

    transport.server_command = command
    root.mkdir(parents=True)
    server = transport.OwnedServer(root, args.port, args.mode == 'cpu', 2)
    guard, monitor = ResourceGuard(), None
    result = dict(status='incomplete', mode=args.mode, human_qualified=False)
    lease = core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            preflight = None
            if args.mode == 'gpu':
                # CPU-only hashing is not a continuously observed GPU run.
                # Keep separate phase guards rather than misclassifying the
                # intentional hashing interval as missing runtime telemetry.
                preflight = startup_resources(reader, ResourceGuard())
                transport.write_json(root / 'startup-before-hashing.json', preflight)
            sources = transport.source_snapshot()
            for path in (Path(__file__), project / 'tools/progressive_native_probe_extension/__init__.py',
                         project / 'tools/build_progressive_long_workflows.py'):
                sources[path.relative_to(project).as_posix()] = file_identity(path)['sha256']
            assets = [file_identity(path) for path in files] if args.mode == 'gpu' else []
            expected = dict(core=verify_core_source(core), sources=sources, assets=assets,
                mode='cpu-smoke' if args.mode == 'cpu' else 'gpu', pilot_graphs={'native_chain': graph},
                runtime_options={'reserve_vram_gib': 5, 'headroom_gib': 2})
            transport.write_json(root / 'paths.json', probe_resource_config(core, project))
            if args.mode == 'gpu':
                # Hashing all trained weights can take minutes. The first
                # observation must not authorize a server after resources change.
                transport.write_json(root / 'startup-after-hashing.json',
                                     startup_resources(reader, guard, prior_preflight=preflight))
            server.start()
            if args.mode == 'gpu':
                monitor = transport.ContinuousGuard(reader, guard, root / 'resources.jsonl', server)
                cleanup.callback(monitor.close)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            transport.wait_ready(server, check)
            info = server.request('GET', '/object_info')
            transport.write_json(root / 'object-info.json', info)
            if args.interface == 'public':
                from tools.build_progressive_long_workflows import build_public_graph, build_workflow
                graph = build_public_graph(info, 'progressive_' + root.name, args.backend,
                                           not args.no_relay, args.eav, args.tst)
                expected['pilot_graphs'] = {'native_chain_public': graph}
                transport.write_json(root / 'UNREVIEWED.workflow.json', build_workflow(graph, info))
            transport.write_json(root / 'expected.json', expected)
            history, _ = transport.execute_graph(server, {
                '1': {'class_type': 'T8ProgressiveEnvironmentAudit', 'inputs': {'expected_json': json.dumps(expected)}},
                '2': {'class_type': 'PreviewAny', 'inputs': {'source': ['1', 0]}}}, root / 'environment', check)
            result['environment'] = transport.preview_report(history, '2')
            if args.mode == 'gpu':
                history, timing = transport.execute_graph(server, graph, root / 'generation', check, timeout=3600)
                result.update(status='generated_independent_audit_and_human_review_pending', timing=timing,
                              report=transport.preview_report(history, '9'))
            else:
                result['status'] = 'CPU_API_graph_validated_no_model_generation'
            for relative, sha in sources.items():
                if file_identity(project / relative)['sha256'] != sha:
                    raise RuntimeError('Source changed during isolated probe: ' + relative)
            for asset in assets:
                if file_identity(asset['path']) != asset:
                    raise RuntimeError('Model file changed during isolated probe')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if monitor:
                monitor.close()
        finally:
            server.stop()
            result.update(server_stop=server.stop_receipt, resources=guard.report())
            transport.write_json(root / 'terminal.json', result)
            print(json.dumps({'status': result['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
