"""Owned, guarded 0.4MP/8s standard4+4 two-window GPU probe (not quality acceptance)."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_progressive_pilot as transport  # noqa: E402
from run_h3_memory_node_probe import audit_media  # noqa: E402
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def build_graph():
    project = Path(__file__).resolve().parents[1]
    workflow = json.loads((project / 'examples/workflows/13-latent-upscale/'
        '2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json').read_text(encoding='utf-8'))
    nodes = {node['id']: node for node in workflow['nodes']}
    graph = {}
    widget_names = {
        1: ['vae_name'], 2: ['vae_name'], 3: ['clip_name', 'type', 'device'],
        4: ['unet_name', 'weight_dtype'], 5: ['image'],
        6: ['upscale_method', 'width', 'height', 'crop'],
        7: ['prompt', 'width', 'height', 'length', 'task_type', 'audio_mode',
            'audio_denoise_strength', 'add_source_as_reference', 'prompt_primary_audio_ordinal',
            'strict_prompt_tags', 'ref_image_size', 'reference_video_policy', 'allow_above_reference_area'],
        8: ['steps', 'shift_video', 'shift_audio', 'sampler_name', 'scheduler'],
        9: ['base_steps', 'coarse_steps', 'refine_steps'],
        10: [], 11: ['noise_seed'], 12: [], 13: [],
        14: ['model_name', 'target_width', 'target_height', 'temporal_chunk_frames',
            'temporal_overlap_frames', 'anchor_strength', 'tile_width', 'tile_height',
            'spatial_overlap', 'spatial_fade', 'minimum_tile_size', 'overlap_blend',
            'precision', 'release_policy', 'spatial_strategy', 'sampling_contract',
            'size_mode', 'scale_by', 'target_megapixels', 'aspect_policy', 'max_anisotropy'],
        15: ['cfg'], 16: [], 17: [], 18: [], 19: ['fps', 'bit_depth', 'color_space', 'codec'],
        23: ['lora_name', 'strength_model'], 24: ['noise_seed'],
        25: ['filename_prefix'],
    }
    widget_names[13] = widget_names[7]
    for node_id, names in widget_names.items():
        node = nodes[node_id]
        graph[str(node_id)] = {'class_type': node['type'], 'inputs':
            dict(zip(names, node.get('widgets_values', [])))}
    for _, source, slot, target, target_slot, _ in workflow['links']:
        if str(target) in graph:
            graph[str(target)]['inputs'][nodes[target]['inputs'][target_slot]['name']] = [str(source), slot]
    values = nodes[25]['widgets_values']
    graph['25']['inputs']['format'] = {'format':values[1], 'codec':{
        'codec':values[2], 'encoding':{'encoding':values[3]}}}
    # Global selector only; no attention.forward replacement. Explicit probe
    # difference from the plain workflow, recorded in expected.json.
    graph['26'] = {'class_type': 'PathchSageAttentionKJ', 'inputs': {
        'model': ['23', 0], 'sage_attention': 'auto', 'allow_compile': False}}
    graph['8']['inputs']['model'] = ['26', 0]
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8236)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new artifact directory inside this project')
    graph = build_graph()
    transport.CORE, transport.PROJECT = args.core.resolve(), project
    original_command = transport.server_command
    def command(*values):
        result = original_command(*values)
        result.insert(result.index('--whitelist-custom-nodes') + 1, 'ComfyUI-KJNodes')
        return result
    transport.server_command = command
    root.mkdir(parents=True)
    expected = {'core': verify_core_source(args.core), 'sources': transport.source_snapshot(),
        'mode': 'gpu', 'pilot_graphs': {'chunked_parity': graph},
        'probe_differences': ['KJ global Sage selector'],
        'runtime_options': {'reserve_vram_gib': 5, 'headroom_gib': 2}}
    transport.write_json(root / 'paths.json', probe_resource_config(args.core, project))
    transport.write_json(root / 'expected.json', expected)
    result = {'status': 'incomplete', 'human_quality_accepted': False}
    guard, monitor = ResourceGuard(), None
    server = transport.OwnedServer(root, args.port, False, 2)
    lease = args.core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError('Startup resource guard: ' + reason)
            server.start()
            monitor = transport.ContinuousGuard(reader, guard, root / 'resources.jsonl', server)
            cleanup.callback(monitor.close)
            monitor.start()
            transport.wait_ready(server, monitor.check)
            info = server.request('GET', '/object_info')
            transport.write_json(root / 'object-info.json', info)
            for node in graph.values():
                if node['class_type'] not in info:
                    raise RuntimeError('Missing node registration: ' + node['class_type'])
            started = time.perf_counter()
            history, timing = transport.execute_graph(server, graph, root / 'generation',
                monitor.check, timeout=2400)
            report = transport.preview_report(history, '17')
            if report['segment_count'] != 2 or report['total_model_calls_including_coarse_contract'] != 12:
                raise RuntimeError('Probe did not execute the declared two-window4+4 contract')
            media = audit_media(root)
            if (media['width'], media['height'], media['frames']) != (896,448,192):
                raise RuntimeError('Unexpected output media geometry')
            if transport.source_snapshot() != expected['sources']:
                raise RuntimeError('Runtime source changed during inference')
            result.update(status='mechanical_pass_human_pending', wall_seconds=time.perf_counter()-started,
                report=report, media=media, timing=timing)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(root / 'terminal.json', result)
        print(json.dumps({'status': result['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
