"""Fixed native8 EMA-B versus trained V2 cold/hot recipes; never user UI.

Default prints the fixed plan only. --execute owns two serial processes, each
with cold then genuinely uncached hot generation, and preserves every receipt.
Not identical models/schedules/backends or a perceptual non-inferiority trial.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from run_fast_h3_v2_probe import (  # noqa: E402
    audit_v2_media, build_graph as v2_graph, isolated_backend_command, model_identity,
)
from run_h3_memory_node_probe import build_graph as memory_graph  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    GuardPolicy, MIB, NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity,
)
from progressive_memory_metrics import validate_completed_interval  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
PROFILES = ('production_ema_b_native8', 'trained_v2_dmd8')


def paired_graphs():
    old, _ = memory_graph(head_chunks=1, ffn_chunks=1)
    old['9']['inputs'].update(width=832, height=480, length=73)
    old['10']['inputs']['steps'] = 8
    old['10']['inputs']['sampler_name'] = 'euler'
    old.pop('11')
    old.pop('12')
    old['13'] = dict(class_type='T8ProgressiveNativeBaseline', inputs=dict(
        model=['10', 0], positive=['9', 0], av_latent=['9', 1],
        sampler=['10', 1], sigmas=['10', 2], seed=2609032101))
    old['22'] = dict(class_type='PreviewAny', inputs=dict(source=['13', 1]))
    new, _ = v2_graph(frames=73, width=832, height=480, min_tokens=0)
    for name, graph in zip(PROFILES, (old, new)):
        graph['13']['class_type'] = 'T8FastH3V2ThermalSamplerProbe'
        graph['13']['inputs']['thermal_profile'] = name
        graph['16']['inputs']['filename_prefix'] = 'MiniMaxH3/Thermal/' + name
    assert old['9'] == new['9']
    assert all(old[k] == new[k] for k in ('6', '7', '8', '15'))
    return dict(zip(PROFILES, (old, new)))


def stage_times(timing, network):
    if timing.get('complete_uncached_graph') is not True or timing.get('graph_cached_nodes'):
        raise ValueError('Cold/hot measurement requires a complete uncached graph')
    completed = network.get('completed_network_forwards', network.get('actual_network_forwards'))
    if completed != 8:
        raise ValueError('Every cold and hot submission must really execute eight forwards')
    rows = {r['node']: r['seconds'] for r in timing['node_intervals']}
    if not {'6', '9', '13', '14', '16'} <= rows.keys():
        raise ValueError('Missing encoding/sampling/decoding/save timing boundaries')
    return dict(graph_wall_seconds=timing['elapsed_seconds'],
        qwen_loader_seconds=rows['6'], conditioning_encode_seconds=rows['9'],
        sampler_node_seconds=rows['13'], observed_sampler_seconds=network['sampler_seconds'],
        joint_vae_decode_seconds=rows['14'], h264_save_seconds=rows['16'],
        actual_forwards=completed, timing_scope=timing['timing_scope'])


def resource_interval(rows, start, end):
    samples = [r for r in rows if start <= r['monotonic'] <= end]
    if len(samples) < 2:
        raise ValueError('Missing bounded resource observations')
    return dict(samples=len(samples),
        maximum_observed_whole_card_bytes=max(r['gpu_used_bytes'] for r in samples),
        maximum_observed_process_rss_bytes=max(r['owned_process_rss_bytes'] for r in samples),
        minimum_observed_system_ram_available_bytes=min(r['ram_available_bytes'] for r in samples),
        scope='periodic samples, not exact peaks; whole card includes unrelated activity')


def read_resource_rows(path):
    # The monitor is still appending. Ignore only a final uncommitted line;
    # malformed complete records must still fail closed.
    lines = Path(path).read_text(encoding='utf8').splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith('\n')]


class ProcessResourceReader:
    def __init__(self, reader, server):
        self.reader, self.server = reader, server

    def sample(self):
        import psutil
        row = self.reader.sample()
        process = self.server.process
        row['owned_process_rss_bytes'] = psutil.Process(process.pid).memory_info().rss if process else 0
        return row


def run_profile(core, root, port, name, graph, assets):
    root.mkdir()
    transport.write_json(root / 'paths.json', probe_resource_config(core, PROJECT))
    sources = transport.source_snapshot()
    for path in (Path(__file__), PROJECT / 'tools/fast_h3_v2_thermal_audit.py'):
        sources[path.relative_to(PROJECT).as_posix()] = file_identity(path)['sha256']
    expected = dict(core=verify_core_source(core), sources=sources, mode='gpu', assets=assets)
    transport.write_json(root / 'expected.json', expected)
    guard = ResourceGuard(GuardPolicy(startup_free_gpu_bytes=512*MIB, startup_free_ram_bytes=4096*MIB))
    server = transport.OwnedServer(root, port, False)
    result = dict(status='incomplete', profile=name, fresh_process=True, runs=[], published=False)
    monitor = None
    try:
        with NvmlResourceReader() as reader, ExitStack() as cleanup:
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError('Measured resource protection: ' + reason)
            server.start()
            monitor = transport.ContinuousGuard(ProcessResourceReader(reader, server), guard, root / 'resources.jsonl', server)
            cleanup.callback(monitor.close)
            monitor.start()
            transport.wait_ready(server, monitor.check)
            transport.write_json(root / 'object-info.json', server.request('GET', '/object_info'))
            h, _ = transport.execute_graph(server, {
                '90': dict(class_type='T8ProgressiveEnvironmentAudit', inputs=dict(expected_json=json.dumps(expected))),
                '91': dict(class_type='PreviewAny', inputs=dict(source=['90', 0]))}, root / 'environment', monitor.check)
            result['environment'] = transport.preview_report(h, '91')
            for thermal in ('cold', 'hot'):
                run = root / thermal
                run.mkdir()
                transport.allocator_action(server, run, 'begin', 'cuda', monitor.check)
                start = time.perf_counter()
                h, timing = transport.execute_graph(server, graph, run / 'generation', monitor.check, timeout=1200)
                end = time.perf_counter()
                network = transport.preview_report(h, '22')
                times = stage_times(timing, network)
                dispatch = network['backend_calls']
                if dispatch.get('status') != 'actual_thermal_backend_calls_completed' or dispatch.get('profile') != name:
                    raise RuntimeError('Missing actual completed backend observation in this interval')
                if name == PROFILES[1]:
                    runtime_dispatch = transport.preview_report(h, '20')
                    if not runtime_dispatch['actual_vsa_dispatched'] or runtime_dispatch['counts'].get('vsa') != 400:
                        raise RuntimeError('V2 must actually dispatch400 VSA calls in each uncached run')
                    dispatch = {**dispatch, 'runtime': runtime_dispatch}
                allocator = transport.allocator_action(server, run, 'finish', 'cuda', monitor.check)
                validate_completed_interval(allocator, run_id=run.name, pid=server.process.pid,
                    device_type='cuda', gpu_uuid=guard.report()['gpu_uuid'])
                transport.write_json(run / 'allocator-memory.json', allocator)
                media = h['outputs']['16']['images'][0]
                path = root / 'output' / media['subfolder'] / media['filename']
                # Audit this exact output without confusing a second warm file.
                audit_root = run / 'media-audit'
                (audit_root / 'output').mkdir(parents=True)
                import shutil
                shutil.copyfile(path, audit_root / 'output' / path.name)
                audit = audit_v2_media(audit_root, width=832, height=480, frames=73)
                rows = read_resource_rows(root / 'resources.jsonl')
                row = dict(thermal=thermal, server_pid=server.process.pid, **times, network=network,
                    dispatch=dispatch, memory=resource_interval(rows, start, end), media=audit,
                    thermal_scope='process/model cold, disk cache not cleared' if thermal == 'cold'
                        else 'same process/OS and compilation caches; cache-none rebuilds graph loaders, not a promised persistent MODEL-object cache')
                transport.write_json(run / 'receipt.json', row)
                result['runs'].append(row)
                print(json.dumps(dict(profile=name, thermal=thermal, graph_seconds=times['graph_wall_seconds'])), flush=True)
            current = transport.source_snapshot()
            for path in (Path(__file__), PROJECT / 'tools/fast_h3_v2_thermal_audit.py'):
                current[path.relative_to(PROJECT).as_posix()] = file_identity(path)['sha256']
            if current != sources or verify_core_source(core) != expected['core']:
                raise RuntimeError('Source changed during benchmark')
            if any(file_identity(Path(a['path']))['sha256'] != a['sha256'] for a in assets):
                raise RuntimeError('A benchmark asset changed')
            result.update(status='two_uncached_thermal_runs_mechanically_pass', source_and_assets_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(owned_server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(root / 'terminal.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8209)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    graphs = paired_graphs()
    if not args.execute:
        print(json.dumps(dict(status='fixed_plan_only_no_inference', graphs=graphs,
            common='832x480x73/24fps, identical prompt/seed/CFG1/Qwen/VAEs',
            difference='EMA-B native8 shift12/3 KJ Sage versus full trained V2 DMD8 shift10/3 learned VSA min_tokens0',
            published=False), ensure_ascii=False))
        return
    core, root = args.core.resolve(strict=True), args.root.resolve()
    if root.exists() or root == PROJECT / 'artifacts' or not root.is_relative_to(PROJECT / 'artifacts'):
        raise ValueError('New task-owned benchmark output required')
    transport.CORE, transport.PROJECT = core, PROJECT
    original = transport.server_command
    # Same encoder/global attention, same reserve5/headroom2 for both processes.
    transport.server_command = lambda *v: isolated_backend_command(original(*v), 'kj_sage', 2)
    root.mkdir()
    assets = [model_identity(core)]
    for folder, name in [('diffusion_models', graphs[PROFILES[0]]['1']['inputs']['unet_name']),
                         ('loras', graphs[PROFILES[0]]['2']['inputs']['lora_name']),
                         ('text_encoders', graphs[PROFILES[0]]['6']['inputs']['clip_name']),
                         ('vae', graphs[PROFILES[0]]['7']['inputs']['vae_name']),
                         ('vae', graphs[PROFILES[0]]['8']['inputs']['vae_name'])]:
        assets.append(file_identity(core / 'models' / folder / name))
    transport.write_json(root / 'assets.json', assets)
    lease = core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    with SerialProbeLease(lease):
        results = [run_profile(core, root / name, args.port, name, graph, assets) for name, graph in graphs.items()]
    transport.write_json(root / 'terminal.json', dict(status='four_fixed_recipe_cold_hot_runs_mechanically_qualified',
        profiles=results, no_perceptual_quality_inferred=True, no_universal_speed_or_memory_claim=True,
        comparative_performance_qualified=True,
        performance_scope='one observed recipe pair, consistent observer overhead, no identical-model or quality equivalence claim',
        published=False))


if __name__ == '__main__':
    main()
