"""Explicit serial two-segment test: real interrupt after upscale, fresh-process resume.

No retry loop: exactly one intentional interrupted run, then one resume. No
runtime monkeypatch. Both use the same immutable graph, output and source set.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def request_interrupt(server):
    # Core's /interrupt returns HTTP200 with an empty body, not JSON.
    import requests
    server.assert_port_owner()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(server.url + '/interrupt', json={}, timeout=(1, 2))
        response.raise_for_status()


def stage_snapshot(output):
    return {path.relative_to(output).as_posix(): file_identity(path)['sha256']
            for stage in ('low_x0', 'high_input')
            for path in output.glob('**/dual_stages/segment_00000/*/' + stage + '-*')}


def assert_interrupted(folder, sent):
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    terminals = [event['type'] for event in events if event['type'] in
                 ('execution_success', 'execution_error', 'execution_interrupted')]
    if not sent or terminals != ['execution_interrupted']:
        raise RuntimeError('Probe did not end in the specifically requested Core interruption')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8208)
    parser.add_argument('--run-gpu', action='store_true', help='Required explicit execution switch')
    parser.add_argument('--reuse-cached-start', type=Path,
        help='Explicitly recover stages from the interrupted HTTP-empty-response controller failure; no runtime migration')
    args = parser.parse_args()
    if not args.run_gpu:
        raise ValueError('This real GPU probe requires --run-gpu')
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new worktree artifact directory')
    transport.CORE, transport.PROJECT = args.core.resolve(), project
    root.mkdir(parents=True)
    output = root / 'job-output'
    inherited = None
    if args.reuse_cached_start:
        prior = args.reuse_cached_start.resolve()
        if not prior.is_relative_to(project / 'artifacts'):
            raise ValueError('Previous evidence must be inside this worktree artifacts')
        prior_terminal = json.loads((prior / 'terminal.json').read_text(encoding='utf-8'))
        if (prior_terminal.get('error') != 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'
                or prior_terminal['attempts'][0]['server_stop']['owned_children_remaining']):
            raise ValueError('Only the recorded empty-response controller failure is recoverable here')
        inherited = json.loads((prior / 'expected.json').read_text(encoding='utf-8'))
        saved_sources = {k: v for k, v in inherited['sources'].items()
                         if k != 'tools/run_dual_model_continuation_probe.py'}
        if saved_sources != transport.source_snapshot():
            raise ValueError('Runtime or dependency sources changed; old stages cannot be reused')
        output = prior / 'job-output'
        if len(stage_snapshot(output)) != 4 or list(output.glob('**/dual_stages/segment_00000/*/high_output-*.json')):
            raise ValueError('Expected intact low/upscale stages and no completed high stage')
    else:
        output.mkdir()
    original_command = transport.server_command
    def command(*values):
        result = original_command(*values)
        result.insert(result.index('--whitelist-custom-nodes') + 1, 'ComfyUI-KJNodes')
        result[result.index('--output-directory') + 1] = str(output)
        return result
    transport.server_command = command
    graph = json.loads((project / 'artifacts/dual-workflows-cpu-v1/Relay.prompt.json').read_text(encoding='utf-8'))
    graph['8']['inputs'].update(total_duration_seconds=8., chain_id='dual_resume_' + root.name,
        base_seed=2609032101, filename_prefix='Dual_Continuation_Resume', resume_existing=True)
    graph['3']['inputs']['strength_model'] = .9
    graph['7']['inputs'].update(length=193,
        local_prompts='She speaks the sentence, then listens.\nShe slowly turns her head toward the door.',
        time_ranges='0-50\n50-100')
    for index, model in [('21', '2'), ('22', '3')]:
        graph[index] = {'class_type': 'PathchSageAttentionKJ', 'inputs': {
            'model': [model, 0], 'sage_attention': 'auto', 'allow_compile': False}}
    graph['8']['inputs'].update(model_pass1=['21', 0], model_pass2=['22', 0])
    if inherited:
        graph = inherited['pilot_graphs']['dual_resume']
    source = transport.source_snapshot()
    source['tools/run_dual_model_continuation_probe.py'] = file_identity(Path(__file__))['sha256']
    expected = {'core': verify_core_source(args.core), 'sources': source,
                'mode': 'gpu', 'pilot_graphs': {'dual_resume': graph}}
    transport.write_json(root / 'expected.json', expected)
    transport.write_json(root / 'output-location.json', {'output': str(output),
        'reused_failed_controller_root': str(args.reuse_cached_start) if inherited else None})
    result = {'status': 'incomplete', 'human_review': 'pending', 'attempts': []}
    before = None
    lease = args.core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    try:
        with SerialProbeLease(lease):
            for phase in ('interrupt_after_upscale', 'resume_fresh_process'):
                folder = root / phase
                folder.mkdir()
                transport.write_json(folder / 'paths.json', probe_resource_config(args.core, project))
                server = transport.OwnedServer(folder, args.port, False)
                guard = ResourceGuard()
                receipt = {'phase': phase, 'status': 'incomplete'}
                try:
                    with NvmlResourceReader() as reader, ExitStack() as cleanup:
                        reason = guard.observe(reader.sample(), startup=True)
                        if reason:
                            raise RuntimeError('Startup resource guard: ' + reason)
                        server.start()
                        monitor = transport.ContinuousGuard(reader, guard, folder / 'resources.jsonl', server)
                        cleanup.callback(monitor.close)
                        monitor.start()
                        transport.wait_ready(server, monitor.check)
                        transport.execute_graph(server, {
                            '1': {'class_type': 'T8ProgressiveEnvironmentAudit', 'inputs': {'expected_json': json.dumps(expected)}},
                            '2': {'class_type': 'PreviewAny', 'inputs': {'source': ['1', 0]}}}, folder / 'environment', monitor.check)
                        sent = False
                        def check():
                            nonlocal sent
                            monitor.check()
                            if phase == 'interrupt_after_upscale' and not sent:
                                ready = list(output.glob('**/dual_stages/segment_00000/*/high_input-*.json'))
                                sampling_entered = not inherited or '0/4' in (folder / 'server.stderr.log').read_text(encoding='utf-8', errors='replace')
                                if ready and sampling_entered:
                                    request_interrupt(server)
                                    sent = True
                                    transport.write_json(folder / 'interrupt-request.json', {'stage_receipt': str(ready[0]), 'sent': True})
                        try:
                            transport.execute_graph(server, graph, folder / 'generation', check, timeout=3000)
                        except RuntimeError:
                            if phase != 'interrupt_after_upscale':
                                raise
                            assert_interrupted(folder / 'generation', sent)
                            before = stage_snapshot(output)
                            if len(before) != 4 or list(output.glob('**/dual_stages/segment_00000/*/high_output-*.json')):
                                raise RuntimeError('Interruption did not preserve exactly low and prepared input stages')
                            transport.write_json(root / 'before-resume.json', before)
                            receipt['status'] = 'intentional_core_interrupt_after_saved_low_and_upscale'
                        else:
                            if phase == 'interrupt_after_upscale':
                                raise RuntimeError('Expected interruption was missed')
                            if before != stage_snapshot(output):
                                raise RuntimeError('Resume rewrote first-pass/upscale evidence')
                            receipt['status'] = 'generation_completed_independent_audit_pending'
                        if transport.source_snapshot() != {k: v for k, v in source.items() if k != 'tools/run_dual_model_continuation_probe.py'}:
                            raise RuntimeError('Sources changed during continuation test')
                finally:
                    server.stop()
                    receipt.update(resources=guard.report(), server_stop=server.stop_receipt)
                    result['attempts'].append(receipt)
                    transport.write_json(folder / 'terminal.json', receipt)
            result['status'] = 'two_segment_resume_completed_independent_audit_and_human_pending'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        transport.write_json(root / 'terminal.json', result)
        print(json.dumps({'status': result['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
