"""Two serial uncached 4+4 jobs per backend; process-cold then process-warm.

Both use the public node's normal stage offload and complete content checks.
Warm means same process/kernel caches, NOT resident model or OS-cold disk.
No hidden warmup, changed settings, checkpoint reuse or automatic retry.
"""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dual_model_pilot as pilot  # noqa: E402


def warm_graph(graph):
    result = deepcopy(graph)
    settings = result['8']['inputs']
    if (settings['total_duration_seconds'] != 3 or settings['coarse_steps'] != 4
            or settings['refine_steps'] != 4 or settings['prompt_relay_mode'] != 'disabled'
            or settings['eav_mode'] != 'disabled' or '28' in graph or '23' in graph):
        raise ValueError('Benchmark requires the fixed plain same-base 3s4+4 recipe')
    settings['chain_id'] += '_process_warm'
    return result


def main():
    transport = pilot.transport
    original_execute = transport.execute_graph
    original_snapshot = transport.source_snapshot
    def snapshot():
        return {**original_snapshot(), 'tools/run_dual_backend_benchmark.py':
                pilot.file_identity(Path(__file__))['sha256']}
    def execute(server, graph, folder, check, timeout=1800):
        if folder.name != 'generation':
            return original_execute(server, graph, folder, check, timeout)
        second = warm_graph(graph)  # Validate before any measured work.
        first_history, first_timing = original_execute(server, graph, folder, check, timeout)
        check()
        _history, second_timing = original_execute(server, second,
            folder.parent / 'generation-process-warm', check, timeout)
        transport.write_json(folder.parent / 'benchmark-pair.json', {
            'status': 'two_uncached_jobs_completed_independent_audit_pending',
            'cold': {'chain_id': graph['8']['inputs']['chain_id'], 'timing': first_timing},
            'process_warm': {'chain_id': second['8']['inputs']['chain_id'], 'timing': second_timing},
            'scope': 'same-process second invocation; model-stage offload retained; no OS disk-cache purge',
            'only_recipe_change': 'chain_id prevents accepted-stage cache reuse'})
        return first_history, first_timing
    transport.source_snapshot = snapshot
    transport.execute_graph = execute
    try:
        pilot.main()
    finally:
        transport.source_snapshot = original_snapshot
        transport.execute_graph = original_execute


if __name__ == '__main__':
    main()
