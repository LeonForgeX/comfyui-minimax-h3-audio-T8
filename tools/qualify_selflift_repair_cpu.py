"""Bind repaired SelfLift interfaces to frozen CPU/Core evidence; no inference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

from audit_patch_stack_policy import inventory
from qualify_patch_stack_policy import junit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--tests', type=Path, nargs='+', required=True)
    parser.add_argument('--core-report', type=Path, required=True)
    parser.add_argument('--selflift-core-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(project / 'artifacts') or destination.exists():
        raise ValueError('Choose a new JSON within project artifacts')
    frozen = json.loads(args.inventory.read_text(encoding='utf-8'))
    current = inventory(project)
    if frozen['sources'] != current['sources']:
        raise RuntimeError('Runtime changed after freeze; rerun final qualification')
    tests = [junit(path) for path in args.tests]
    if any(test['failures_or_errors'] or not test['cases'] for test in tests):
        raise RuntimeError('Final affected tests are incomplete or have failures')
    ids = Counter(name for test in tests for name in test['case_ids'])
    if any(count != 1 for count in ids.values()):
        raise RuntimeError('Overlapping final tests must not be double-counted')
    worker_skips = []
    for path in args.tests:
        for case in ET.parse(path).getroot().iter('testcase'):
            skipped = case.find('skipped')
            if skipped is not None:
                if case.get('name') not in {'test_fresh_process_worker', 'test_first_relay_worker'}:
                    raise RuntimeError('Unexpected skipped affected regression: ' + case.get('name', ''))
                worker_skips.append({'test': case.get('classname', '') + '::' + case.get('name', ''),
                                     'reason': skipped.get('message', '')})
    original = ['test_same_latents_do_not_reuse_checkpoint_after_producer_change',
                'test_fabricated_producer_labels_fail_before_sampling'] + [
                f'test_progressive_consumes_public_model_config_per_phase[{phase}]'
                for phase in ('both', 'low', 'high')]
    for name in original:
        if not any(identifier.endswith('::' + name) for identifier in ids):
            raise RuntimeError('Missing original failed interface regression: ' + name)
    seam_modules = ('test_dual_picture_context', 'test_dual_high_video_prefix',
                    'test_long_video_dual_model_runner', 'test_long_video_dual_stage_cache',
                    'test_nodes_long_video_dual_model')
    if any(not any(module in identifier for identifier in ids) for module in seam_modules):
        raise RuntimeError('Accepted dual-model seam regression modules are missing')
    core = json.loads(args.core_report.read_text(encoding='utf-8'))
    features = json.loads((project / 'features.json').read_text(encoding='utf-8'))
    selflift = json.loads(args.selflift_core_report.read_text(encoding='utf-8'))
    if (not core['validation'][0] or core['registered_node_ids'] != features['nodes']
            or len(selflift['graphs']) != 3 or selflift['registered_nodes'] != len(features['nodes'])):
        raise RuntimeError('Actual Core schemas or accepted SelfLift graphs differ')
    for evidence in (core, selflift):
        if evidence['cuda_initialized'] or evidence['queued'] or evidence['browser_used']:
            raise RuntimeError('This report requires CPU-only/no-queue qualification')
    for graph in selflift['graphs']:
        if (not graph['validation'][0] or hashlib.sha256(Path(graph['workflow']).read_bytes()).hexdigest()
                != graph['sha256']):
            raise RuntimeError('Saved accepted SelfLift workflow changed after validation')
    config = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8-sig'))
    if any('roadmap' in name.casefold() for name in config['tool']['comfy']['includes']):
        raise RuntimeError('Private ROADMAP must not appear in package includes')
    previous_path = project / 'artifacts/development/patch-stack-advisory-20260917'
    previous = json.loads((previous_path / 'guard-inventory-final-v4.json').read_text(encoding='utf-8'))
    changed = sorted(name for name, sha in current['sources'].items() if previous['sources'].get(name) != sha)
    expected_changes = sorted('h3_t8/' + name + '.py' for name in (
        'progressive_sampling_runtime', 'progressive_sampling_composed', 'progressive_eav',
        'progressive_job', 'enhance_a_video_advanced', 'nodes_progressive_sampling'))
    if changed != expected_changes or set(previous['sources']) - set(current['sources']):
        raise RuntimeError('Runtime changes exceed the six-module interface repair scope')
    formal_path = project / ('examples/workflows/13-latent-upscale/'
                             '2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json')
    formal_sha = hashlib.sha256(formal_path.read_bytes()).hexdigest()
    previous_evidence = json.loads((previous_path / 'verification.json').read_text(encoding='utf-8'))
    if formal_sha != previous_evidence['workflow_sha256']:
        raise RuntimeError('Accepted formal chunked workflow changed during the interface repair')
    report = {'schema': 't8.h3.selflift_interface_repair.local_cpu/v1',
        'status': 'original_five_and_affected_CPU_Core_pass_not_GPU_quality_or_full_repository',
        'runtime_source_count': len(current['sources']), 'sources': current['sources'],
        'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        'tests': tests, 'passed': sum(test['passed'] for test in tests),
        'skipped': sum(test['skipped'] for test in tests), 'original_five_repaired': original,
        'worker_only_skips': worker_skips, 'changed_runtime_sources': changed,
        'actual_core': core, 'saved_selflift_graphs': selflift,
        'original_sol_source_unchanged': True, 'roadmap_not_includes': True,
        'accepted_formal_workflow_unchanged': {'path': str(formal_path), 'sha256': formal_sha},
        'boundaries': ['No new pretrained whole-model GPU generation or human quality review.',
                       'No accepted seam/default recipe changes or paused research restoration.',
                       'No Git operations, publication, user server restart or queue submission.',
                       'Historical full-repository fixture/doc issues are not claimed fixed.']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'passed', 'skipped', 'runtime_source_count')}))


if __name__ == '__main__':
    main()
