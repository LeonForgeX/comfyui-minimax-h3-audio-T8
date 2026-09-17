"""Bind local patch-policy evidence to frozen sources, without Git or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

from audit_patch_stack_policy import inventory


def junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failed = []
    skipped = 0
    for case in cases:
        skipped += case.find('skipped') is not None
        for tag in ('failure', 'error'):
            error = case.find(tag)
            if error is not None:
                failed.append({'test': case.get('classname', '') + '::' + case.get('name', ''),
                               'kind': tag, 'message': error.get('message', '')})
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cases': len(cases), 'passed': len(cases) - skipped - len(failed),
            'skipped': skipped, 'failures_or_errors': failed,
            'case_ids': [case.get('classname', '') + '::' + case.get('name', '') for case in cases]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--core-report', type=Path, required=True)
    parser.add_argument('--tests', type=Path, nargs='+', required=True)
    parser.add_argument('--separate-interface-gaps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(project / 'artifacts') or destination.exists():
        raise ValueError('Choose a new JSON path within project artifacts')
    before = json.loads(args.inventory.read_text(encoding='utf-8'))
    current = inventory(project)
    if before['sources'] != current['sources']:
        raise RuntimeError('Runtime source changed after final freeze; rerun affected qualification')
    core = json.loads(args.core_report.read_text(encoding='utf-8'))
    features = json.loads((project / 'features.json').read_text(encoding='utf-8'))
    if (not core['validation'][0] or core['registered_node_ids'] != features['nodes']
            or core['cuda_initialized'] or core['queued'] or core['browser_used']):
        raise RuntimeError('Actual Core/schema/workflow CPU qualification differs')
    tests = [junit(path) for path in args.tests]
    if any(result['failures_or_errors'] or not result['cases'] for result in tests):
        raise RuntimeError('Affected final tests are incomplete or have failures')
    repetitions = Counter(name for result in tests for name in result['case_ids'])
    if any(count != 1 for count in repetitions.values()):
        raise RuntimeError('Final test batches overlap; do not double-count qualification')
    config = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8-sig'))
    includes = config['tool']['comfy']['includes']
    if any('roadmap' in item.casefold() for item in includes):
        raise RuntimeError('Private ROADMAP must not appear in package includes')
    gaps = junit(args.separate_interface_gaps)
    if not gaps['failures_or_errors']:
        raise RuntimeError('Expected separate unresolved interfaces were not reproduced; recheck documentation')
    report = {
        'schema': 't8.h3.patch_stack_policy.local_qualification/v1',
        'status': 'affected_policy_qualification_pass_not_full_repository_or_GPU_quality',
        'runtime_sources_frozen': len(before['sources']),
        'inventory': str(args.inventory.resolve()),
        'guard_review_candidates': len(current['guard_candidates']),
        'advisories': len(current['advisories']),
        'affected_tests': tests, 'affected_passed': sum(result['passed'] for result in tests),
        'actual_core_report': str(args.core_report.resolve()),
        'registered_nodes': core['registered_nodes'], 'serialization': core['serialization'],
        'registration_only_sol_unchanged': True,
        'workflow_sha256': hashlib.sha256((project / 'examples/workflows/13-latent-upscale/2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json').read_bytes()).hexdigest(),
        'separate_unresolved_interfaces': gaps,
        'package_includes_do_not_name_roadmap': True,
        'boundaries': [
            'Static guard candidates are not a universal behavioral compatibility proof.',
            'Full-v1 collection failed on missing local research fixture; full-v2 was stopped as obsolete at about43%.',
            'Neither unfinished full run nor separate interface failures count as passes.',
            'No new whole-model GPU/video/quality/speed qualification or publication.',
            'Git commands are unavailable on this host; no Git/index-clean claim.',
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'affected_passed': report['affected_passed'],
                      'registered_nodes': report['registered_nodes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
