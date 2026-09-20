"""Create a CPU-only browser replay workflow in an owned probe user directory."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.api_to_frontend_workflow import convert  # noqa: E402
from tools.build_dual_model_workflows import defaults  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve(strict=True)
    if not root.is_relative_to(project / 'artifacts/five-track-development-20260918'):
        raise ValueError('Owned probe directory required')
    ready = json.loads((root / 'ready.json').read_text(encoding='utf8'))
    assert ready['cpu_only'] and not ready['user_frontend_changed']
    with urllib.request.urlopen(ready['url'] + '/object_info', timeout=15) as response:
        info = json.load(response)
    kind = 'MiniMaxH3TAEH3SamplingPreviewEXPT8'
    graph = {
        '1': {'class_type': kind, 'inputs': {**defaults(info[kind]), 'checkpoint': 'taeh3.safetensors'}},
        '2': {'class_type': 'T8OwnedPreviewCancelReplay', 'inputs': {
            'preview_node_id': '1', 'fixture_path': str(root / 'preview-fixture.json'), 'seconds': 45}},
        '3': {'class_type': 'PreviewAny', 'inputs': {'source': ['2', 0]}},
    }
    # MODEL observer intentionally has no executable output consumer. Only
    # the CPU replay-to-text branch is queued by native Run.
    workflow = convert(graph, info, 'Owned CPU protocol replay — not GPU generation')
    workflow['nodes'][0]['pos'] = [0, 0]
    workflow['nodes'][0]['size'] = [600, 1000]
    workflow['nodes'][1]['pos'] = [650, 0]
    workflow['nodes'][2]['pos'] = [650, 350]
    if args.suffix not in {'', '_retry'}:
        raise ValueError('Unknown isolated test revision')
    if args.suffix:
        graph['2']['inputs']['seconds'] = 60
        workflow = convert(graph, info, 'Owned CPU cancellation retry — not GPU generation')
        workflow['nodes'][0]['size'] = [600, 1000]
    path = root / ('user/default/workflows/OWNED_CPU_PREVIEW_CANCEL' + args.suffix + '.json')
    assert not path.exists()
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding='utf8')
    diagnostic = convert({'1': {'class_type': 'MiniMaxH3NodeSourceDiagnosticT8',
        'inputs': {'node_id': 'MiniMaxH3TAEH3SamplingPreviewEXPT8'}}}, info, 'Owned P1 diagnostic recovery')
    diagnostic['nodes'][0]['size'] = [800, 650]
    second = root / ('user/default/workflows/OWNED_P1_RECOVERY' + args.suffix + '.json')
    assert not second.exists()
    second.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'paths': [str(path), str(second)], 'qualification': 'CPU browser protocol only'}))


if __name__ == '__main__':
    main()
