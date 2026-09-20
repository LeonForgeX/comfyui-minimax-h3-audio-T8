"""Read-only audit of the isolated live Core cancellation/recovery history."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cancelled-prompt', required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve(strict=True)
    if not root.is_relative_to(project / 'artifacts/five-track-development-20260918'):
        raise ValueError('Owned probe root required')
    ready = json.loads((root / 'ready.json').read_text(encoding='utf8'))
    assert ready['cpu_only'] and ready['url'] == 'http://127.0.0.1:8213'
    with urllib.request.urlopen(ready['url'] + '/history', timeout=15) as response:
        history = json.load(response)
    cancelled = history[args.cancelled_prompt]
    assert cancelled['prompt'][2]['2']['class_type'] == 'T8OwnedPreviewCancelReplay'
    assert any(event == 'execution_interrupted' for event, _ in cancelled['status']['messages'])
    assert '3' not in cancelled['outputs'], 'Cancelled replay must not produce the text output'
    recovery_id, recovery = next((key, value) for key, value in reversed(list(history.items()))
        if any(node['class_type'] == 'MiniMaxH3NodeSourceDiagnosticT8' for node in value['prompt'][2].values()))
    assert recovery['status']['status_str'] == 'success'
    assert recovery['outputs']['1']['text'] and '实际模块' in recovery['outputs']['1']['text'][0]
    with urllib.request.urlopen(ready['url'] + '/queue', timeout=15) as response:
        queue = json.load(response)
    assert not queue['queue_running'] and not queue['queue_pending']
    hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in ('web/taeh3_preview.js', 'web/readable_audio.js')}
    result = dict(status='live_browser_private_CPU_protocol_cancel_and_next_request_recovery_pass',
        qualification='CUA browser loaded real source extension; replayed actual GPU frames, NOT live GPU sampling',
        cancelled_prompt_id=args.cancelled_prompt, cancelled_history=cancelled,
        recovery_prompt_id=recovery_id, recovery_history=recovery, queue_empty=True,
        owned_core_pid=ready['pid'], source_hashes=hashes, published=False,
        user_frontend_changed=False, live_native_request_scoped_cancel=True)
    path = root / 'live-ui-audit.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({key: value for key, value in result.items() if not key.endswith('_history')}))


if __name__ == '__main__':
    main()
