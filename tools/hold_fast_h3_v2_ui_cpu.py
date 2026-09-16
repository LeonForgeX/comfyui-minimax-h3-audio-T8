"""Task-owned CPU-only native frontend; no generation or user canvas changes."""
import argparse
from pathlib import Path

import run_progressive_pilot as transport
from vdn_probe_environment import probe_resource_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8209)
    parser.add_argument('--with-sol', action='store_true', help='Registration only, CUDA remains disabled')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    core, root = args.core.resolve(strict=True), args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts') or root == project / 'artifacts':
        raise ValueError('New task-owned output directory required')
    root.mkdir()
    transport.CORE, transport.PROJECT = core, project
    if args.with_sol:
        original_command = transport.server_command
        def command(*values):
            result = original_command(*values)
            result.insert(result.index('--whitelist-custom-nodes') + 1, 'ComfyUI-sol-attn')
            return result
        transport.server_command = command
    transport.write_json(root / 'paths.json', probe_resource_config(core, project))
    before = transport.source_snapshot()
    server = transport.OwnedServer(root, args.port, True)
    result = dict(status='incomplete', gpu_generation=False, user_frontend_modified=False)
    try:
        server.start()
        transport.wait_ready(server, lambda: None)
        info = server.request('GET', '/object_info')
        transport.write_json(root / 'object-info.json', info)
        assert all(n in info for n in ('MiniMaxH3FastH3V2SetupEXPT8',
            'MiniMaxH3FastH3V2RuntimeAuditEXPT8', 'MiniMaxH3FastH3V2DualModelLongVideoEXPT8'))
        print(f'CPU native UI ready {server.url}; Enter stops only owned process', flush=True)
        input()
        result['source_unchanged'] = before == transport.source_snapshot()
        assert result['source_unchanged']
        result['status'] = 'isolated_cpu_server_closed_browser_proof_separate'
    finally:
        server.stop()
        result['owned_server_stop'] = server.stop_receipt
        transport.write_json(root / 'server-terminal.json', result)


if __name__ == '__main__':
    main()
