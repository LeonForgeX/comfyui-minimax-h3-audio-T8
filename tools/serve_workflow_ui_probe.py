"""Time-bounded isolated CPU-only UI, with disposable copies of four workflows."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8209)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts') or not 1 <= args.seconds <= 3600:
        raise ValueError('Use a new bounded worktree-artifact probe')
    root.mkdir(parents=True)
    transport.CORE, transport.PROJECT = args.core.resolve(), project
    transport.write_json(root / 'paths.json', probe_resource_config(args.core, project))
    server = transport.OwnedServer(root, args.port, True)
    original = transport.server_command
    def command(*values):
        result = original(*values)
        result.insert(result.index('--whitelist-custom-nodes') + 1, 'ComfyUI-KJNodes')
        return result
    transport.server_command = command
    try:
        server.start()
        destination = root / 'user/default/workflows'
        destination.mkdir(parents=True)
        copies = []
        for folder, pattern in [('28-progressive-sampling', '*.json'), ('04-long-video', '2026-09-11*.json')]:
            for source in sorted((project / 'examples/workflows' / folder).glob(pattern)):
                target = destination / source.name
                shutil.copyfile(source, target)
                copies.append({'source': str(source), 'copy': str(target)})
        transport.wait_ready(server, lambda: None)
        transport.write_json(root / 'ready.json', {'url': server.url, 'cpu_only': True,
            'pid': server.process.pid, 'copies': copies, 'deadline_seconds': args.seconds})
        print(json.dumps({'url': server.url, 'status': 'ready_cpu_only', 'copies': len(copies)}), flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and server.process.poll() is None and not (root / 'STOP').exists():
            time.sleep(.5)
    finally:
        server.stop()
        transport.write_json(root / 'terminal.json', {'server_stop': server.stop_receipt,
            'qualification': 'server lifecycle only; UI verification separate'})


if __name__ == '__main__':
    main()
