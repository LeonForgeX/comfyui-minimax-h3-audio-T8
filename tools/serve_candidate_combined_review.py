"""Loopback-only allowlisted candidate review; never serve source or audit folders."""
import argparse
import json
from pathlib import Path
import re

from aiohttp import web

from tools.build_candidate_combined_review import digest


def create_app(root, artifacts):
    root, artifacts = Path(root).resolve(strict=True), Path(artifacts).resolve(strict=True)
    if root.name != 'public' or not root.is_relative_to(artifacts):
        raise ValueError('Only a dedicated candidate public directory')
    delivery = json.loads((root.parent / 'delivery.json').read_text(encoding='utf-8'))
    paths = {'review.html': (root / 'review.html').resolve(strict=True)}
    for group in delivery['groups']:
        for clip in group['clips']:
            if not re.fullmatch(r'clip-[0-9]{2}\.mp4', clip['url']):
                raise ValueError('Invalid public asset name')
            path = (root / clip['url']).resolve(strict=True)
            if path.parent != root or digest(path) != clip['sha256']:
                raise ValueError('Public media escaped or changed')
            paths[clip['url']] = path
    if not 2 <= len(paths) <= 65 or any(not p.is_file() or p.parent != root for p in paths.values()):
        raise ValueError('Missing or unsafe public review files')

    async def serve(request):
        name = request.match_info.get('name', 'review.html')
        if name not in paths:
            raise web.HTTPNotFound()
        return web.FileResponse(paths[name], headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

    app = web.Application()
    app.router.add_get('/', serve)
    app.router.add_get('/{name}', serve)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError('Invalid review port')
    artifacts = Path(__file__).resolve().parents[1] / 'artifacts'
    web.run_app(create_app(args.root, artifacts), host='127.0.0.1', port=args.port, print=None)
