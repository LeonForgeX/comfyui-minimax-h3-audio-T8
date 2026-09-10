"""File/hash/loopback HTTP audit of the aggregate page, without browser actions."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.serve_progressive_review_hub import create_app  # noqa: E402


class Players(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos, self.canvases, self.iframes = [], 0, 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'video':
            if 'autoplay' in attrs:
                raise ValueError('No automatic review playback permitted')
            self.videos.append(attrs['src'])
        if tag == 'canvas':
            self.canvases += 1
        if tag == 'iframe':
            self.iframes += 1


async def audit(root):
    from aiohttp import ClientSession, ClientTimeout, web
    root = Path(root).resolve(strict=True)
    public = root/'public'
    receipt = json.loads((root/'combined-private-receipt.json').read_text(encoding='utf8'))
    html = (public/'review.html').read_bytes()
    if hashlib.sha256(html).hexdigest() != receipt['review_html_sha256']:
        raise ValueError('Page changed after build')
    parser = Players()
    parser.feed(html.decode('utf8'))
    expected = {f'{section["id"]}-{side}.mp4': section['mapping'][side]['sha256']
                for section in receipt['sections'] for side in ('A', 'B')}
    if len(parser.videos) != len(expected) or set(parser.videos) != set(expected) or parser.canvases != len(expected) or parser.iframes:
        raise ValueError('Every pair must have inline original videos and local crops')
    for name, digest in expected.items():
        if hashlib.sha256((public/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Review media differs from qualified original')
    runner = web.AppRunner(create_app(public, research=root.parent))
    await runner.setup()
    try:
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        base = f'http://127.0.0.1:{runner.addresses[0][1]}'
        async with ClientSession(timeout=ClientTimeout(total=20)) as client:
            for name in expected:
                path = public/name
                async with client.head(base+'/'+name) as response:
                    if response.status != 200 or int(response.headers['Content-Length']) != path.stat().st_size:
                        raise ValueError('Video HEAD/size failed')
                async with client.get(base+'/'+name, headers={'Range': 'bytes=0-63'}) as response:
                    with path.open('rb') as video:
                        first = video.read(64)
                    if response.status != 206 or await response.read() != first:
                        raise ValueError('Original video byte-range delivery failed')
            for name in ('private-receipt.json', 'combined-private-receipt.json', 'source.json', 'missing.mp4'):
                async with client.get(base+'/'+name) as response:
                    if response.status != 404:
                        raise ValueError('Non-public path leaked')
            async with client.get(base+'/') as response:
                if response.status != 200 or await response.read() != html:
                    raise ValueError('Homepage delivery failed')
    finally:
        await runner.cleanup()
    return {'status': 'original_media_inline_hash_and_HTTP_range_pass', 'pairs': len(expected)//2,
            'videos': len(expected), 'review_id': receipt['review_id'], 'browser_opened': False,
            'browser_playback_qualified': False, 'human_qualified': False, 'server_stopped': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = asyncio.run(audit(args.root))
    with args.report.open('x', encoding='utf8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result))
