import asyncio
import hashlib
import json

from aiohttp import ClientSession, web
import pytest

from tools.serve_candidate_combined_review import create_app


@pytest.fixture
def public(tmp_path):
    root = tmp_path / 'public'
    root.mkdir()
    (root / 'review.html').write_text('<html>inline fixture</html>')
    (root / 'clip-01.mp4').write_bytes(b'0123456789')
    (root / 'private.json').write_text('not public')
    (tmp_path / 'delivery.json').write_text(json.dumps({'groups': [{'clips': [
        {'url': 'clip-01.mp4', 'sha256': hashlib.sha256(b'0123456789').hexdigest()}]}]}))
    return root


def test_range_head_and_private_paths(public):
    async def run():
        runner = web.AppRunner(create_app(public, public.parent))
        await runner.setup()
        try:
            await web.TCPSite(runner, '127.0.0.1', 0).start()
            base = f'http://127.0.0.1:{runner.addresses[0][1]}'
            async with ClientSession() as client:
                async with client.get(base + '/clip-01.mp4', headers={'Range': 'bytes=2-5'}) as response:
                    assert response.status == 206 and await response.read() == b'2345'
                    assert response.headers['Content-Range'] == 'bytes 2-5/10'
                async with client.head(base + '/clip-01.mp4') as response:
                    assert response.status == 200 and response.headers['Content-Length'] == '10'
                async with client.get(base + '/') as response:
                    assert response.status == 200 and 'inline fixture' in await response.text()
                for path in ('/private.json', '/delivery.json', '/manifest.json', '/nested/clip-01.mp4'):
                    async with client.get(base + path) as response:
                        assert response.status == 404
        finally:
            await runner.cleanup()
    asyncio.run(run())


def test_changed_media_fails_closed(public):
    (public / 'clip-01.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError):
        create_app(public, public.parent)


def test_nonpublic_directory_fails_closed(public):
    with pytest.raises(ValueError):
        create_app(public.parent, public.parent)


def test_traversal_asset_fails_closed(public):
    path = public.parent / 'delivery.json'
    value = json.loads(path.read_text())
    value['groups'][0]['clips'][0]['url'] = '../clip-01.mp4'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        create_app(public, public.parent)
