import asyncio

from aiohttp import ClientSession, web
import pytest

from tools.serve_progressive_review_hub import create_app


@pytest.fixture
def public(tmp_path):
    folder = tmp_path / 'public'
    folder.mkdir()
    (folder / 'review.html').write_text('<html>fixture</html>', encoding='utf-8')
    for label in ('A', 'B'):
        (folder / f'pair-0-{label}.mp4').write_bytes(b'0123456789')
    (folder / 'private-receipt.json').write_text('never expose this fixture', encoding='utf-8')
    return folder


def test_loopback_head_range_and_private_paths(public):
    async def check():
        runner = web.AppRunner(create_app(public, research=public.parent))
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        try:
            await site.start()
            port = runner.addresses[0][1]
            base = f'http://127.0.0.1:{port}'
            async with ClientSession() as client:
                async with client.get(base+'/pair-0-A.mp4', headers={'Range': 'bytes=2-5'}) as response:
                    assert response.status == 206 and await response.read() == b'2345'
                    assert response.headers['Content-Range'] == 'bytes 2-5/10'
                async with client.head(base+'/pair-0-B.mp4') as response:
                    assert response.status == 200 and response.headers['Content-Length'] == '10'
                async with client.get(base+'/') as response:
                    assert response.status == 200 and 'fixture' in await response.text()
                for path in ('/private-receipt.json', '/missing.mp4', '/subdir/private.json'):
                    async with client.get(base+path) as response:
                        assert response.status == 404
        finally:
            await runner.cleanup()
    asyncio.run(check())


def test_half_pair_not_served(public):
    (public / 'pair-0-B.mp4').unlink()
    with pytest.raises(ValueError):
        create_app(public, research=public.parent)


def test_nonpublic_directory_not_served(public):
    with pytest.raises(ValueError):
        create_app(public.parent, research=public.parent)
