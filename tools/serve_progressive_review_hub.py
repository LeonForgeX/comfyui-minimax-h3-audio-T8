"""Loopback-only inline review server. No directory listing, uploads or browser launch."""
import argparse
from pathlib import Path
import re

from aiohttp import web


def create_app(root, *, research):
    root, research = Path(root).resolve(strict=True), Path(research).resolve(strict=True)
    if root.name != "public" or not root.is_relative_to(research):
        raise ValueError("Only a dedicated public research directory can be served")
    paths = {"review.html": (root / "review.html").resolve(strict=True)}
    for candidate in root.iterdir():
        if re.fullmatch(r"pair-[0-9]+-[AB]\.mp4", candidate.name):
            paths[candidate.name] = candidate.resolve(strict=True)
    if not 2 <= len(paths)-1 <= 64 or any(not p.is_file() or p.parent != root for p in paths.values()):
        raise ValueError("Review files absent, out of bounds, or escaping public directory")
    for name in set(paths)-{"review.html"}:
        partner = name[:-5] + ("B" if name[-5] == "A" else "A") + ".mp4"
        if partner not in paths:
            raise ValueError("Every inline pair needs both A and B")

    async def serve(request):
        name = request.match_info.get("name", "review.html")
        if name not in paths:
            raise web.HTTPNotFound()
        return web.FileResponse(paths[name], headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    app = web.Application()
    app.router.add_get("/", serve)
    app.router.add_get("/{name}", serve)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid loopback review port")
    research = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909"
    web.run_app(create_app(args.root, research=research), host="127.0.0.1", port=args.port, print=None)
