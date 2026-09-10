"""Local-only, three-file review server with HTTP Range support for video seeking."""
import argparse
from pathlib import Path

from aiohttp import web


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8786)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    research = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909"
    if not root.is_relative_to(research) or root.name != "public":
        raise ValueError("Only a dedicated public review directory can be served")
    paths = {name: (root / name).resolve(strict=True) for name in ("review.html", "A.mp4", "B.mp4")}
    if any(path.parent != root for path in paths.values()):
        raise ValueError("Review files may not escape the public directory")

    async def serve(request):
        name = request.match_info["name"]
        if name not in paths:
            raise web.HTTPNotFound()
        return web.FileResponse(paths[name], headers={"Cache-Control": "no-store"})

    app = web.Application()
    app.router.add_get("/{name}", serve)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
