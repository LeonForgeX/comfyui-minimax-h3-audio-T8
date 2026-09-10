"""Load draft nodes only in the explicitly isolated outpaint test server."""
import importlib
from functools import partial
import os
from pathlib import Path
import sys
import types

from comfy_api.latest import ComfyExtension


class OutpaintProbeExtension(ComfyExtension):
    async def get_node_list(self):
        name = "t8_outpaint_isolated_probe"
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(Path(__file__).resolve().parents[2] / "h3_t8"), str(Path(__file__).resolve().parents[2])]
            sys.modules[name] = package
        module = importlib.import_module(name+".nodes_video_outpaint")
        capture_path = os.environ.get("T8_OUTPAINT_CAPTURE_RGB")
        if capture_path:
            compose = importlib.import_module(name+".video_outpaint_compose")
            module.compose_sampled_outpaint = partial(compose.compose_sampled_outpaint, capture_rgb_path=capture_path)
        return module.VIDEO_OUTPAINT_DRAFT_NODE_CLASSES


def comfy_entrypoint():
    return OutpaintProbeExtension()
