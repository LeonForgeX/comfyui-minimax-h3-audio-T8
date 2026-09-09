"""Geometry-only draft registration for an explicitly isolated CPU test server.

No Sample/Prepare/Compose nodes are registered here. This directory is not a
production custom_nodes installation and does not change the main node registry.
"""
import importlib
from pathlib import Path
import sys
import types

from comfy_api.latest import ComfyExtension


class OutpaintPreviewExtension(ComfyExtension):
    async def get_node_list(self):
        name = "t8_outpaint_geometry_isolated_probe"
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(Path(__file__).resolve().parents[2])]
            sys.modules[name] = package
        stages = importlib.import_module(name + ".nodes_video_outpaint")
        preview = importlib.import_module(name + ".nodes_video_outpaint_preview")
        return [stages.MiniMaxH3VideoOutpaintPlanT8, preview.MiniMaxH3VideoOutpaintGeometryPreviewT8]


def comfy_entrypoint():
    return OutpaintPreviewExtension()
