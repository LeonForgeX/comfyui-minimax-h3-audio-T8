"""Draft registration only for an explicitly isolated candidate test server.

No replacement nodes or sampler overrides. Never imported by the main registry.
"""
import importlib
from pathlib import Path
import sys
import types

from comfy_api.latest import ComfyExtension


class OutpaintCandidateExtension(ComfyExtension):
    async def get_node_list(self):
        name = "t8_outpaint_candidate_isolated_probe"
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(Path(__file__).resolve().parents[2] / "h3_t8"), str(Path(__file__).resolve().parents[2])]
            sys.modules[name] = package
        stages = importlib.import_module(name + ".nodes_video_outpaint")
        candidates = importlib.import_module(name + ".nodes_video_outpaint_candidates")
        reload = importlib.import_module(name + ".nodes_video_outpaint_reload")
        preview = importlib.import_module(name + ".nodes_video_outpaint_preview")
        guidance = importlib.import_module(name + ".nodes_video_outpaint_guidance")
        dlss = importlib.import_module(name + ".nodes_dlss_nr_advanced")
        return [*stages.VIDEO_OUTPAINT_DRAFT_NODE_CLASSES,
                *candidates.VIDEO_OUTPAINT_CANDIDATE_DRAFT_NODE_CLASSES,
                *guidance.VIDEO_OUTPAINT_GUIDANCE_DRAFT_NODE_CLASSES,
                reload.MiniMaxH3VideoOutpaintLoadPreparedT8,
                preview.MiniMaxH3VideoOutpaintGeometryPreviewT8,
                dlss.MiniMaxH3DLSSNRRuntimeAuditT8Advanced,
                dlss.MiniMaxH3DLSSNRVideoFileT8Advanced]


def comfy_entrypoint():
    return OutpaintCandidateExtension()
