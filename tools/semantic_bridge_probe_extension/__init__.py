"""Owned validation-only nodes. Never part of the production node registry."""
import importlib
import json
from pathlib import Path

from comfy_api.latest import ComfyExtension, io


class SemanticBridgeLatentCapture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8SemanticBridgeLatentCapture", category="T8/Probe only",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent):
        import folder_paths
        import nodes
        expected = Path(__file__).resolve().parents[2]
        owner = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3SemanticBridgeApplyT8"]
        package_name = owner.__module__.rsplit(".", 1)[0]
        package = importlib.import_module(package_name)
        if Path(package.__file__).resolve() != expected / "__init__.py":
            raise RuntimeError("Bridge capture resolved a different checkout")
        core = importlib.import_module(package_name + ".core")
        capture = importlib.import_module(package_name + ".tools.trt_latent_capture")
        video, audio = core.nested_av_parts(av_latent)
        report = capture.capture_parts(video, audio, folder_paths.get_output_directory(),
                                       allowed_root=expected / "artifacts")
        return io.NodeOutput(av_latent, json.dumps(report))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [SemanticBridgeLatentCapture]


async def comfy_entrypoint():
    return Extension()
