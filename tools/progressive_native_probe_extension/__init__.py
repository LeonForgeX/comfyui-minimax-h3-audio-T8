"""Isolated trained-model chain entry; never part of the production registry."""

import importlib
import json
import logging
from pathlib import Path

from comfy_api.latest import ComfyExtension, io


def project_module(name):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS['MiniMaxH3ProgressiveSamplerEXPT8']
    package = importlib.import_module(cls.__module__.rsplit('.', 1)[0])
    expected = Path(__file__).resolve().parents[2] / '__init__.py'
    if Path(package.__file__).resolve() != expected:
        raise RuntimeError('Native probe registered against the wrong project')
    return importlib.import_module(package.__name__ + '.' + name)


class NativeChainProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='T8ProgressiveNativeChainProbe', category='T8/Probe only',
            inputs=[io.Model.Input('model'), io.Model.Input('model_hires'),
                    io.Clip.Input('clip'), io.Vae.Input('video_vae'), io.Vae.Input('audio_vae'),
                    io.String.Input('chain_id'), io.String.Input('global_prompt', multiline=True),
                    io.String.Input('upscaler_model'),
                    io.Combo.Input('eav_mode', options=['disabled', 'report_only', 'apply_exp']),
                    io.Combo.Input('tst_mode', options=['disabled', 'report_only', 'apply_exp']),
                    io.Custom('H3_T8_PROMPT_RELAY_PLAN').Input('prompt_relay_plan', optional=True)],
            outputs=[io.String.Output('video_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, model_hires, clip, video_vae, audio_vae, chain_id, global_prompt,
                upscaler_model, eav_mode, tst_mode, prompt_relay_plan=None):
        import torch
        import comfy.model_management as mm
        import comfy.utils
        if mm.get_torch_device().type != 'cuda':
            raise RuntimeError('Full trained generation requires explicit GPU mode')
        torch.set_num_threads(2)
        core = project_module('core')
        sampling = project_module('sampling')
        template, _ = core.empty_av_latent(896, 448, 124)
        low, sampler, sigmas = sampling.setup_dual_clock_sampling(
            model, template, 8, 12., 3., 'euler', 'native_flow')
        high, _, high_sigmas = sampling.setup_dual_clock_sampling(
            model_hires, template, 8, 12., 3., 'euler', 'native_flow')
        if not torch.equal(sigmas, high_sigmas):
            raise RuntimeError('Native phase schedules differ')
        if tst_mode != 'disabled':
            bind = project_module('tst_model').build_tst_model
            low, _ = bind(low, sigmas, mode=tst_mode, tau=.2, max_workspace_mib=2048)
            high, _ = bind(high, sigmas, mode=tst_mode, tau=.2, max_workspace_mib=2048)
        logging.info('[T8 native chain] binding actual trained producers and model contents')
        job = project_module('progressive_job').NativeProgressiveJob(
            low, high, sampler, sigmas, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            chain_id=chain_id, total_duration_seconds=8., width=896, height=448,
            upscaler_model=upscaler_model, global_prompt=global_prompt, base_seed=2609032101,
            render_window_frames=124, context_frames=22, low_evaluations=4, low_scale=.5,
            prompt_relay_plan=prompt_relay_plan, query_chunk_rows=256,
            sampling_options=dict(reserve_vram_mib=2048, eav_mode=eav_mode, eav_tau=.2))
        progress = comfy.utils.ProgressBar(16)

        def notify(segment, step, prediction, state, total):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(segment * 8 + step + 1, 16)
            logging.info('[T8 native chain] segment=%d step=%d/%d', segment, step + 1, total)

        path, report = project_module('progressive_delivery').ProgressiveChainDelivery(
            job, filename_prefix='Progressive_Trained_UNREVIEWED').run(callback=notify)
        body = json.loads(report)
        if body.get('human_qualified') is not False:
            raise RuntimeError('Mechanical execution must not claim human qualification')
        return io.NodeOutput(path, report)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [NativeChainProbe]


async def comfy_entrypoint():
    return Extension()
