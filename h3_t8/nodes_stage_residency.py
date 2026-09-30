"""Registered public helpers for Leon draft identity and owned stage release."""
import json

from comfy_api.latest import io


class MiniMaxH3StageResidencyReleaseT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3StageResidencyReleaseT8',
            display_name='MiniMax H3 Finished Stage Release (T8)',
            category='T8/MiniMax H3/Performance/Experimental',
            description='只释放明确传入、已结束阶段的模型驻留；保留其他缓存，不保证峰值不OOM。',
            inputs=[io.Model.Input('model', optional=True), io.Model.Input('model_hires', optional=True),
                io.Clip.Input('clip', optional=True), io.Vae.Input('video_vae', optional=True),
                io.Vae.Input('audio_vae', optional=True), io.Vae.Input('vae', optional=True)],
            outputs=[io.String.Output('report_json')])

    @classmethod
    def execute(cls, model=None, model_hires=None, clip=None, video_vae=None, audio_vae=None, vae=None):
        from .progressive_stage_residency import release_finished_stage
        return io.NodeOutput(json.dumps(release_finished_stage(model, model_hires, clip,
            video_vae, audio_vae, vae), ensure_ascii=False, allow_nan=False))


class MiniMaxH3FirstPassFingerprintT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3FirstPassFingerprintT8',
            display_name='MiniMax H3 First Pass Fingerprint (T8)',
            category='T8/MiniMax H3/Performance/Experimental',
            description='校验实际一采模型/LoRA/条件/AV/时钟内容；未知补丁保留，但禁止跨执行误用缓存。',
            inputs=[io.Model.Input('model'), io.Conditioning.Input('positive'), io.Latent.Input('av_latent'),
                io.Sigmas.Input('sigmas'), io.Int.Input('seed', min=0, max=2**64-1),
                io.String.Input('scope_json', extra_dict={'forceInput': True})],
            outputs=[io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, positive, av_latent, sigmas, seed, scope_json):
        from .progressive_first_pass_draft import fingerprint_first_pass
        return io.NodeOutput(json.dumps(fingerprint_first_pass(model, positive, av_latent,
            sigmas, seed, scope_json), ensure_ascii=False, allow_nan=False))


STAGE_RESIDENCY_NODE_CLASSES = [MiniMaxH3StageResidencyReleaseT8, MiniMaxH3FirstPassFingerprintT8]
