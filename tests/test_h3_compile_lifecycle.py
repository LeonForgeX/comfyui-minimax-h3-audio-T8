"""Real CPU H3 graphs through Core's compile wrapper, not CUDA compile parity.

The counting backend executes captured FX graphs eagerly. VDN ownership guards
are exercised, but the small model does not load trained VDN branch weights.
"""

import copy
from types import MethodType

import pytest
import torch
import comfy.ops
from comfy.ldm.minimax import model as minimax
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher
from comfy.model_management import InterruptProcessingException
from comfy.patcher_extension import WrapperExecutor, WrappersMP

from h3_audio_t8_pkg import multikeyframe_advanced as multi
from h3_audio_t8_pkg import vdn_h3_advanced as vdn
from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend


class _SmallBase(torch.nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion_model = diffusion

    def apply_model(self, x, timestep, context, options, payload, abort=None):
        result = self.diffusion_model(
            x, timestep, context, transformer_options=options, minimax_payload=payload)
        if abort is not None:
            raise abort("controlled interruption after real H3 forward")
        return result


def _small_model(monkeypatch, adapted):
    monkeypatch.setattr(minimax, "optimized_attention", attention.attention_pytorch)
    torch.manual_seed(817)
    diffusion = minimax.MiniMaxH3Model(
        hidden_size=8, num_layers=1, token_refiner_num_layers=0,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
        rope_inv_freq_len=1, dtype=torch.float32, device="cpu",
        operations=comfy.ops.disable_weight_init)
    for parameter in diffusion.parameters():
        torch.nn.init.normal_(parameter, std=.05)
    diffusion.requires_grad_(False)
    diffusion.rope.inv_freq.fill_(1.)
    if adapted:
        diffusion._forward = MethodType(multi._multikeyframe_forward, diffusion)
    patcher = ModelPatcher(_SmallBase(diffusion), torch.device("cpu"), torch.device("cpu"))
    set_h3_attention_backend(patcher, attention.attention_pytorch)
    return patcher


@pytest.mark.parametrize("adapted", [False, True], ids=["native", "multikeyframe"])
@pytest.mark.parametrize("compile_first", [False, True], ids=["owner-first", "compile-first"])
def test_real_h3_compile_graphs_keep_ownership_and_restore_on_error_and_cancel(
    adapted, compile_first, monkeypatch,
):
    compiler = pytest.importorskip("comfy_api.torch_helpers.torch_compile")
    import torch._dynamo
    torch._dynamo.reset()
    model = _small_model(monkeypatch, adapted)
    original_diffusion = model.model.diffusion_model
    graphs, executions = [], []

    def eager_graph_backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        def execute(*args):
            executions.append(True)
            return graph_module.forward(*args)
        return execute

    def install_compiler():
        compiler.set_torch_compile_wrapper(model, backend=eager_graph_backend, fullgraph=False)
        assert len(model.get_wrappers(WrappersMP.APPLY_MODEL, compiler.COMPILE_KEY)) == 1

    try:
        if compile_first:
            install_compiler()

        def block_hook(args, extra):
            return extra["original_block"](args)
        options = model.model_options["transformer_options"]
        options[vdn.OWNER_HOOKS_KEY] = (block_hook,)
        model.set_model_patch_replace(block_hook, "dit", "double_block", 0)
        def ownership_guard(executor, x, timestep, context, transformer_options=None, **kwargs):
            vdn.validate_vdn_runtime_options(transformer_options)
            return executor(x, timestep, context, transformer_options, **kwargs)
        model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "vdn_contract_probe", ownership_guard)
        if not compile_first:
            install_compiler()
        options = {**model.model_options["transformer_options"], "wrappers": model.wrappers,
                   "sample_sigmas": torch.tensor([1., .5, 0.])}
        x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
        timestep, context = torch.tensor([500.]), torch.randn(1, 2, 8)
        payload = {"layout": minimax.PackedLayout(2, 2, 4, 4, 3),
                   multi.PAYLOAD_VISUAL_NOISE_AUGS_KEY: []}
        def run(abort=None, runtime_options=None):
            runtime_options = copy.deepcopy(options) if runtime_options is None else runtime_options
            executor = WrapperExecutor.new_class_executor(
                model.model.apply_model, model.model,
                model.get_wrappers(WrappersMP.APPLY_MODEL, compiler.COMPILE_KEY))
            return executor.execute(x, timestep, context, runtime_options, payload, abort=abort)
        with torch.no_grad():
            expected = model.model.apply_model(x, timestep, context, copy.deepcopy(options), payload)
            for _ in range(2):
                actual = run()
                for baseline, result in zip(expected, actual):
                    torch.testing.assert_close(baseline, result, rtol=1e-5, atol=1e-6)
                assert model.model.diffusion_model is original_diffusion
            assert graphs and executions, "compile must capture AND execute real tensor graphs"
            for error in (RuntimeError, KeyboardInterrupt, InterruptProcessingException):
                count = len(executions)
                with pytest.raises(error, match="controlled interruption"):
                    run(abort=error)
                assert len(executions) > count
                assert model.model.diffusion_model is original_diffusion
            corrupted = copy.deepcopy(options)
            corrupted["patches_replace"]["dit"][("double_block", 0)] = lambda *args: None
            with pytest.raises(RuntimeError, match="indices.*0"):
                run(runtime_options=corrupted)
            assert model.model.diffusion_model is original_diffusion
            actual = run()
            for baseline, result in zip(expected, actual):
                torch.testing.assert_close(baseline, result, rtol=1e-5, atol=1e-6)
            assert options["patches_replace"]["dit"][("double_block", 0)] is block_hook
            assert model.model.diffusion_model is original_diffusion
    finally:
        torch._dynamo.reset()
