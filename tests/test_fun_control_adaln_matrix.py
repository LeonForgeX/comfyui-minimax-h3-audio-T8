"""Executable input widths across all three wrappers; never filename policy."""
from types import SimpleNamespace
import pytest
import torch
from h3_audio_t8_pkg import h3_fun_control_advanced as fun


def module(width, blocks):
    if width is None:
        return SimpleNamespace()
    projection = SimpleNamespace(linear=SimpleNamespace(weight=torch.empty(2, width)))
    return SimpleNamespace(**{blocks: [SimpleNamespace(adaln_proj=projection)]})


@pytest.mark.parametrize("backend", ["official_model_patch", "compatibility", "native"])
@pytest.mark.parametrize("base_width,control_width", [(8, 8), (2688, 2688), (8, 2688), (2688, 8),
                                                     (None, 8), (8, None), (None, None)])
def test_complete_live_width_matrix(backend, base_width, control_width):
    base, control = module(base_width, "blocks"), module(control_width, "control_blocks")
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=base))
    holder = ({"model": control} if backend == "compatibility" else
              SimpleNamespace(**{"control_model" if backend == "native" else "model": control}))
    bundle = fun.H3FunControlBundle(backend, holder, "any_user_name.safetensors", "arbitrary", {})
    if base_width is not None and control_width is not None and base_width != control_width:
        with pytest.raises(RuntimeError, match=f"{base_width}.*{control_width}"):
            fun._assert_compatible_adaln_pair(model, bundle)
    else:
        result = fun._assert_compatible_adaln_pair(model, bundle)
        assert result["base_input_dim"] == base_width and result["control_input_dim"] == control_width
        assert result["status"] == ("unknown" if None in (base_width, control_width) else "matched_live_input_width")
        assert result["basis"] == "live_tensor_shapes_not_filenames"
