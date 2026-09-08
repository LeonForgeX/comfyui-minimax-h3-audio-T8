import json

import pytest

from tools.vdn_probe_extension import NativeBranchProbe, NativeBranchResult, check_native
from test_vdn_attention_compat import model_fixture
from h3_audio_t8_pkg.vdn_h3_advanced import ATTACHMENT_KEY, ADDITIONAL_MODEL_KEY


def native():
    model = model_fixture()
    model.patches = {f"target_{i}": [] for i in range(259)}
    return model


@pytest.mark.parametrize("kind", ["attachment", "additional", "options", "dit", "patch_count"])
def test_native_probe_rejects_actual_vdn_contract_keys(kind):
    model = native()
    options = model.model_options["transformer_options"]
    if kind == "attachment":
        model.set_attachments(ATTACHMENT_KEY, {"stage": "stage_dmd_8nfe"})
    elif kind == "additional":
        model.additional_models[ADDITIONAL_MODEL_KEY] = []
    elif kind == "options":
        options["t8_openvdn_expected_block_hooks"] = ()
    elif kind == "dit":
        options["patches_replace"] = {"dit": {("double_block", 0): object()}}
    else:
        model.patches.pop("target_1")
    with pytest.raises(RuntimeError):
        check_native(model, options)


def test_native_probe_observes_every_forward_and_rejects_incomplete_run():
    model = native()
    patched, state = NativeBranchProbe.execute(model, model_fixture()).result
    wrapper = patched.get_wrappers("diffusion_model", "t8_native_branch_probe")[0]
    options = patched.model_options["transformer_options"]
    for _ in range(4):
        assert wrapper(lambda *a, **kw: "native", None, None, None, options) == "native"
    latent = {"samples": object()}
    result, report = NativeBranchResult.execute(latent, state, 4).result
    assert result is latent and json.loads(report)["forwards"] == 4
    with pytest.raises(RuntimeError, match="every expected forward"):
        NativeBranchResult.execute(latent, state, 5)


def test_native_probe_rechecks_late_contamination_before_model_call():
    model = native()
    patched, state = NativeBranchProbe.execute(model, model_fixture()).result
    wrapper = patched.get_wrappers("diffusion_model", "t8_native_branch_probe")[0]
    options = {"t8_openvdn_expected_block_hooks": ()}
    with pytest.raises(RuntimeError, match="VDN transformer state"):
        wrapper(lambda *a: pytest.fail("must not sample"), None, None, None, options)
    assert state["forwards"] == 0
