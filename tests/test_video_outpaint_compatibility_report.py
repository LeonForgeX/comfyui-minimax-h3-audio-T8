from h3_audio_t8_pkg.video_outpaint_compatibility_report import (
    OUTPAINT_COMBINATION_POLICY,
    outpaint_model_compatibility_report,
)
from test_video_outpaint_execution import _model


def test_policy_names_every_requested_acceleration_and_postprocess_boundary():
    assert OUTPAINT_COMBINATION_POLICY["native_stock_20_steps"] == "supported"
    assert "supported_memory" in OUTPAINT_COMBINATION_POLICY["pinned_kj_low_vram_attention"]
    assert OUTPAINT_COMBINATION_POLICY["dlss_nr_after_completed_compose"].startswith("supported")
    for name in ("turbo_lora", "speed", "sla_attention", "vdn", "fast_h3", "prompt_relay"):
        assert OUTPAINT_COMBINATION_POLICY[name].startswith("unsupported")


def test_report_accepts_bare_native_and_rejects_unknown_weight_patch():
    model = _model()
    assert outpaint_model_compatibility_report(model)["ready"] is True
    model.patches["diffusion_model.fake.weight"] = object()
    report = outpaint_model_compatibility_report(model)
    assert report["ready"] is False
    assert report["model_filename_used_as_evidence"] is False
