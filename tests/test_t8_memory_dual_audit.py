from __future__ import annotations

import copy

import pytest
import torch

from tools.audit_t8_memory_dual_pilot import (
    validate_t8_memory_stage,
    validate_video_latent_geometry,
)


def valid_inputs():
    memory = {
        "kind": "t8_h3_memory",
        "head_chunks": 4,
        "ffn_settings": [2, 4096],
        "source_sha256s": ["a" * 64],
        "wrapper_keys": [
            "t8_minimax_h3_low_vram_attention_owner_v1",
            "t8_minimax_h3_chunk_ffn_owner_v1",
        ],
        "runtime_option_keys": [
            "t8_minimax_h3_memory_tokens_v1",
            "sol_take_forward",
        ],
    }
    report = {
        "completed_network_forwards": 4,
        "memory_composition": memory,
        "backend": {
            "status": "not_measured_by_plain_selector_observer",
            "memory_composition": copy.deepcopy(memory),
        },
        "prompt_relay_execution": {
            "completed_forwards": 4,
            "routed_attention_calls": 800,
        },
    }
    terminal = {
        "backend": "t8-memory",
        "memory_head_chunks": 4,
        "memory_ffn_chunks": 2,
    }
    return report, terminal


def test_t8_memory_stage_accepts_exact_authenticated_execution():
    report, terminal = valid_inputs()
    observed = validate_t8_memory_stage(report, terminal)
    assert observed["relay_routed_attention_calls"] == 800
    assert observed["ffn_settings"] == [2, 4096]


def test_t8_memory_stage_accepts_head1_deterministic_control():
    report, terminal = valid_inputs()
    terminal["memory_head_chunks"] = 1
    report["memory_composition"]["head_chunks"] = 1
    report["backend"]["memory_composition"]["head_chunks"] = 1
    observed = validate_t8_memory_stage(report, terminal)
    assert observed["head_chunks"] == 1
    assert observed["ffn_settings"] == [2, 4096]


def test_t8_audit_uses_actual_h3_vae_and_dit_spatial_stride():
    low = torch.zeros((1, 24, 37, 14, 28))
    assert validate_video_latent_geometry(
        low, width=448, height=224, label="Low-stage"
    ) == {"pixels": [448, 224], "latent_spatial": [14, 28]}
    with pytest.raises(ValueError, match="latent geometry"):
        validate_video_latent_geometry(
            torch.zeros((1, 24, 37, 16, 32)),
            width=448,
            height=224,
            label="Low-stage",
        )


@pytest.mark.parametrize(
    "mutation",
    ["heads", "ffn", "wrapper", "receipt", "forwards", "relay", "source"],
)
def test_t8_memory_stage_fails_closed_on_partial_or_descriptive_evidence(mutation):
    report, terminal = valid_inputs()
    if mutation == "heads":
        report["memory_composition"]["head_chunks"] = 3
    elif mutation == "ffn":
        report["memory_composition"]["ffn_settings"] = [4, 4096]
    elif mutation == "wrapper":
        report["memory_composition"]["wrapper_keys"].pop()
    elif mutation == "receipt":
        report["backend"]["memory_composition"] = {}
    elif mutation == "forwards":
        report["completed_network_forwards"] = 3
    elif mutation == "relay":
        report["prompt_relay_execution"]["routed_attention_calls"] = 0
    else:
        report["memory_composition"]["source_sha256s"] = []
    with pytest.raises(ValueError, match="authenticated T8 memory"):
        validate_t8_memory_stage(report, terminal)
