"""Real many-target Core Bypass LoRA receipts stay bounded; synthetic CPU H3."""
from copy import deepcopy
import json

import comfy.lora
import comfy.sd
import torch

from h3_audio_t8_pkg.progressive_first_pass_draft import low_model_identity
from h3_audio_t8_pkg.progressive_native_bypass_identity import _capture
from h3_audio_t8_pkg.long_video_in_node_loop_advanced import _sha256_json
from test_progressive_first_pass_draft import run, same
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401


def many_target_models():
    low = tiny_model()
    net = low.model.diffusion_model
    net.blocks = torch.nn.ModuleList([deepcopy(net.blocks[0]) for _ in range(50)])
    low.model.model_config.unet_config['num_layers'] = 50
    high = low.clone()
    aliases = {target: alias for alias, target in comfy.lora.model_lora_keys_unet(low.model, {}).items()}
    weights = {}
    for name, layer in low.model.named_modules():
        if not isinstance(layer, torch.nn.Linear) or name + '.weight' not in aliases:
            continue
        alias = aliases[name + '.weight']
        weights[alias + '.lora_up.weight'] = torch.full((layer.out_features, 1), .003)
        weights[alias + '.lora_down.weight'] = torch.full((1, layer.in_features), .004)
        weights[alias + '.alpha'] = torch.tensor(1.)
    return comfy.sd.load_bypass_lora_for_models(low, None, weights, .75, 0)[0], high, len(weights) // 3


def test_many_target_native_lora_save_and_selected_replay_remain_compact(tmp_path, stub_lifter):  # noqa: F811
    low, high, targets = many_target_models()
    assert targets == 258
    first, saved = run(tmp_path, model=low, high=high)
    assert saved['first_pass_draft']['saved_low']
    receipt, = tmp_path.glob('low-boundary-*.json')
    # Before compaction this real producer wrote >11 MiB, silently rejected
    # by Leon's old 512 KiB catalog limit. No synthetic completion is forged.
    assert receipt.stat().st_size < 512 * 1024
    record = json.loads(receipt.read_text())
    summary = record['contract']['model']['weights']['first_pass_native_bypass']
    assert set(summary) == {'schema', 'contract_sha256', 'selection_count', 'target_count'}
    assert summary['target_count'] == targets and summary['selection_count'] == 1
    assert summary['contract_sha256'] == _sha256_json(_capture(low).contract)
    replay, report = run(tmp_path, model=low, high=high, mode='reuse_only',
                         draft_id=saved['first_pass_draft']['cache_id'])
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    same(first, replay)
    before = low_model_identity(low, comfy.samplers.ksampler('euler'))
    proof = _capture(low)
    hook = next(iter(proof.hooks.values()))
    hook.adapter.loaded_keys.add('extra-current-key')
    after = low_model_identity(low, comfy.samplers.ksampler('euler'))
    assert before != after
    assert before['weights']['first_pass_native_bypass']['contract_sha256'] != after['weights']['first_pass_native_bypass']['contract_sha256']
