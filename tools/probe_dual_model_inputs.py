"""Load actual native assets and verify dual-stage identities on CPU, no inference."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip-only', action='store_true', help='Only rerun the previously failing CLIP identity')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output file')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.core))
    sys.argv = ['identity-probe', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import torch
    import nodes
    spec = importlib.util.spec_from_file_location('dual_probe_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from dual_probe_pkg.long_video_dual_identity import stage_model_identity
    from dual_probe_pkg.nodes_long_video_dual_model import _component_identity
    from dual_probe_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
    torch.set_num_threads(2)
    result = {'status': 'running', 'cpu_only': True, 'generation': False, 'checks': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if args.clip_only:
            print('Loading and hashing actual NVFP4/AWQ CLIP on CPU', flush=True)
            component = nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax', 'cpu')[0]
            result['checks']['clip'] = _component_identity(component)
            result['status'] = 'actual_CPU_CLIP_identity_pass'
            return
        print('Loading actual INT8/ConvRot native H3 on CPU', flush=True)
        model = nodes.UNETLoader().load_unet('minimax_h3_fl2va_int8_convrot.safetensors', 'default')[0]
        lora = args.core / 'models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors'
        first, first_report = load_minimax_h3_lora_model(model, str(lora), 1.0)
        second, second_report = load_minimax_h3_lora_model(model, str(lora), .9)
        print('Hashing actual independent LoRA branches', flush=True)
        identity_first, identity_second = stage_model_identity(first), stage_model_identity(second)
        if identity_first['sha256'] == identity_second['sha256'] or model.patches:
            raise AssertionError('Independent LoRA strengths must differ without changing the shared base')
        result['checks']['models'] = {'first': identity_first, 'second': identity_second,
            'first_lora': json.loads(first_report), 'second_lora': json.loads(second_report)}
        del model, first, second
        gc.collect()
        for label, filename in [('video_vae', 'minimax_h3_video_vae_fp16.safetensors'),
                                ('audio_vae', 'minimax_h3_audio_vae_fp32.safetensors')]:
            print('Loading and hashing ' + label, flush=True)
            component = nodes.VAELoader().load_vae(filename)[0]
            result['checks'][label] = _component_identity(component)
            del component
            gc.collect()
        print('Loading and hashing actual NVFP4/AWQ CLIP on CPU', flush=True)
        component = nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax', 'cpu')[0]
        result['checks']['clip'] = _component_identity(component)
        result['status'] = 'actual_CPU_asset_identity_pass'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'output': str(args.output),
                          'cuda_initialized': result['cuda_initialized']}), flush=True)


if __name__ == '__main__':
    main()
