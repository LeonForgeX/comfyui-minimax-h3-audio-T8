"""Actual native prepared input loading on CPU only, without DiT/VAE models."""
import argparse
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new CPU evidence directory')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0', OMP_NUM_THREADS='2')
    backend = Path(__file__).resolve().parents[1] / 'h3_t8'
    package = types.ModuleType('_prepared_cpu')
    package.__path__ = [str(backend)]
    sys.modules[package.__name__] = package
    from _prepared_cpu.prepared_generation_contract import read_bundle, generation_request
    bundle = read_bundle(args.bundle)
    request = generation_request(bundle, 8301, 'CPU_ONLY', {})
    if bundle['kind'] == 'ltx_refine':
        sys.path[:0] = [str(backend / 'prepared_backend'), *request['isolated_paths']]
    else:
        sys.path[:0] = [str(backend / 'prepared_backend'), str(Path(request['source']) / 'src')]
    import torch
    torch.set_num_threads(2)
    report = {'kind': bundle['kind'], 'models_loaded': False, 'GPU_rerun': False,
        'full_asset_hash_verification': 'separate_bundle_builder_and_generation_controller'}
    if bundle['kind'] == 'ltx_refine':
        from ltx_prepared_request import load_prepared
        inputs, cache = load_prepared(request)
        video_context, audio_context = cache.contexts(request['prompt'], torch.device('cpu'))
        report.update(input_shapes={key: list(value.shape) for key, value in inputs.items()},
            context_shapes={'video': list(video_context.shape), 'audio': list(audio_context.shape)},
            prompt_cache_matched=True, geometry=request['geometry'])
    else:
        from taomate_input_preflight import load_inputs
        text, _, _, known, receipt = load_inputs(request, torch)
        report.update(text_shapes={key: list(value.shape) for key, value in text.items()},
            clean_audio_shape=list(known.shape), teacher_receipt=receipt, all_three_saved_milestones_exact=True)
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError('CPU input qualification accessed CUDA')
    report.update(status='actual_native_prepared_inputs_CPU_pass_not_inference', cuda_initialized=False)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
