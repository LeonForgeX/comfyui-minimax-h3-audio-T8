"""Reproduce the pinned control header's missing conversion on native Core CPU.

Meta tensors contain shapes, not downloaded/decoded model weights. This proves
loader selection only; not a quantized forward or numerical equivalence.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence output')
    sys.path.insert(0, str(args.core.resolve(strict=True)))
    sys.argv = ['fun-header-audit', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import comfy.utils
    import torch
    torch.set_num_threads(2)
    header = json.loads(args.header.read_text(encoding='utf-8'))
    if header['revision'] != '4709bcbb03fa9dc34e342bf46d19d9039463244f':
        raise ValueError('Expected pinned target header')
    dtypes = {'I8': torch.int8, 'U8': torch.uint8, 'F32': torch.float32,
        'F16': torch.float16, 'BF16': torch.bfloat16, 'F8_E4M3': torch.float8_e4m3fn,
        'F8_E5M2': torch.float8_e5m2}
    sd = {key: torch.empty(item['shape'], dtype=dtypes[item['dtype']], device='meta')
        for key, item in header['tensors'].items()}
    before = comfy.utils.detect_layer_quantization(sd, '')
    quant = header['quantization_metadata']
    metadata = {'_quantization_metadata': json.dumps(quant)}
    converted, returned_metadata = comfy.utils.convert_old_quants(dict(sd), metadata=metadata)
    after = comfy.utils.detect_layer_quantization(converted, '')
    layers = sorted(quant['layers'])
    if before is not None or after != {'mixed_ops': True} or len(layers) != 20:
        raise ValueError('Expected unconverted selection failure and converted mixed-ops selection')
    for key in layers:
        if json.loads(bytes(converted[key + '.comfy_quant'].tolist())) != quant['layers'][key]:
            raise ValueError('Converted metadata differs from original layer definition')
    if any(converted[key] is not value for key, value in sd.items()) or returned_metadata != metadata:
        raise ValueError('Native conversion changed original weights or metadata')
    report = {'status': 'header_selection_gap_reproduced_cpu_no_weights_or_inference',
        'before_detection': before, 'after_native_conversion': after,
        'layers': len(layers), 'header_sha256': header['header_sha256'],
        'source_header_report_sha256': hashlib.sha256(args.header.read_bytes()).hexdigest(),
        'core_functions': {name: hashlib.sha256(inspect.getsource(getattr(comfy.utils, name)).encode()).hexdigest()
            for name in ('detect_layer_quantization', 'convert_old_quants')},
        'original_tensors_identity_preserved': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'full_tensor_payload_downloaded': False,
        'scope': 'actual native Core detection on pinned header shapes; runtime loader fix and full forward pending'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
