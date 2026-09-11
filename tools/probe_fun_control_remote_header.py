"""Read only a pinned HF safetensors header with strict bounded HTTP ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import requests


REVISION = '4709bcbb03fa9dc34e342bf46d19d9039463244f'
REPOSITORY = 'berryber09/MiniMax-H3-Fun-Controlnet-Union-w4a8'
FILENAME = 'minimax_h3_fun_controlnet_union_pruned_w4a8.safetensors'
URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'
LIMIT = 4 * 1024 * 1024


def bounded_range(session, start, end):
    if not 0 <= start <= end < LIMIT + 8:
        raise ValueError('Header request exceeds bounded range')
    # Refuse ignored Range requests without reading a multi-GB response body.
    with session.get(URL, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                     timeout=(15, 45), stream=True) as response:
        if response.status_code != 206:
            raise ValueError(f'Server did not honor byte range: HTTP {response.status_code}')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match or tuple(map(int, match.groups()[:2])) != (start, end):
            raise ValueError('Unexpected Content-Range; no body consumed')
        raw = response.raw.read(end - start + 2)
        if len(raw) != end - start + 1:
            raise ValueError('Range payload length differs')
        return raw, int(match.group(3))


def inspect_header():
    with requests.Session() as session:
        # Public repository only; no HF tokens, authentication files or cookies loaded.
        session.trust_env = False
        prefix, total = bounded_range(session, 0, 7)
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= LIMIT or length + 8 >= total:
            raise ValueError('Invalid safetensors header length')
        header, repeated_total = bounded_range(session, 8, 7 + length)
        if repeated_total != total:
            raise ValueError('Remote size changed between bounded reads')
    record = json.loads(header)
    metadata = record.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError('Invalid safetensors metadata')
    for name, tensor in record.items():
        if not isinstance(tensor, dict) or not {'dtype', 'shape', 'data_offsets'} <= tensor.keys():
            raise ValueError(f'Invalid tensor descriptor {name}')
        start, end = tensor['data_offsets']
        if not 0 <= start <= end <= total - length - 8:
            raise ValueError('Tensor offsets exceed remote file size')
    quant = metadata.get('_quantization_metadata')
    quant = json.loads(quant) if quant is not None else None
    return {'status': 'header_only_not_loaded_or_inference_tested', 'repository': REPOSITORY,
        'revision': REVISION, 'file': FILENAME, 'source_url': URL, 'total_file_bytes': total,
        'downloaded_bytes': length + 8, 'header_sha256': hashlib.sha256(header).hexdigest(),
        'metadata_keys': sorted(metadata), 'quantization_metadata': quant,
        'tensors': record, 'tensor_count': len(record), 'gpu_used': False,
        'scope': 'actual pinned remote tensor descriptors; no tensor payload, no full model SHA'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence output')
    report = inspect_header()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ('status', 'total_file_bytes',
        'downloaded_bytes', 'header_sha256', 'metadata_keys', 'tensor_count')}))


if __name__ == '__main__':
    main()
