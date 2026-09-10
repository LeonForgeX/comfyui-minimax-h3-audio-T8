"""Independent CPU tensor/PNG and lifecycle audit of the static text fixture."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def metrics(a, b):
    import torch
    diff = a.double()-b.double()
    mse = float(torch.mean(diff.square()))
    return {'mse': mse, 'psnr_db': -10*math.log10(mse) if mse else None,
            'max_abs': float(diff.abs().max()), 'bit_exact': bool(torch.equal(a, b))}


def audit(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    request, result, terminal, resources = map(read, ('request.json','result.json','terminal.json','resource-summary.json'))
    job = terminal['isolated']
    if (request['schema'] != 't8-trt-public-text1-v1' or terminal['status'] != 'worker_complete_result_requires_audit'
            or job['status'] != 'complete' or job['exit_code'] or job['active_after_cleanup']
            or not job['job_assigned_before_task'] or resources['status'] != 'observations_within_policy'):
        raise ValueError('Owned worker/resource audit failed')
    for name, sha in request['sources'].items():
        path = Path(name)
        frozen = root/'source-snapshot'/(sha+'-'+path.name) if path.suffix == '.py' else path
        if digest_file(frozen) != sha:
            raise ValueError('Source evidence changed: '+path.name)
    for name, sha in result['files'].items():
        if Path(name).name != name or digest_file(root/name) != sha:
            raise ValueError('Result evidence changed')
    if (result['native_calls'] != {'encode':15,'decode':15}
            or len(result['leases']) != 3 or any(r['calls'] != 15 or r['status'] != 'complete' for r in result['leases'])):
        raise ValueError('Actual full-size T1 dispatch not established')
    for row, bundle in zip(result['leases'], (request['decoder'],request['t1'],request['decoder']), strict=True):
        manifest = json.loads((Path(bundle)/'manifest.json').read_text(encoding='utf8'))
        if row['engine_sha256'] != manifest['engine_sha256'] or row['free_before_deserialize']-row['free_after_release'] > 512*1024**2:
            raise ValueError('Wrong engine or failed release')
    tensors = load_file(root/'outputs.safetensors', device='cpu')
    for key, shape in {'source':(1,512,1024,3),'native':(1,512,1024,3),
                       'trt_same_latent':(1,512,1024,3),'trt_full':(1,512,1024,3),
                       'native_latent':(1,24,1,32,64),'trt_latent':(1,24,1,32,64)}.items():
        if tensors[key].shape != shape or tensors[key].dtype != torch.float32 or not torch.isfinite(tensors[key]).all():
            raise ValueError('Invalid text tensor: '+key)
        if 'latent' not in key or key == 'trt_same_latent':
            raw = (tensors[key][0].clamp(0,1).numpy()*255).round().astype(np.uint8)
            with Image.open(root/(key+'.png')) as image:
                if not np.array_equal(raw,np.array(image)):
                    raise ValueError('Displayed PNG differs from tensor: '+key)
    rows = read('text-layout.json')['rows']
    if len(rows) != 14:
        raise ValueError('Text layout missing')
    comparisons = {}
    for label, a, b in (('native_vs_source','source','native'),
                         ('trt_decoder_vs_native','native','trt_same_latent'),
                         ('trt_full_vs_native','native','trt_full')):
        comparisons[label] = {'whole_image':metrics(tensors[a],tensors[b]),'text_rows':[]}
        for row in rows:
            x0,y0,x1,y1 = row['bbox']
            if not (0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 512):
                raise ValueError('Text extends beyond chart')
            comparisons[label]['text_rows'].append({**row,**metrics(tensors[a][:,y0:y1,x0:x1],tensors[b][:,y0:y1,x0:x1])})
    return {'status':'static_text_actual_public_vae_tensor_png_lifecycle_pass_human_pending',
            'files':result['files'],'comparisons':comparisons,'cuda_initialized':torch.cuda.is_initialized(),
            'limits':'Static synthetic text, not temporal flicker or generated writing. PSNR measures reconstruction difference, not OCR correctness or human readability.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    root = parser.parse_args().run_dir
    result = audit(root)
    write_new_json(root/'independent-text-audit.json',result)
    print(json.dumps({'status':result['status'],'comparisons':{k:v['whole_image'] for k,v in result['comparisons'].items()}}))
