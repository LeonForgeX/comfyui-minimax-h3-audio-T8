"""CPU-only evidence audit for the public, fixture-independent T1 export."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json,ENCODER_T1_MODEL_SHA  # noqa: E402


def audit(root):
    import torch
    from safetensors.torch import load_file
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    request,terminal,manifest,resources = [read(n) for n in ('request.json','terminal.json','manifest.json','resource-summary.json')]
    job = terminal['isolated']
    if (terminal['status'] != 'pinned_t1_export_published_not_runtime_qualified' or job['exit_code'] != 0 or
            job['status'] != 'complete' or job['active_after_cleanup'] or not job['job_assigned_before_task'] or
            resources['status'] != 'observations_within_policy'):
        raise ValueError('Public export did not complete under ownership/resource protection')
    for name,digest in request['sources'].items():
        source = Path(name)
        frozen = root/'source-snapshot'/(digest+'-'+source.name) if source.suffix == '.py' else source
        if digest_file(frozen) != digest:
            raise ValueError('Export source identity changed')
    if digest_file(request['native_vae']) != '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522':
        raise ValueError('Unexpected native weights')
    if any(digest_file(p) != ENCODER_T1_MODEL_SHA for p in (root/'encoder-t1.onnx',terminal['output'])):
        raise ValueError('Portable graph is not byte-identical to the already validated graph')
    if manifest['onnx_sha256'] != ENCODER_T1_MODEL_SHA or digest_file(root/'reference.safetensors') != manifest['reference_sha256']:
        raise ValueError('Manifest evidence differs')
    evidence = load_file(str(root/'reference.safetensors'),device='cpu')
    shapes = {'normalized_pixels':(1,3,1,256,256),'native_moments':(1,48,1,16,16),'static_moments':(1,48,1,16,16)}
    for name,shape in shapes.items():
        value = evidence[name]
        if tuple(value.shape) != shape or value.dtype != torch.float16 or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid actual trace evidence')
    if not torch.equal(evidence['native_moments'],evidence['static_moments']) or not manifest['native_static_bit_identical']:
        raise ValueError('Native/specialized trace not identical')
    if manifest['trace_error'] != {'max_absolute':0.0,'relative_rmse':0.0}:
        raise ValueError('Trace metric differs from independent equality check')
    return {'status':'portable_t1_exact_graph_and_native_trace_verified','graph_sha256':ENCODER_T1_MODEL_SHA,
            'native_static_bit_identical':True,'finite_input_and_outputs':True,'resources':resources,
            'files':{n:digest_file(root/n) for n in ('request.json','terminal.json','manifest.json','resource-summary.json','reference.safetensors')},
            'scope':'Synthetic trace, not generated media. Public output bytes equal prior real-image/runtime-validated T1 graph. No new TRT compilation, sampling, long-video or human qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    write_new_json(args.run_dir/'independent-portable-t1-audit.json',result)
    print(json.dumps(result,indent=2))
