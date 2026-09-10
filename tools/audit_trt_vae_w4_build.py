"""Independent CPU audit of actual W4 build, inspector, provenance and cleanup."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,validate_request,validate_quantized_manifest,write_new_json,DECODER_W4_MODEL_SHA  # noqa: E402


def audit(root):
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    request,terminal,resources = read('request.json'),read('terminal.json'),read('resource-summary.json')
    if request['schema'] != 't8-trt-decoder-w4a16-build-v1' or digest_file(request['model_path']) != DECODER_W4_MODEL_SHA:
        raise ValueError('Not the audited W4 graph')
    job = terminal['isolated']
    if (terminal['status'] != 'compiled_not_execution_qualified' or job['status'] != 'complete' or
            job['exit_code'] != 0 or job['active_after_cleanup'] or not job['job_assigned_before_task'] or
            resources['status'] != 'observations_within_policy'):
        raise ValueError('Build resource or cleanup evidence failed')
    for path,digest in request['worker_sources'].items():
        if digest_file(root/'source-snapshot'/(digest+'-'+Path(path).name)) != digest:
            raise ValueError('Frozen compiler source changed')
    bundle = Path(terminal['bundle'])
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf8'))
    if manifest['request_sha256'] != validate_request(request) or manifest['source_request'] != request:
        raise ValueError('Published bundle request differs')
    validate_quantized_manifest(manifest,request)
    if digest_file(bundle/'model.engine') != manifest['engine_sha256'] or manifest['engine_sha256'] != terminal['engine_sha256']:
        raise ValueError('Published engine bytes changed')
    layers = read('engine-layers.json')['Layers']
    dq = [layer for layer in layers if 'DequantizeLinear' in layer.get('Metadata','')]
    names = [name for layer in dq for name in re.findall(r'ONNX Layer: ([^\]]*DequantizeLinear)',layer['Metadata'])]
    if len(names) != 144 or len(set(names)) != 144:
        raise ValueError('Actual inspector did not retain all144 distinct quantized weight dequantization origins')
    if any(any(out.get('Format/Datatype') != 'Half' for out in layer.get('Outputs',[])) for layer in dq):
        raise ValueError('Unexpected dequantization output type')
    return {'status':'actual_w4_build_inspector_and_cleanup_verified_not_quality_or_speed',
            'engine_sha256':manifest['engine_sha256'],'engine_bytes':manifest['engine_bytes'],
            'build_seconds':manifest['build_seconds'],'layers':len(layers),
            'layer_types':dict(Counter(layer['LayerType'] for layer in layers)),
            'distinct_weight_dequantize_origins':len(names),'dequantization_output':'Half',
            'strongly_typed':manifest['strongly_typed'],'resources':resources,
            'files':{n:digest_file(root/n) for n in ('request.json','terminal.json','resource-summary.json','engine-layers.json')},
            'limits':'INT4 weight graph plus actual compiled dequantization to half; not all-INT4 arithmetic. Engine file size is not measured VRAM saving. Inference and human review are separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir/'independent-w4-build-audit.json',report)
    print(json.dumps(report,indent=2))
