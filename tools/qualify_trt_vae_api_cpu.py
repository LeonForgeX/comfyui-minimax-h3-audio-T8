"""Isolated actual ComfyUI CPU API validation, without browser or inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'tools'))
sys.path.insert(0,str(PROJECT))
from run_progressive_pilot import (CORE,RESEARCH,OwnedServer,execute_graph,preview_report,source_snapshot,wait_ready,write_json)  # noqa: E402
from vdn_probe_environment import probe_resource_config,verify_core_source  # noqa: E402
from tools.build_trt_vae_workflows import recipe  # noqa: E402
from audit_progressive_workflows import audit_candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8204)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New isolated research directory required')
    graphs,paths = {},{}
    for mode,task,name in (('check',None,'Runtime_Check'),('compile',None,'Compile'),
                           ('decoder','T2VA','T2VA_Decoder'),('full','I2VA','I2VA_Full'),
                           ('compare','T2VA','Same_Latent_Compare')):
        source = json.loads((RESEARCH/'pilot-api-drafts'/f'{task}_native8.prompt.json').read_text(encoding='utf8')) if task else None
        graphs[name] = recipe(source,mode)
        paths[name] = PROJECT/f'examples/workflows/30-trt-vae/2026-09-10_H3_TRT_VAE_{name}_EXP.json'
    validation = deepcopy(graphs)
    for name in ('T2VA_Decoder','I2VA_Full','Same_Latent_Compare'):
        values = validation[name]['101' if name == 'Same_Latent_Compare' else '1']['inputs']
        values['decoder_engine'] = 'decoder-flex-fp16-db66c4fb89234e6f85b2f9b1d4d4f425'
        values['runtime_directory'] = str(RESEARCH/'trt-runtime-cu13-10.13.3.9.post1/site-packages')
    validation['I2VA_Full']['1']['inputs'].update(
        image_encoder_engine='encoder-t1-fp16-3b170cb8ccfd4c50be9269113b45f0d4',
        video_encoder_engine='encoder-fp16-2001120ef8c84a0f8adf685e8eaf50e2')
    expected = {'core':verify_core_source(CORE),'sources':source_snapshot(),'mode':'cpu-smoke','pilot_graphs':validation}
    root.mkdir(parents=True)
    server = OwnedServer(root,args.port,True)
    result = {'status':'incomplete','browser_used':False,'gpu_or_generation_queued':False}
    try:
        write_json(root/'paths.json',probe_resource_config(CORE,PROJECT))
        write_json(root/'expected.json',expected)
        server.start()
        wait_ready(server,lambda:None)
        history,_ = execute_graph(server,{
            '1':{'class_type':'T8ProgressiveEnvironmentAudit','inputs':{'expected_json':json.dumps(expected)}},
            '2':{'class_type':'PreviewAny','inputs':{'source':['1',0]}}},root/'environment',lambda:None,timeout=120)
        actual = preview_report(history,'2')
        if actual['device'] != 'cpu' or actual['cuda_initialized'] or actual['pid'] != server.process.pid:
            raise ValueError('Owned API server must remain CPU-only')
        info = server.request('GET','/object_info')
        write_json(root/'object-info.json',info)
        audits = {name:audit_candidate(graph,json.loads(paths[name].read_text(encoding='utf8')),info)
                  for name,graph in graphs.items()}
        check_history,_ = execute_graph(server,{
            '1':{'class_type':'MiniMaxH3TRTVAECheckEXPT8','inputs':{'runtime_directory':str(root/'missing-runtime')}},
            '2':{'class_type':'PreviewAny','inputs':{'source':['1',2]}}},root/'missing-runtime',lambda:None,timeout=45)
        missing = preview_report(check_history,'2')
        if missing['status'] != 'missing_runtime_files' or missing['gpu_initialized'] or missing['installed_or_downloaded']:
            raise ValueError('Actual missing-runtime API execution did not fail safely')
        queue = server.request('GET','/queue')
        if queue.get('queue_running') or queue.get('queue_pending'):
            raise ValueError('Unexpected queue activity')
        result.update(status='actual_CPU_API_five_workflows_and_optional_runtime_pass',environment=actual,
                      workflow_audits=audits,missing_runtime=missing,
                      limits='Generation/compile graphs validated only,not executed. Only CPU environment and static check executed. No browser open/save/export claim.')
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        server.stop()
        result['server_stop'] = server.stop_receipt
        write_json(root/'terminal.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('environment','missing_runtime')},indent=2))


if __name__ == '__main__':
    main()
