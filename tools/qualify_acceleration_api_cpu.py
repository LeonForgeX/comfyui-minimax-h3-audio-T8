"""Headless CPU API/schema qualification; never opens or controls a browser."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_pilot import (CORE, PROJECT, RESEARCH, OwnedServer, execute_graph,  # noqa: E402
    preview_report, source_snapshot, wait_ready, write_json)
from progressive_probe_control import file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402
from audit_progressive_workflows import audit_candidate  # noqa: E402


def run(root, port):
    root = Path(root).resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New isolated research directory required')
    graphs = {task: json.loads((RESEARCH/'pilot-api-drafts'/f'{task}_progressive6plus2.prompt.json').read_text())
              for task in ('T2VA','I2VA')}
    fi = {'1':{'class_type':'LoadVideo','inputs':{'file':'请选择你的原视频.mp4'}},
          '2':{'class_type':'MiniMaxH3DLSSFrameInterpolationEXPT8','inputs':{
              'source_video':['1',0],'runtime_directory':'','cut_frames':'','timeout_seconds':600}}}
    source = json.loads((RESEARCH/'dlss-fi-file-probe-20260910-v1/source.json').read_text())['file']
    if file_identity(source['path']) != source:
        raise ValueError('FI test source changed')
    validation_graphs = {**graphs, 'FI':json.loads(json.dumps(fi))}
    validation_graphs['FI']['1']['inputs']['file'] = 'fi-source.mp4 [output]'
    expected = {'core':verify_core_source(CORE), 'sources':source_snapshot(), 'mode':'cpu-smoke',
                'pilot_graphs':validation_graphs}
    for p in (PROJECT/'dlss_fi_backend').glob('*.py'):
        expected['sources'][p.relative_to(PROJECT).as_posix()] = file_identity(p)['sha256']
    root.mkdir(parents=True)
    server = OwnedServer(root,port,True)
    result = {'status':'incomplete','browser_used':False,'generation_queued':False}
    try:
        write_json(root/'paths.json',probe_resource_config(CORE,PROJECT))
        write_json(root/'expected.json',expected)
        server.start()
        wait_ready(server,lambda:None)
        shutil.copyfile(source['path'], root/'output/fi-source.mp4')
        if file_identity(root/'output/fi-source.mp4')['sha256'] != source['sha256']:
            raise ValueError('Scoped API fixture copy differs')
        history,_ = execute_graph(server,{
            '1':{'class_type':'T8ProgressiveEnvironmentAudit','inputs':{'expected_json':json.dumps(expected)}},
            '2':{'class_type':'PreviewAny','inputs':{'source':['1',0]}}},root/'environment',lambda:None,timeout=90)
        actual = preview_report(history,'2')
        if actual['device'] != 'cpu' or actual['cuda_initialized'] or actual['pid'] != server.process.pid:
            raise ValueError('Owned server did not remain CPU-only')
        info = server.request('GET','/object_info')
        write_json(root/'object-info.json',info)
        paths = {task:PROJECT/f'examples/workflows/28-progressive-sampling/2026-09-09_H3_Progressive_{task}_6plus2_EXP.json'
                 for task in ('T2VA','I2VA')}
        paths['FI'] = PROJECT/'examples/workflows/29-dlss-fi/2026-09-10_H3_DLSS_FI_File_2x_EXP.json'
        reports = {name:audit_candidate(graph,json.loads(paths[name].read_text(encoding='utf8')),info)
                   for name,graph in {**graphs,'FI':fi}.items()}
        if server.request('GET','/queue').get('queue_running') or server.request('GET','/queue').get('queue_pending'):
            raise ValueError('Unexpected queue activity')
        result.update(status='actual_CPU_API_validation_and_three_workflow_contracts_pass', environment=actual,
                      workflows=reports, workflow_files={k:file_identity(p) for k,p in paths.items()},
                      scope='headless_API_validation_not_browser_open_save_export')
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        server.stop()
        result['server_stop'] = server.stop_receipt
        write_json(root/'terminal.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8199)
    args = parser.parse_args()
    print(json.dumps(run(args.root,args.port),ensure_ascii=False))
