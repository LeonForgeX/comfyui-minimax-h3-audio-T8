"""One predeclared long32 or Sage pair; two owned GPU runs, strictly serial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_progressive_pilot import PROJECT, CORE, RESEARCH, instrument_recipe, source_snapshot, write_json  # noqa: E402
from progressive_qualification import QUALIFICATION_CASES, qualification_recipe  # noqa: E402
from progressive_probe_control import SerialProbeLease, ResourceGuard  # noqa: E402
from progressive_pilot_analysis import load_run, compare_runs  # noqa: E402
from audit_progressive_warm_pair import queue_evidence  # noqa: E402
from vdn_probe_environment import verify_core_source  # noqa: E402


def frozen_plan(qualification):
    jobs = []
    for route in ('native8','progressive6plus2'):
        case = 'T2VA_'+route
        graph = instrument_recipe(json.loads((RESEARCH/'pilot-api-drafts'/f'{case}.prompt.json').read_text(encoding='utf8')))
        jobs.append({'case':case,'qualification_case':qualification,'folder':route,
                     'graph':qualification_recipe(graph,case,qualification)})
    return jobs


def audit_completed(folder,item,expected):
    run = load_run(folder)
    if run['graph'] != item['graph'] or run['terminal'].get('qualification_case') != item['qualification_case']:
        raise ValueError('Actual qualification graph differs from the frozen case')
    if any(run['environment'][key] != expected[key] for key in ('core','sources')):
        raise ValueError('Qualification source/Core changed')
    histories = []
    for part in ('environment','allocator-begin','generation','allocator-finish'):
        graph,_,history = queue_evidence(folder/part)
        histories.append(history)
        if part=='environment' and json.loads(graph['1']['inputs']['expected_json']) != run['environment']:
            raise ValueError('Environment queue did not receive the recorded identity')
        if part.startswith('allocator') and graph['1']['inputs'] != {
                'run_id':folder.name,'action':part.split('-')[1],'device_type':'cuda','expected_pid':run['live']['pid']}:
            raise ValueError('Allocator queue has wrong owner/action')
    if len({h['prompt'][1] for h in histories}) != 4 or any(a['prompt'][0]>=b['prompt'][0] for a,b in zip(histories,histories[1:])):
        raise ValueError('Qualification queue order/uniqueness differs')
    guard = ResourceGuard()
    startup = json.loads((folder/'startup-resources.json').read_text())
    guard.observe(startup['sample'],startup=True)
    if guard.report() != startup['guard'] or guard.reason:
        raise ValueError('Startup resources not qualified')
    for line in (folder/'resources.jsonl').read_text().splitlines():
        guard.observe(json.loads(line))
    if guard.report() != run['terminal']['resource_guard'] or guard.reason:
        raise ValueError('Raw resource observations differ from claimed success')
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('plan','gpu'),default='plan')
    parser.add_argument('--qualification',choices=QUALIFICATION_CASES,required=True)
    parser.add_argument('--headroom-gib',type=int,choices=(0,2),default=0)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--identities',type=Path)
    parser.add_argument('--ffmpeg',type=Path)
    parser.add_argument('--ffprobe',type=Path)
    args = parser.parse_args()
    if args.headroom_gib and args.qualification != 'long32':
        raise ValueError('Additional headroom is only for the diagnosed long32 qualification')
    plan = frozen_plan(args.qualification)
    if args.mode == 'plan':
        print(json.dumps({'mode':'plan_only_no_GPU','jobs':plan}))
        return
    if not all((args.identities,args.ffmpeg,args.ffprobe)):
        raise ValueError('GPU needs explicit source identities and media executables')
    root = args.run_root.resolve()
    if root.exists() or root==RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New dedicated qualification directory required')
    with SerialProbeLease(RESEARCH/'qualification-controller.lock'):
        root.mkdir(parents=True)
        expected = {'core':verify_core_source(CORE),'sources':source_snapshot(),'jobs':plan,'headroom_gib':args.headroom_gib}
        write_json(root/'plan.json',expected)
        terminal,child,runs = {'status':'incomplete','completed':[],'human_review':'pending'},None,[]
        try:
            for item in plan:
                if (root/'STOP').exists():
                    raise RuntimeError('Explicit stop between jobs')
                if source_snapshot()!=expected['sources'] or verify_core_source(CORE)!=expected['core']:
                    raise ValueError('Source changed during qualification pair')
                folder = root/item['folder']
                command = [sys.executable,'-X','utf8',str(PROJECT/'tools/run_progressive_pilot.py'),'--mode','gpu',
                           '--case',item['case'],'--qualification-case',args.qualification,'--run-root',str(folder),
                           '--identities',str(args.identities.resolve()),'--ffmpeg',str(args.ffmpeg.resolve()),'--ffprobe',str(args.ffprobe.resolve())]
                if args.headroom_gib:
                    command += ['--headroom-gib',str(args.headroom_gib)]
                print('Starting serial qualification '+args.qualification+'/'+item['folder'],flush=True)
                child = subprocess.Popen(command,cwd=PROJECT)
                code = child.wait()
                if code:
                    raise RuntimeError(f'Qualification child exit{code}; stop, no retry/next job')
                run = audit_completed(folder,item,expected)
                runs.append(run)
                terminal['completed'].append({'folder':item['folder'],'seconds':run['timing']['elapsed_seconds'],
                                              'media_sha256':run['media']['file']['sha256']})
                write_json(root/f'checkpoint-{len(runs)}.json',terminal)
            comparison = compare_runs(*runs)
            write_json(root/'comparison.json',comparison)
            terminal['status'] = 'two_serial_qualified_runs_human_pending'
        except BaseException as error:
            terminal.update(status='failed_or_interrupted',error=f'{type(error).__name__}: {error}')
            if child is not None and child.poll() is None:
                terminal['bounded_child_still_owns_cleanup_pid'] = child.pid
            raise
        finally:
            write_json(root/'terminal.json',terminal)
            print(json.dumps(terminal),flush=True)


if __name__=='__main__':
    main()
