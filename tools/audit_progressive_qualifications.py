"""Separate read-only postflight of a completed, predeclared qualification pair."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_progressive_qualifications import frozen_plan,audit_completed  # noqa: E402
from progressive_pilot_analysis import compare_runs  # noqa: E402
from run_progressive_pilot import write_json  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    def read(name):
        return json.loads((args.run_root/name).read_text(encoding='utf8'))
    expected,terminal=read('plan.json'),read('terminal.json')
    if terminal['status']!='two_serial_qualified_runs_human_pending' or len(terminal['completed'])!=2:
        raise ValueError('Pair is not complete; cannot substitute a partial result')
    qualification=expected['jobs'][0]['qualification_case']
    plan=frozen_plan(qualification)
    if expected['jobs']!=plan:
        raise ValueError('Pair plan differs from the declared qualification')
    runs=[audit_completed(args.run_root/item['folder'],item,expected) for item in plan]
    comparison=compare_runs(*runs)
    if comparison!=read('comparison.json'):
        raise ValueError('Pair comparison changed or is not supported by raw evidence')
    write_json(args.report,{'status':'two_runs_eight_queues_independent_postflight_pass',
                            'qualification':qualification,'comparison':comparison,
                            'attention':[run['reports']['sampler'].get('attention_audit') for run in runs],
                            'human_review':'pending'})
    print(json.dumps({'status':'postflight_pass','qualification':qualification,'saved_fraction':comparison['saved_fraction']}))


if __name__=='__main__':
    main()
