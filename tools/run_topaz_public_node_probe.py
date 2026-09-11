"""Actual public Topaz nodes on an isolated output directory; GPU strictly serial."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'install', 'definitions', 'data', 'source', 'output', 'lease'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scale', choices=['1x', '2x', '4x'], required=True)
    parser.add_argument('--size-mode', choices=['scale', 'target_dimensions'], default='scale')
    parser.add_argument('--target-width', type=int, default=0)
    parser.add_argument('--target-height', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new isolated evidence directory')
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.core), str(project), str(project / 'tools')]
    sys.argv = ['topaz-public-probe', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import folder_paths
    import torch
    from comfy_api.latest import InputImpl
    from progressive_probe_control import SerialProbeLease

    spec = importlib.util.spec_from_file_location('topaz_public_probe_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    nodes = {cls.define_schema().node_id: cls for cls in classes}
    args.output.mkdir(parents=True)
    media_output = args.output.resolve() / 'output'
    media_output.mkdir()
    folder_paths.set_output_directory(str(media_output))
    source = InputImpl.VideoFromFile(str(args.source.resolve(strict=True)))
    with SerialProbeLease(args.lease):
        environment = nodes['MiniMaxH3TopazEnvironmentEXPT8'].execute(
            str(args.install), str(args.definitions), str(args.data))
        result = nodes['MiniMaxH3TopazVideoEXPT8'].execute(
            environment.result[0], source, 'iris-3', args.scale, .8, '{}',
            size_mode=args.size_mode, target_width=args.target_width, target_height=args.target_height).result
    report = json.loads(result[3])
    if result[1] is not source or result[2] != report['output']['path']:
        raise AssertionError('Public outputs lost source identity or published path')
    receipt = {'status': 'public_nodes_real_execution_media_pass_human_pending',
        'registered_nodes': len(classes), 'parent_cuda_initialized': torch.cuda.is_initialized(),
        'automatic_dimensions': args.size_mode == 'scale', 'scale': args.scale,
        'size_mode': args.size_mode, 'target_width': args.target_width, 'target_height': args.target_height,
        'source_object_preserved': True, 'result': report}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'result'}))


if __name__ == '__main__':
    main()
