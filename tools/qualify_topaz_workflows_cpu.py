"""Build native-schema Topaz workflows and validate their API without execution/UI."""
import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.output.exists():
        raise ValueError('Use a new qualification directory')
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = ['topaz-workflows', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import folder_paths
    folder_paths.set_input_directory(str(args.source.resolve(strict=True).parent))
    import nodes
    import execution
    import torch
    from comfy_extras.nodes_video import LoadVideo
    spec = importlib.util.spec_from_file_location('topaz_workflow_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    for cls in [*classes, LoadVideo]:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    from tools.api_to_frontend_workflow import convert
    from tools.frontend_workflow_compat import normalize_native_widget_inputs
    from tools.audit_progressive_workflows import audit_candidate
    names = ('MiniMaxH3TopazEnvironmentEXPT8', 'MiniMaxH3TopazVideoEXPT8', 'LoadVideo')
    info = json.loads(json.dumps({name: nodes.NODE_CLASS_MAPPINGS[name].GET_NODE_INFO_V1() for name in names}))
    environment = {'class_type': names[0], 'inputs': {'install_directory': '',
        'model_definitions_directory': '', 'model_data_directory': ''}}
    graphs = {'Environment': {'1': environment}, 'Video': {'1': environment,
        '2': {'class_type': 'LoadVideo', 'inputs': {'file': args.source.name}},
        '3': {'class_type': names[1], 'inputs': {'topaz_runtime': ['1', 0], 'source_video': ['2', 0],
            'model_id': 'iris-3', 'scale': '2x', 'vram_fraction': .8, 'parameters_json': '{}'}}}}
    args.output.mkdir(parents=True)
    report = {'nodes': len(classes), 'cuda_initialized': torch.cuda.is_initialized(),
        'scope': 'actual Core schema/API only; no environment execution, queue or browser', 'cases': {}}
    # Validate the saved API with the new optional fields entirely absent.
    legacy = asyncio.run(execution.validate_prompt('topaz-legacy-video', graphs['Video'], None))
    if not legacy[0]:
        raise ValueError(legacy)
    report['legacy_missing_dimensions_validation'] = legacy
    graphs['Video']['3']['inputs'].update(size_mode='scale', target_width=0, target_height=0)
    custom = deepcopy(graphs['Video'])
    custom['3']['inputs'].update(size_mode='target_dimensions', target_width=1536, target_height=768)
    valid_custom = asyncio.run(execution.validate_prompt('topaz-custom-video', custom, None))
    if not valid_custom[0]:
        raise ValueError(valid_custom)
    custom_workflow = convert(custom, info, 'Official Topaz custom size audit')
    normalize_native_widget_inputs(custom_workflow)
    report['custom_dimensions'] = {'validation': valid_custom,
        'serialization': audit_candidate(custom, custom_workflow, info)}
    for name, graph in graphs.items():
        valid = asyncio.run(execution.validate_prompt('topaz-' + name, graph, None))
        if not valid[0]:
            raise ValueError(valid)
        workflow = convert(graph, info, 'Official Topaz ' + name + ' EXP')
        normalize_native_widget_inputs(workflow)
        audit = audit_candidate(graph, workflow, info)
        positions = {'1': [0, 0], '2': [0, 430], '3': [620, 0]}
        for key, node in zip(graph, workflow['nodes']):
            node['pos'] = positions[key]
            node['size'] = [520, 370]
        note_id = workflow['last_node_id'] + 1
        workflow['nodes'].append({'id': note_id, 'type': 'MarkdownNote', 'pos': [620, 470],
            'size': [700, 430], 'flags': {}, 'order': len(graph), 'mode': 0, 'inputs': [], 'outputs': [],
            'properties': {}, 'title': '先准备正式安装与对应模型 / Read first',
            'widgets_values': ['填写自己的正式 Topaz 程序、模型定义与数据目录。环境节点只检查，不下载、不增强。'
                'VIDEO图请选择已保存的原生视频文件。2x需要对应正式2x权重，缺少会明确提示。'
                'Iris1x/2x短片机械验证通过，画质待审；4x与星光未验收。'
                '高级size_mode默认scale；target_dimensions按指定宽高输出，忽略倍率。'
                '自定义尺寸先经Topaz增强，再按原比例调整到目标宽高；范围1至4倍、偶数32至8192。'
                '输出为无损MOV/MKV母版，原音频包和解码PCM均检查；浏览器不一定能播，可后接原生Save Video转H.264。'
                '不插帧，不改变原音轨，不使用star2.6；不会自动安装或处理授权。参数说明见docs/TOPAZ_EXP.md。']})
        workflow['last_node_id'] = note_id
        report['cases'][name] = {'validation': valid, 'serialization': audit}
        for suffix, payload in (('prompt.json', graph), ('workflow.json', workflow)):
            (args.output / (name + '.' + suffix)).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
