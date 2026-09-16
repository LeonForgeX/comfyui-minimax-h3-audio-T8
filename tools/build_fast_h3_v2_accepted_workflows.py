"""Transfer six accepted control recipes without touching a user's canvas."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
CASES = [
    ('FastH3_V2_Trained_VSA_73f_h1c1_EXP', 'fasth3-v2-trained-t2va-gpu-v1', 'trained-memory', 'clip-01.mp4',
     '83c89a4bba1ebcec68dec33a406946cdfd9e07413f25f42415506da282e52e13'),
    ('FastH3_V2_Trained_VSA_73f_h4c2_EXP', 'fasth3-v2-trained-h4c2-gpu-v1', 'trained-memory', 'clip-02.mp4',
     '8142a11339b16fc7dc61228a5ba3bd84a46d820bd377915e106b66e51866f707'),
    ('FastH3_V2_Official_Comfy_Template_124f_EXP', 'fasth3-v2-official-template-124-headroom2-gpu-v1', 'official-template', 'clip-05.mp4',
     '85aca5cc13940c6984f3beaae894c57af938d5f0dc47f361a929a30c9938cdfa'),
    ('FastH3_V2_Dense_Relay_Dual_4plus4_8s_EXP', 'fasth3-v2-dense-relay-loop-8s-gpu-v2', 'dense-relay-loop', 'clip-06.mp4',
     '02884a07c8fd41048408d20e219eeea7343eb92e541fa624ec704bf8662fb7ae'),
    ('FastH3_V2_Trained_VSA_Dual_4plus4_Resume_8s_EXP', 'fasth3-v2-native-trained-resume-8s-headroom4-gpu-v1', 'trained-native-resume', 'clip-07.mp4',
     'bbce0048b154c2ca08c6c64be11116bf26cffa239b5cb3122175fa1fdb85156a'),
    ('FastH3_V2_Dense_Sol_Audio_Protected_73f_EXP', 'fasth3-v2-sol-audio-protected-gpu-v1', 'sol-audio-fix', 'clip-02.mp4',
     '3830e795479c9fc37f4581a6715a5c43dcb4ab79f1e22d109ed465b1a9d42c34'),
]


def native_sampler_graph(source):
    graph = deepcopy(source)
    probes = [k for k, n in graph.items() if n['class_type'].startswith('T8FastH3V2')]
    if not probes:
        return graph
    if probes != ['13'] or graph['13']['class_type'] != 'T8FastH3V2SamplerProbe':
        raise ValueError('Unknown diagnostic graph cannot be promoted')
    values = graph['13']['inputs']
    if set(values) != {'model', 'positive', 'av_latent', 'sampler', 'sigmas', 'seed'}:
        raise ValueError('Unknown diagnostic sampler contract')
    if '11' in graph or '12' in graph:
        raise ValueError('Native sampler node ID collision')
    graph['11'] = dict(class_type='RandomNoise', inputs=dict(noise_seed=values['seed']))
    graph['12'] = dict(class_type='BasicGuider', inputs=dict(model=values['model'], conditioning=values['positive']))
    graph['13'] = dict(class_type='SamplerCustomAdvanced', inputs=dict(
        noise=['11', 0], guider=['12', 0], sampler=values['sampler'], sigmas=values['sigmas'], latent_image=values['av_latent']))
    if graph.get('22', {}).get('inputs', {}).get('source') == ['13', 1]:
        graph.pop('22')
    assert all(graph[k] == v for k, v in source.items() if k not in {'13', '22'})
    if any(n['class_type'].startswith('T8FastH3V2') for n in graph.values()):
        raise ValueError('Diagnostic node escaped removal')
    return graph


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--core', type=Path, required=True)
    args = parser.parse_args()
    root, core = args.root.resolve(), args.core.resolve(strict=True)
    if root.exists() or not root.is_relative_to(PROJECT / 'artifacts'):
        raise ValueError('New task-owned artifact directory required')
    acceptance = json.loads((PROJECT / 'artifacts/sol-b-user-acceptance-20260917.json').read_text(encoding='utf8'))
    if acceptance.get('user_reply_verbatim') != 'B 正常，可以验收' or acceptance.get('media_sha256') != CASES[-1][-1]:
        raise ValueError('Bound repaired B user acceptance required')
    old_review_root = PROJECT / 'artifacts/fasth3-v2-final-combined-review-v1'
    old_delivery = json.loads((old_review_root / 'delivery.json').read_text(encoding='utf8'))
    old_feedback = Path('D:/Backup/Downloads/dual_topaz_r1_combined_human_review (6).json')
    assert sha(old_feedback) == '403b715e804af217fc41a5f5025de9475e2b35580bc0746b4b42f0c9ab303444'
    feedback = json.loads(old_feedback.read_text(encoding='utf8'))
    assert feedback['review_id'] == old_delivery['review_id']
    answers = {a['id']: a for a in feedback['answers']}
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(core), str(PROJECT), str(PROJECT / 'tools')]
    sys.argv = ['accepted-control-transfer', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    name = '_t8_accepted_workflow_schema'
    formal = core / 'custom_nodes/minimax-h3-audio-T8'
    spec = importlib.util.spec_from_file_location(name, formal / '__init__.py', submodule_search_locations=[str(formal)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    nodes = asyncio.run(module.comfy_entrypoint().get_node_list())
    assert len(nodes) == 339
    current = {c.define_schema().node_id: c.GET_NODE_INFO_V1() for c in nodes}
    from build_fast_h3_v2_workflows import selected_frontend_schema
    from api_to_frontend_workflow import convert
    from audit_progressive_workflows import audit_candidate
    root.mkdir()
    records = {}
    for case_name, directory, group, clip_name, media_sha in CASES:
        source_path = PROJECT / 'artifacts' / directory / 'generation/prompt.json'
        source = json.loads(source_path.read_text(encoding='utf8'))
        graph = native_sampler_graph(source)
        if group == 'sol-audio-fix':
            review_root = PROJECT / 'artifacts/fasth3-v2-sol-audio-review-v2'
            review_id = acceptance['review_id']
            ratings = {'画面与细节': '正常／可接受', '口型': '正常／可接受',
                       '音乐与人声': 'Explicit user clarification: B 正常，可以验收', '事件跟随与接缝': '不适用／无法判断'}
        else:
            review_root, review_id = old_review_root, old_delivery['review_id']
            answer = answers[group]
            assert media_sha in {m['sha256'] for m in answer['media']}
            assert len(answer['played_to_end']) == len(answer['media'])
            assert all(answer['ratings'][key] == '正常／可接受' for key in ['画面与细节', '音乐与人声', '口型'])
            ratings = answer['ratings']
        assert sha(review_root / 'public' / clip_name) == media_sha
        info_path = PROJECT / 'artifacts' / directory / 'object-info.json'
        if not info_path.is_file():
            info_path = PROJECT / 'artifacts/fasth3-v2-native-trained-interrupt-8s-headroom4-gpu-v2/object-info.json'
        info = json.loads(info_path.read_text(encoding='utf8'))
        info.update(current)
        missing = {n['class_type'] for n in graph.values()} - set(info)
        if missing:
            raise ValueError('Missing native schema: ' + str(missing))
        graph, info = selected_frontend_schema(graph, info)
        workflow = convert(graph, info, case_name)
        note_id = workflow['last_node_id'] + 1
        text = ('FastH3 V2 完整学生权重；本图转录已接受的固定样片配方，仍为EXP，不代表任意素材／LoRA／时长／显存均通过。\n\n'
                '不是旧EMA/Turbo加速LoRA；复用原生Qwen与音视频VAE。单MODEL用固定8步、CFG1、最终sigma0联合解码。'
                'trained / Dense / official template是不同配方，不可混称等价。73帧图与官方124帧图分别保留已审长度；单MODEL min_tokens=0是已审探针控制，不更改节点缺省12288。\n\n'
                'h4/c2训练图本机更慢且周期整卡占用更高，不作通用提速／省显存推荐。Sol仅在Dense、无Relay的已认证分支精确保留非video Q/KV；不做后期增音量。\n\n'
                '双MODEL为256×384→原有learned3D→512×768，两个裸MODEL各可加自己的普通LoRA；不要加旧EMA/Turbo。'
                'window124/context22，两段总8秒，约5.17秒是内部边界，不是两段各4秒。Relay图保留完整全局／局部／时间线；训练恢复图关闭Relay/EAV。\n\n'
                '首帧引用的是已审本地图，请自己放入ComfyUI/input或换自己的2:3参考，不拉伸；没有隐式下载。'
                '双MODEL执行前换新的chain_id；只有参数不变才resume。更换模型／LoRA／参考／代码需新链，不搬旧缓存。\n\n'
                '此图移除诊断观察器后保留同一Core RandomNoise→BasicGuider→SamplerCustomAdvanced执行连接；不是重新GPU生成／逐像素复现承诺。'
                '旧失败Sol A不保存成推荐图。模型与说明见docs/FAST_H3_V2_EXP.md。')
        workflow['nodes'].append(dict(id=note_id, type='MarkdownNote', title='FastH3 V2 · 通过配方与边界', pos=[0, -760],
            size=[1000, 690], flags={}, order=len(graph), mode=0, inputs=[], outputs=[], properties={}, widgets_values=[text]))
        workflow['last_node_id'] = note_id
        workflow['extra']['t8_bound_review'] = dict(status='accepted_in_this_review_scope', review_id=review_id,
            media_sha256=media_sha, ratings=ratings, not_universal_quality_claim=True,
            control_transfer='Exact accepted control values; diagnostic observer removed, native Core sampler unchanged; not a fresh GPU run')
        audit = audit_candidate(graph, workflow, info)
        api_path, frontend_path = root / (case_name + '.api.json'), root / (case_name + '.json')
        for path, data in [(api_path, graph), (frontend_path, workflow)]:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
        records[case_name] = dict(frontend_sha256=sha(frontend_path), api_sha256=sha(api_path), source_prompt_sha256=sha(source_path),
            media_sha256=media_sha, review_id=review_id, audit=audit, schema_source=str(info_path),
            current339_schema_overlay=True, native_sampler_transfer='13' in source and source['13']['class_type'] == 'T8FastH3V2SamplerProbe')
    assert not torch.cuda.is_initialized()
    (root / 'receipt.json').write_text(json.dumps(dict(status='six_bound_control_workflows_serialization_pass',
        cases=records, cuda_initialized=False, published=False, fresh_gpu_run=False), ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status='six_bound_control_workflows_serialization_pass', count=len(records), root=str(root)), ensure_ascii=False))


if __name__ == '__main__':
    main()
