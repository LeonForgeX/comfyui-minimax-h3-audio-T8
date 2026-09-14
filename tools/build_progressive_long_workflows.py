"""Build unreviewed public-node candidates from actual Core object_info."""

from .run_progressive_native_chain import build_graph
from .build_dual_model_workflows import defaults
from .api_to_frontend_workflow import convert
from .frontend_workflow_compat import normalize_native_widget_inputs


NODE = 'MiniMaxH3ProgressiveLongVideoEXPT8'
REVIEWED_EAV_TAU = 8.0
REVIEWED_EAV_START = 0.15
REVIEWED_EAV_END = 0.90


def build_public_graph(info, chain_id, backend='kj-memory', relay=True, eav='disabled', tst='disabled'):
    graph = build_graph(chain_id, backend, relay, eav, tst)
    original = graph['8']['inputs']
    graph['12'] = {'class_type': 'MiniMaxH3ProgressiveSetupEXPT8', 'inputs': {
        'model': ['21', 0], 'model_hires': ['22', 0], 'steps': 8, 'shift_video': 12., 'shift_audio': 3.}}
    inputs = {**defaults(info[NODE]), **{key: original[key] for key in (
        'clip', 'video_vae', 'audio_vae', 'chain_id', 'global_prompt', 'upscaler_model', 'eav_mode')},
        'model': ['12', 0], 'model_hires': ['12', 1], 'sampler': ['12', 2], 'sigmas': ['12', 3]}
    if eav == 'apply_exp':
        inputs.update(
            eav_tau=REVIEWED_EAV_TAU,
            eav_start_video_progress=REVIEWED_EAV_START,
            eav_end_video_progress=REVIEWED_EAV_END,
        )
    if relay:
        inputs['prompt_relay_plan'] = original['prompt_relay_plan']
    if tst != 'disabled':
        for node_id, slot, target in [('13', 0, 'model'), ('14', 1, 'model_hires')]:
            graph[node_id] = {'class_type': 'MiniMaxH3TSTModelEXPT8', 'inputs': {
                'model': ['12', slot], 'sigmas': ['12', 3], 'mode': tst, 'tau': .2, 'max_workspace_mib': 2048}}
            inputs[target] = [node_id, 0]
    graph['8'] = {'class_type': NODE, 'inputs': inputs}
    graph['9']['inputs']['source'] = ['8', 2]
    graph['10']['inputs']['source'] = ['8', 1]
    return graph


def build_workflow(graph, info):
    if graph['8']['class_type'] != NODE:
        raise ValueError('Expected the public progressive long-video node')
    workflow = convert(graph, info, 'H3 Progressive 4+4 / UNREVIEWED 开发候选')
    normalize_native_widget_inputs(workflow)
    note_id = workflow['last_node_id'] + 1
    text = ('开发候选，未完成完整模型成片验收，不是已通过示例。\n'
            '两路底模→各自H3兼容LoRA链→每路一个后端→Progressive Setup。'
            'Setup给出两路MODEL、标准Euler和完整8步SIGMAS；长视频LOW填4就是4+4。\n'
            '独立TST接Setup之后，LOW/HIGH分别配置，均连接同一完整SIGMAS。'
            'EAV使用长视频节点内部开关，不能外接未知EAV包装。Sol遇Relay偏置可能回退。\n'
            '全局只写持续场景/人物；只说一次的台词写局部事件。手填百分比时选percent，'
            'joint_av_exp路由生成对白。锁源音频改用video_only_paper。\n'
            '8秒=192输出帧，两段接缝约5.17秒；Relay计划193帧。改时长同步改Relay覆盖长度。'
            '39帧是保存尾部容量；context22/39为实际使用量，不能把两者混为一谈。\n'
            '同一chain_id+输入不变+resume_existing=true可校验恢复。改模型/提示/时间/音频用新ID。'
            'false拒绝已有链，不删除文件。音频cosine_bridge是5ms音频交接，不是视频叠化。\n'
            '缺权重请自行选择已有合法文件，不自动下载。不要叠加Sage和Sol替换器。')
    if graph['8']['inputs'].get('eav_mode') == 'apply_exp':
        text += (
            '\nEAV已审候选使用tau=8、视频进度15%–90%，默认表现偏克制。'
            '希望更高动态时可小幅逐步提高eav_tau；不要误调TST自己的tau。'
            '数值越高越可能放大身份漂移、局部形变、闪烁或联合AV带来的间接声音变化，'
            '每次调整后都要重新检查完整画面和声音。g_hard_limit仍保持1.5。'
        )
    workflow['nodes'].append({'id': note_id, 'type': 'Note', 'pos': [0, -700], 'size': [760, 560],
        'flags': {}, 'order': len(workflow['nodes']), 'mode': 0, 'inputs': [], 'outputs': [],
        'properties': {}, 'widgets_values': [text], 'title': '接线与注意事项 / UNREVIEWED'})
    workflow['last_node_id'] = note_id
    return workflow
