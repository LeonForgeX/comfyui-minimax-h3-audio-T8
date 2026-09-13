"""Prepared generation recipes. No fixed clip playback or model downloads."""
import uuid

from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs


def build_prompt(*, route, bundle_path='', lease_path=''):
    if route not in ('tao5s', 'ltx_refine'):
        raise ValueError('Unknown prepared recipe')
    return {
        '1': {'class_type': 'MiniMaxH3PreparedGenerationBundleEXPT8',
            'inputs': {'prepared_bundle_path': bundle_path}},
        '2': {'class_type': 'MiniMaxH3PreparedVideoEXPT8', 'inputs': {
            'prepared_bundle': ['1', 0], 'noise_seed': 8301,
            'chain_id': f'{route}_prepared_trial_01', 'resume_existing': True,
            'serial_lease_path': lease_path}},
    }


def build_workflow(info, *, route, **paths):
    graph = build_prompt(route=route, **paths)
    workflow = convert(graph, info, f'H3 Prepared Generation / {route} / EXP')
    normalize_native_widget_inputs(workflow)
    for index, node in enumerate(workflow['nodes']):
        node['pos'] = [index * 700, 0]
        node['size'] = [620, 250 if index == 0 else 650]
        node['title'] = ('1. 读取准备清单 · 不是普通工作流JSON' if index == 0
            else '2. 新生成 / 严格恢复 → 独立解码 → 原文件预览')
    note = ('## 实验入口，先完成准备步骤\n\n'
        '这是可生成新latent再解码的接线，不是读取固定旧片的播放图。准备工具、实际清单迁移和'
        '真实Core缓存贯通、原生浏览器/API及实际解包检查已有通过证据；新封装worker未重跑GPU。'
        '指定LTX和Tao新对白恢复片已通过画面、运动及音频口型人审；Tao原生成收尾RAM保护失败仍保留，'
        '媒体接受不等于新封装worker完整GPU可靠性通过。\n\n'
        '1. prepared_bundle_path 填本机准备JSON绝对路径。换提示词必须同时准备匹配文本与音频/AV条件，'
        '不能只改文件名或JSON里的prompt。\n'
        '2. serial_lease_path 必须与其他研究入口共用同一个GPU锁文件；不得换锁绕过占用。Windows单GPU实验。\n'
        '3. noise_seed 控制视频噪声。改seed、输入、模型或代码后换chain_id。'
        'resume_existing=true仅复用全身份一致且逐文件哈希通过的阶段；解码失败只补解码。'
        'false不是覆盖旧链，仍须换新chain_id。\n'
        '4. 生成进程全部退出后才加载VAE，保留内存和GPU保护线。不要同时排其他GPU任务。\n'
        '5. 输出节点已保存并直接预览原MP4，无需另接SaveVideo重新编码音频。'
        'report_json区分新生成、缓存恢复与经过证据核对的旧样片迁移。\n\n'
        + ('Tao：复用匹配Base10 teacher与文本，原生单请求5秒，864×480。'
           '视频种子可变；教师音频种子由清单绑定，不因video seed改变。'
           '需要现有官方底模及Tao适配器，不是自动下载或重新生成teacher。'
           if route == 'tao5s' else
           'LTX：输入必须已是normalized LTX AV且没有参考前缀，使用匹配post-connector文本缓存。'
           '固定原生3次Euler更新/LoRA0.8，原AAC单独保留，LTX生成音频不用于交付。'
           'CFR24/8n+1帧/≤8秒；目前真实GPU证据只有73帧2048×1024。'
           '不要把普通H3 latent直接送入此图或靠改shape冒充转换。')
        + '\n\n改动不会触发下载/安装。不保证任意模型、任意尺寸或画质；新候选仍需集中审片，不重复审已接受片。'
          '详见 docs/PREPARED_GENERATION_INTEGRATION_EXP.md。')
    nid = workflow['last_node_id'] + 1
    workflow['nodes'].append({'id': nid, 'type': 'MarkdownNote', 'title': '必读：准备输入与缓存边界',
        'pos': [1400, 0], 'size': [850, 850], 'flags': {}, 'order': 2, 'mode': 0,
        'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [note]})
    workflow['last_node_id'] = nid
    workflow['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 't8:prepared-generation:20260913:' + route))
    workflow['extra']['prepared_route'] = route
    workflow['extra']['delivery_status'] = 'candidate_integration_pending'
    return workflow
