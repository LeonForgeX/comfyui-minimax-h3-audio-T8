"""Schema-driven dual-MODEL workflow candidates; does not load models or queue."""
from copy import deepcopy
import uuid

from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs


NODE = 'MiniMaxH3DualModelLongVideoEXPT8'
LORA = 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'
PLAN = 'MiniMaxH3PromptRelayPlanT8Advanced'
PROMPT = ('A continuous cinematic medium shot of a woman in an old concert hall, warm stable lighting. '
          'She looks toward the camera, speaks one short Mandarin sentence: <d>你在哪里</d>, '
          'then listens quietly. Soft classical piano and strings play in the background. '
          'Clean clear speech, natural mouth movements, no subtitles, no additional speech.')
SCENE_PROMPT = ('A continuous cinematic medium shot of a woman in an old concert hall, warm stable lighting. '
                'Soft classical piano and strings play quietly in the background. '
                'Natural motion, consistent face and clothing, no subtitles or screen text.')
RELAY_EVENTS = ('She looks toward the camera and says one Mandarin sentence: <d>你在哪里</d>. '
                'Natural mouth movements, clear speech.\n'
                'She is silent, listening quietly, and turns her head toward the open door. No speech.\n'
                'Without speaking, she walks slowly toward the open door.\n'
                'She pauses at the door and looks into the distance. Only instrumental music, no speech.')


def defaults(info):
    result = {}
    for name, spec in info['input']['required'].items():
        if isinstance(spec[0], list):
            result[name] = deepcopy(spec[1].get('default', spec[0][0]))
        elif len(spec) > 1 and 'default' in spec[1]:
            result[name] = deepcopy(spec[1]['default'])
    return result


def build_prompt(info, *, relay=False):
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': 'minimax_h3_fl2va_int8_convrot.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': LORA, 'inputs': {'model': ['1', 0],
            'lora_name': 'minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors', 'strength_model': 1.}},
        '3': {'class_type': LORA, 'inputs': {'model': ['1', 0],
            'lora_name': 'minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors', 'strength_model': 1.}},
        '4': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'type': 'minimax', 'device': 'default'}},
        '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        '6': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'}},
    }
    inputs = {**defaults(info[NODE]), 'model_pass1': ['2', 0], 'model_pass2': ['3', 0],
        'clip': ['4', 0], 'video_vae': ['5', 0], 'audio_vae': ['6', 0],
        'low_width': 512, 'low_height': 256, 'width': 1024, 'height': 512,
        'upscaler_model': 'minimax_h3_latent_upscaler_3d_fp16.safetensors',
        'chain_id': 'dual_model_4plus4_relay' if relay else 'dual_model_4plus4_plain',
        'global_prompt': '' if relay else SCENE_PROMPT + ' She listens to the music without speaking.', 'total_duration_seconds': 8.,
        'coarse_steps': 4, 'refine_steps': 4, 'eav_mode': 'disabled', 'color_match': True,
        'video_context_mode': 'high_native_mask_exp',
        'prompt_relay_mode': 'apply_exp' if relay else 'disabled',
        'second_audio_source': 'auto', 'second_audio_strength': 0.,
        'filename_prefix': 'H3_Dual_Model_4plus4'}
    if relay:
        graph['7'] = {'class_type': PLAN, 'inputs': {**defaults(info[PLAN]),
            'global_prompt': SCENE_PROMPT, 'local_prompts': RELAY_EVENTS,
            'length': 193, 'timing_mode': 'percent', 'time_ranges': '0-15\n15-40\n40-75\n75-100'}}
        inputs['prompt_relay_plan'] = ['7', 0]
    graph['8'] = {'class_type': NODE, 'inputs': inputs}
    # The output node itself creates/validates the composed file and preview.
    return graph


def build_workflow(info, *, relay=False):
    graph = build_prompt(info, relay=relay)
    workflow = convert(graph, info, 'H3 双模型 4 + 潜空间放大 + 4 / EXP')
    normalize_native_widget_inputs(workflow)
    positions = {'1': [0, 0], '2': [450, 0], '3': [450, 270], '4': [0, 320],
                 '5': [0, 580], '6': [0, 780], '7': [960, -760], '8': [960, 0]}
    titles = {'1': '原生 H3 底模（两路共享底模，独立 LoRA）', '2': '一采 LoRA · 可独立更换',
              '3': '二采 LoRA · 可独立更换', '4': 'H3 文本编码器', '5': '视频 VAE',
              '6': '音频 VAE', '7': '全片时间线 · 每行一个事件',
              '8': '串行内循环：小画幅 4 步 → 学习放大 → 大画幅 4 步'}
    for key, node in zip(graph, workflow['nodes']):
        node.update(title=titles[key], pos=positions[key])
        if key == '8':
            node['size'] = [780, 1700]
        if key == '7':
            node['size'] = [780, 710]
    note_id = workflow['last_node_id'] + 1
    text = ('# 双 MODEL 内循环 · 未发布候选\n\n'
            '本模板默认两段共8秒，接缝约5.17秒。已评8秒KJ+Relay样片的画面、声音、口型和接缝可接受；不代表所有素材或长片质量。'
            'video_context_mode=high_native_mask_exp锁定二采已知视频重叠区域；旧工作流缺省仍为reference_only，不强改旧任务。\n\n'
            '一采和二采是两个独立 MODEL 插口。默认共用原生 H3 底模，各接自己的新版 EMA B LoRA；'
            '可以分别换 LoRA 或接兼容的另一原生 H3 底模。VDN 不属于这个 4+4 模板。\n\n'
            'low_width / low_height 是一采尺寸，width / height 是最终尺寸。这里 512×256 → 1024×512，'
            '不是像素域成片超分。放大模型放 models/latent_upscale_models/。\n\n'
            '音频auto：4+4由二采继续完成声音；完整Stock20才锁定一采声音。4+4 时 EAV 必须关闭。不要把短片测试当成长片验收。'
            'chain_id 标识断点任务；改变模型、LoRA、提示词或尺寸后用新的 chain_id，不手动混用缓存。\n\n'
            'KJ Sage / Sol 可以分别接在两路 LoRA 后，但只使用文档明确验证的组合；'
            '当前图不预接未完成整片验证的加速器。Relay 的浮点时间偏置不会为了提速被丢弃。\n\n'
            + ('Relay 每行一个事件；只说一次的台词只写在对应局部事件，不能放进全局提示词让每段重复收到。'
               '改变总时长也要同步 Plan 的 length。'
               if relay else '基础图用全局场景提示词，默认只有音乐无对白；它会用于每段。'
               '只说一次的台词或多镜头脚本请使用独立 Relay 时间线图。'))
    workflow['nodes'].append({'id': note_id, 'type': 'MarkdownNote', 'title': '使用说明与当前验证范围',
        'pos': [1820, 0], 'size': [700, 670], 'flags': {}, 'order': len(graph), 'mode': 0,
        'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [text]})
    workflow['last_node_id'] = note_id
    workflow['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, f't8:dual-model-loop:20260911:{relay}'))
    workflow['extra']['dual_model_delivery_status'] = 'unreleased_scoped_8s_human_accepted_not_universal_quality'
    return workflow
