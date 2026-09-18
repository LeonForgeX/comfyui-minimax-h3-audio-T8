"""Independent phase1 Avatar and native voice-reference candidates; no inference."""
from copy import deepcopy

from tools.api_to_frontend_workflow import convert
from tools.audit_progressive_workflows import audit_candidate
from tools.build_dual_model_workflows import defaults
from tools.build_fast_h3_v2_workflows import selected_frontend_schema


def graph(info, *, route, image, audio):
    if route not in {'avatar', 'avatar_preview', 'voice_neutral', 'voice_emotion'}:
        raise ValueError('Unknown independent route')
    is_avatar = route in {'avatar', 'avatar_preview'}
    common = {
        '1': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': 'minimax_h3_fl2va_int8_convrot.safetensors' if is_avatar else 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
            'weight_dtype': 'default'}},
        '2': {'class_type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced', 'inputs': {'model': ['1', 0],
            'lora_name': 'minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors', 'strength_model': 1. if is_avatar else 0.}},
        '3': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'type': 'minimax', 'device': 'default'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'}},
        '6': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '7': {'class_type': 'LoadAudio', 'inputs': {'audio': audio}},
    }
    if not is_avatar:
        common.pop('2')
    condition = {**defaults(info['MiniMaxH3AudioConditioningT8']), 'clip': ['3', 0],
        'video_vae': ['4', 0], 'audio_vae': ['5', 0], 'width': 512, 'height': 768, 'length': 73,
        'audio_mode': 'lock_source' if is_avatar else 'native',
        'task_type': 'I2VA' if is_avatar else 'Ref2VA', 'add_source_as_reference': False,
        'prompt_primary_audio_ordinal': 0}
    if is_avatar:
        common['18'] = {'class_type': 'MiniMaxH3AudioWindowT8', 'inputs': {
            'audio': ['7', 0], 'scene_start_seconds': 0., 'scene_duration_seconds': 73 / 24,
            'warmup_seconds': 0., 'cooldown_seconds': 0., 'ensure_minimum_context': False}}
        condition.update(first_frame=['6', 0], drive_audio=['18', 0], length=['18', 1],
            prompt='Stable medium close-up of the woman from the first frame. She speaks in synchronization with the supplied recording. Natural restrained facial movements, stable soft lighting, no scene cut, no subtitles.')
    else:
        emotion = 'S1 speaks gently with relief and restrained joy, natural intonation.' if route == 'voice_emotion' else 'S1 speaks calmly in a neutral conversational voice.'
        condition.update({'ref_images.ref_image_0': ['6', 0], 'ref_audios.ref_audio_0': ['7', 0],
            'prompt': 'subject_definitions:\n<Subject 1> is the woman in <Picture 1>. S1 uses the vocal identity/timbre reference in <Audio 1>; the reference is not the target dialogue.\nsummary:\nOne stable portrait shot, consistent face and clothing, soft fixed studio lighting.\ndetailed_description:\n' + emotion + ' S1 looks at the camera and says <d>[Mandarin]你终于回来了。</d> No additional speech, no subtitles.\noverall_soundscape:\nClear newly generated speech, quiet room tone, no music.'})
    common['8'] = {'class_type': 'MiniMaxH3AudioConditioningT8', 'inputs': condition}
    common['9'] = {'class_type': 'MiniMaxH3DualClockSamplerT8', 'inputs': {
        'model': ['2', 0] if is_avatar else ['1', 0], 'av_latent': ['8', 1], 'steps': 8 if is_avatar else 20,
        'shift_video': 12., 'shift_audio': 3., 'sampler_name': 'euler', 'scheduler': 'native_flow'}}
    common['10'] = {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['8', 0]}}
    if is_avatar:
        kind = 'MiniMaxH3AvatarProgressiveEXPT8'
        common['11'] = {'class_type': kind, 'inputs': {**defaults(info[kind]),
            'model': ['9', 0], 'positive': ['8', 0], 'negative': ['10', 0], 'av_latent': ['8', 1],
            'sampler': ['9', 1], 'sigmas': ['9', 2], 'low_evaluations': 4, 'low_scale': .5,
            'task': 'i2va', 'cfg': 1., 'seed': 20260918,
            'upscaler_model': 'minimax_h3_latent_upscaler_3d_fp16.safetensors'}}
    else:
        common['11'] = {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 20260918}}
        common['12'] = {'class_type': 'BasicGuider', 'inputs': {'model': ['9', 0], 'conditioning': ['8', 0]}}
        common['13'] = {'class_type': 'SamplerCustomAdvanced', 'inputs': {
            'noise': ['11', 0], 'guider': ['12', 0], 'sampler': ['9', 1], 'sigmas': ['9', 2], 'latent_image': ['8', 1]}}
    sampled = ['11', 0] if is_avatar else ['13', 0]
    common['14'] = {'class_type': 'MiniMaxH3AVDecodeT8', 'inputs': {'av_latent': sampled,
        'video_vae': ['4', 0], 'audio_vae': ['5', 0]}}
    # Avatar optionally delivers the real source recording, not decoded-VAE
    # audio. Native voice must deliver newly generated audio, never reference.
    common['15'] = {'class_type': 'CreateVideo', 'inputs': {'images': ['14', 0], 'fps': 24.,
        'audio': ['18', 0] if is_avatar else ['14', 1]}}
    common['16'] = {'class_type': 'SaveVideo', 'inputs': {'video': ['15', 0],
        'filename_prefix': 'video/T8_' + route, 'format': 'auto', 'codec': 'auto'}}
    common['17'] = {'class_type': 'MiniMaxH3AudioSourceExplanationT8', 'inputs': {'report_json': ['8', 5]}}
    if route == 'avatar_preview':
        kind = 'MiniMaxH3TAEH3SamplingPreviewEXPT8'
        common['101'] = {'class_type': kind, 'inputs': {**defaults(info[kind]),
            'model': ['9', 0], 'checkpoint': 'taeh3.safetensors', 'phase': 'low', 'latent_prefix': '7'}}
        common['11']['inputs']['model'] = ['101', 0]
    return common


def build(info, recipe, name):
    selected, schemas = selected_frontend_schema(recipe, info)
    workflow = convert(selected, schemas, name)
    is_avatar = any(n['class_type'] == 'MiniMaxH3AvatarProgressiveEXPT8' for n in selected.values())
    note = workflow['last_node_id'] + 1
    text = ('# 未验收开发候选，不是正式画质推荐\n\n'
        '图片示例是2:3纵向，目标512×768、LOW256×384，不拉伸。更换图片须保持自身比例及32像素对齐。'
        '独立单段73帧、24fps≈3.042秒；本图不演示长视频接缝。人物／声音文件换成自己的合法素材。\n\n'
        + ('Avatar：Audio Conditioning的drive_audio锁源录音、首帧I2VA → 新独立Avatar入口。'
           '现成录音编码进目标audio latent并audio mask=0参与LOW4＋原learned3D＋HIGH4采样。'
           '不是仅保存时贴音轨，不生成新台词／克隆音色。保存分支明确使用原录音；音频文件须恰好覆盖成片，'
           '本图已接Audio Window裁成73/24秒，关闭最小上下文扩展，过短则补齐。为了保留原录音，不要把生成audio替代源录音。'
           '可另接HIGH MODEL／LoRA；首期没有空间tile或新Avatar模型。GPU已获确认，成片仍待最后人审。'
           if is_avatar else
           '原生音色参考：人物接ref_images.ref_image_0；声音接ref_audios.ref_audio_0；Ref2VA、native，'
           'drive_audio/final_audio都留空。参考声音不是目标音轨，SaveVideo使用generated_audio。'
           '根据media_map_json核实Audio编号；混入视频声音后不能继续硬写Audio1。'
           '新台词只写<d>内；角色绑定与情绪表演写在<d>外，不保证逐字台词或声纹100%保持。'
           '基线使用匹配Ref2VA权重、Stock20／Euler／12与3，不加Turbo LoRA。'
           '两单段Stock20真实生成及完整AV审计已完成，声音身份／情绪／逐字仍待人审；'
           '不能把Stock20音色证据迁移为Turbo4+4证据。')
        + '\n\n不改旧图／默认／已验收双采、音频及接缝。不自动排队或下载，用户补丁保留；CPU通过不等于GPU质量通过。')
    workflow['nodes'].append(dict(id=note, type='MarkdownNote', title='用法与验收边界',
        pos=[0, -650], size=[1000, 600], flags={}, order=len(selected), mode=0,
        inputs=[], outputs=[], properties={}, widgets_values=[text]))
    if any(n['class_type'] == 'MiniMaxH3TAEH3SamplingPreviewEXPT8' for n in selected.values()):
        workflow['nodes'][-1]['widgets_values'][0] += ('\n\nTAEH3可选预览：Setup MODEL→预览→原Avatar入口；sampler/sigmas仍接同一个Setup。'
            '只观察实际LOW x0的连续开头前缀，不是整个视频／最终画质／声音。默认7 latent约22帧，缩小后12fps播放。'
            '不增加扩散步，解码／传输有开销；调低分辨率和频率可减少开销。模型放models/vae_approx/taeh3.safetensors。'
            '缺少tiny模型时关闭预览但原采样继续。面板取消仅请求当前绑定prompt，不清队列／关闭服务；旧Core没有按请求接口时不调用全局interrupt。'
            '关闭enabled精确MODEL恒等，旧图无需加此节点。缓存命中没有实时callback，不把旧画面当作新预览。')
        workflow['nodes'][-1]['widgets_values'][0] += ('\n独立新进程OFF/ON完整4+4已逐步及最终AV精确相同，'
            'ON完成3次真实tiny解码。旧同进程多次采样／恢复差异仍未定位；没有据此改旧math。')
    workflow['last_node_id'] = note
    workflow.setdefault('extra', {})['t8_candidate_status'] = 'CPU_interface_candidate_not_GPU_or_human_qualified'
    # This auditor currently knows simple widgets; autogrow sockets are
    # checked separately by native Core instead of inventing schemas.
    if is_avatar:
        audit_candidate(selected, workflow, schemas)
    return deepcopy(selected), workflow


def voice_long_graph(info, *, dual, image, audio):
    """Reuse existing native reference inputs; do not invent a voice-clone MODEL."""
    kind = 'MiniMaxH3DualModelLongVideoEXPT8' if dual else 'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'
    plan = 'MiniMaxH3PromptRelayPlanT8Advanced'
    global_prompt = ('subject_definitions:\n<Subject 1> is the woman in <Picture 1>. '
        'S1 uses the vocal identity/timbre reference in <Audio 1>; it is not the target dialogue.\n'
        'summary:\nOne stable portrait scene, consistent face and clothing, fixed soft studio lighting, '
        'restrained natural movement, no scene cut or subtitles.\noverall_soundscape:\n'
        'Clear newly generated Mandarin dialogue, quiet room tone, no music.')
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'type': 'minimax', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '6': {'class_type': 'LoadAudio', 'inputs': {'audio': audio}},
        '7': {'class_type': plan, 'inputs': {**defaults(info[plan]), 'global_prompt': global_prompt,
            'local_prompts': 'S1 calmly says <d>[Mandarin]你终于回来了。</d>\n'
                'S1 speaks gently with relief and restrained joy: <d>[Mandarin]这次别再走了。</d>\n'
                'S1 listens silently with a small smile. No additional speech.',
            'length': 193, 'timing_mode': 'percent', 'time_ranges': '0-35\n35-70\n70-100'}},
    }
    values = {**defaults(info[kind]), 'clip': ['2', 0], 'video_vae': ['3', 0], 'audio_vae': ['4', 0],
        'width': 512, 'height': 768, 'total_duration_seconds': 8., 'render_window_frames': 124,
        'context_frames': 22, 'global_prompt': '', 'segment_prompts_json': '',
        'prompt_relay_plan': ['7', 0], 'prompt_relay_mode': 'apply_exp', 'eav_mode': 'disabled',
        'audio_mode': 'native', 'task_type': 'Ref2VA', 'add_source_as_reference': False,
        'prompt_primary_audio_ordinal': 0, 'context_audio': 'video_and_audio',
        'ref_images.ref_image_0': ['5', 0], 'ref_audios.ref_audio_0': ['6', 0],
        'base_seed': 20260918, 'chain_id': 'voice_reference_' + ('dual20plus4' if dual else 'native20') + '_fresh',
        'filename_prefix': 'T8_Voice_Reference_' + ('Dual' if dual else 'Long')}
    if dual:
        graph['10'] = {'class_type': 'MiniMaxH3LowVRAMAttentionT8Advanced', 'inputs': {'model': ['1', 0], 'head_chunks': 4}}
        graph['11'] = {'class_type': 'MiniMaxH3ChunkFeedForwardT8Advanced', 'inputs': {'model': ['10', 0], 'chunks': 2, 'seq_threshold': 4096}}
        values.update(model_pass1=['11', 0], model_pass2=['11', 0], low_width=256, low_height=384,
            upscaler_model='minimax_h3_latent_upscaler_3d_fp16.safetensors', coarse_steps=20, refine_steps=4,
            second_audio_source='auto', second_audio_strength=0., color_match=True,
            low_context_source='accepted_picture_low_context_v1', video_context_mode='high_native_mask_ramp_exp',
            color_match_mode='bounded_motion_color_exp')
    else:
        values.update(model=['1', 0], steps=20, shift_video=12., shift_audio=3.,
            sampler_name='dual_clock_euler', scheduler='native_flow')
    graph['8'] = {'class_type': kind, 'inputs': values}
    graph['9'] = {'class_type': 'MiniMaxH3AudioSourceExplanationT8', 'inputs': {'report_json': ['8', 5]}}
    return graph
