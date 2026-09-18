"""Build and validate current formal candidates in actual Core, no queue/UI/GPU."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--joint-only', action='store_true', help='Qualify one extra explicit native voice joint-AV candidate without replacing the six original graphs')
    selection.add_argument('--long-joint-only', action='store_true', help='Qualify one extra single-model joint-AV candidate without replacing old graphs')
    selection.add_argument('--long-window-only', action='store_true', help='Qualify an explicit window-text candidate; preserve existing graphs and timeline')
    selection.add_argument('--dialogue-owner-only', action='store_true', help='Qualify a separate dialogue event-start owner experiment; never overwrite old graphs')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('New owned artifact directory required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = ['avatar-voice-workflows-cpu', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import execution
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('avatar_voice_qualification', project / '__init__.py',
                                                submodule_search_locations=[str(project)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    from tools.build_avatar_voice_workflows import build, graph, voice_long_graph
    from tools.build_fast_h3_v2_workflows import selected_frontend_schema

    def saved_api(workflow, schemas):
        edges = {edge[0]: edge for edge in workflow['links']}
        result = {}
        for node in workflow['nodes']:
            if node['type'] == 'MarkdownNote':
                continue
            values, inputs = iter(node['widgets_values']), {}
            specs = schemas[node['type']]['input']
            for section in ('required', 'optional'):
                for name, spec in specs.get(section, {}).items():
                    options = spec[1] if len(spec) > 1 else {}
                    if not options.get('forceInput') and (isinstance(spec[0], list) or spec[0] in {'INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'}):
                        inputs[name] = next(values)
                        if options.get('control_after_generate', name in {'seed', 'noise_seed'}):
                            if next(values) != 'fixed':
                                raise ValueError('Unexpected seed widget')
            leftovers = list(values)
            if leftovers:
                raise ValueError(f'Unaccounted frontend widgets: {node["type"]}: {leftovers}')
            for pin in node['inputs']:
                if pin.get('link') is not None:
                    edge = edges[pin['link']]
                    inputs[pin['name']] = [str(edge[1]), edge[2]]
            result[str(node['id'])] = dict(class_type=node['type'], inputs=inputs)
        return result

    async def validate():
        for name in ('nodes_custom_sampler.py', 'nodes_video.py', 'nodes_audio.py'):
            if not await nodes.load_custom_node(str(args.core / 'comfy_extras' / name), module_parent='comfy_extras'):
                raise RuntimeError('Native node import failed: ' + name)
        own = await pkg.comfy_entrypoint().get_node_list()
        previous_root = project / 'artifacts/h16-source-release-20260917/package/unpacked/minimax-h3-audio-T8'
        previous_spec = importlib.util.spec_from_file_location('five_track_previous_343', previous_root / '__init__.py',
                                                              submodule_search_locations=[str(previous_root)])
        previous_pkg = importlib.util.module_from_spec(previous_spec)
        sys.modules[previous_spec.name] = previous_pkg
        previous_spec.loader.exec_module(previous_pkg)
        previous = await previous_pkg.comfy_entrypoint().get_node_list()
        current_ids = [cls.define_schema().node_id for cls in own]
        previous_ids = [cls.define_schema().node_id for cls in previous]
        assert len(previous_ids) == 343 and current_ids[:343] == previous_ids
        prior349 = json.loads((project / 'artifacts/five-track-development-20260918/avatar-voice-workflow-cpu-v15/report.json').read_text(encoding='utf8'))['registered_node_ids']
        assert len(prior349) == 349 and current_ids[:349] == prior349
        for cls in own:
            nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
        info = {}
        for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
            if hasattr(cls, 'GET_NODE_INFO_V1'):
                info[key] = cls.GET_NODE_INFO_V1()
            else:
                info[key] = dict(input=cls.INPUT_TYPES(), output=cls.RETURN_TYPES, output_name=getattr(cls, 'RETURN_NAMES', cls.RETURN_TYPES))
        info = json.loads(json.dumps(info))
        available_images = info['LoadImage']['input']['required']['image'][0]
        image = next((name for name in available_images if name == 't8_h4c2_bund_korean_mv_ref_20260916.png'), available_images[0])
        from PIL import Image
        import folder_paths
        with Image.open(folder_paths.get_annotated_filepath(image)) as reference:
            if reference.width * 3 != reference.height * 2:
                raise ValueError('Choose an actual2:3 reference, never qualify a stretched default')
        audio_spec = info['LoadAudio']['input']['required']['audio']
        available_audio = audio_spec[0] if isinstance(audio_spec[0], list) else audio_spec[1]['options']
        audio = 'h3_twopass_voice_5683_5p152s.flac'
        if audio not in available_audio:
            raise ValueError('Known local voice reference is missing')
        args.output.mkdir(parents=True)
        results = []
        routes = (('voice_long_dialogue_owner',) if args.dialogue_owner_only else ('voice_long_window',) if args.long_window_only else ('voice_long_joint',) if args.long_joint_only else ('voice_dual_joint',) if args.joint_only
                  else ('avatar', 'avatar_preview', 'voice_neutral', 'voice_emotion', 'voice_dual', 'voice_long'))
        for route in routes:
            recipe = (voice_long_graph(info, dual=route.startswith('voice_dual'), image=image, audio=audio)
                if route in {'voice_dual', 'voice_dual_joint', 'voice_long', 'voice_long_joint', 'voice_long_window', 'voice_long_dialogue_owner'} else graph(info, route=route, image=image, audio=audio))
            if route in {'voice_dual_joint', 'voice_long_joint', 'voice_long_window', 'voice_long_dialogue_owner'}:
                from tools.qualify_native_voice_dual_gpu import prepared_probe_graph, with_query_route
                recipe = with_query_route(recipe, 'joint_av_exp')
                recipe['8']['inputs']['chain_id'] = ('voice_reference_native20_jointAV_fresh' if route in {'voice_long_joint', 'voice_long_window', 'voice_long_dialogue_owner'}
                                                    else 'voice_reference_dual20plus4_jointAV_fresh')
                if route in {'voice_long_joint', 'voice_long_window', 'voice_long_dialogue_owner'}:
                    recipe = prepared_probe_graph(recipe, single_model=True)
                if route in {'voice_long_window', 'voice_long_dialogue_owner'}:
                    from tools.qualify_native_voice_dual_gpu import with_window_text_graph
                    policy = ('dialogue_start_owner_exp' if route == 'voice_long_dialogue_owner'
                              else 'accepted_window_text_exp')
                    recipe = with_window_text_graph(recipe, text_policy=policy)
                    recipe['8']['inputs']['chain_id'] = ('voice_reference_native20_dialogue_owner_fresh'
                        if route == 'voice_long_dialogue_owner' else 'voice_reference_native20_window_text_fresh')
            api, workflow = build(info, recipe, 'T8_' + route)
            if route in {'voice_dual', 'voice_dual_joint', 'voice_long', 'voice_long_joint', 'voice_long_window', 'voice_long_dialogue_owner'}:
                workflow['nodes'][-1]['widgets_values'] = [
                    '# 原生人物音色／情绪长视频接线 EXP，尚未做本图GPU／人审\n\n'
                    '匹配Ref2VA原生底模；没有专用声音克隆权重，没有Turbo LoRA。'
                    '人物接ref_images，声音接ref_audios；native，不接drive_audio/final_audio。'
                    '内部生成并交付新音轨，不拿参考录音直接mux。\n\n'
                    '三块脚本：Global仅固定人物/声音绑定/场景；Local每行一次台词及情绪；'
                    'Timeline百分比0-35/35-70/70-100；length193按原生17n+5自动对齐为209帧计划，交付裁成192帧/8秒。'
                    '本图实际事件区间0–3.042/3.042–6.083/6.083–8.708秒，尾部超出交付被裁去；不是精确0–2.8/2.8–5.6/5.6–8秒。'
                    '两内循环段，接缝约5.17秒，不是各4秒。改变时长须同步length/时间线。'
                    '无前景参考视频声音时Audio1是声音参考；加入视频声音后按media_map重写编号。\n\n'
                    + ('本图原生20步一采完成联合AV，再用原learned3D与独立HIGH4细化。'
                       'auto锁定已完成一采音频，不把20步音色资格推给4+4Turbo。'
                       '本图H4 LowVRAM与C2 ChunkFFN只是计算分块，不是空间tile。'
                       '可另接二采MODEL/LoRA，但该组合须另验。'
                       if route.startswith('voice_dual') else '本图每段原生Stock20；原生已完成AV尾部续接，无新接缝算法。')
                    + '\n\n只做CPU/Core接口验证，不等于本图长视频音色、口型、接缝或16GB性能通过。'
                    '跨语言声音参考属实验，逐字及声纹/情绪不保证。改配置用新chain_id；不动旧图。']
            if route == 'voice_dual_joint':
                workflow['nodes'][-1]['widgets_values'][0] += (
                    '\n\n显式音画扩展候选：Plan→Query Route（joint_av_exp）→内循环，'
                    '不是旧节点默认迁移。video_only_paper只路由视频，旧基线ASR发现第一句重复。'
                    '新的joint候选两段8秒48前向与完整AV机械完成，离线无目标提示ASR只读出两句，'
                    '但自动语言识别误标en；不把文字证据当声纹/情绪/口型/接缝通过或精确时刻证明。'
                    '联合音频时间路由未经Prompt Relay论文音频验证；原音频sigma/LOW20+HIGH4/'
                    '已完成一采音频锁定/learned3D/接缝不改，人审待，不能当正式推荐。'
                    '有音色参考整体声音较小，未自动补偿，试听再判断。')
            if route == 'voice_long_joint':
                workflow['nodes'][-1]['widgets_values'][0] += (
                    '\n\n显式单模型联合音画候选：Plan→Query Route（joint_av_exp）→原内循环。'
                    '每段20步、两段8秒，H4/C2只是计算分块；没有learned3D或HIGH二采。'
                    '旧节点默认/参考音频/采样sigma/原生AV尾部接缝不改，GPU以实际terminal为准。'
                    '联合音频路由未经Prompt Relay论文音频验证，人审待，不保证精确时刻或音色克隆。'
                    '参考只参与条件，不mux参考录音，不自动补偿音量，改配置用新chain_id。')
            if route == 'voice_long_window':
                workflow['nodes'][-1]['widgets_values'][0] += (
                    '\n\n窗口文本单变量EXP：原Plan→joint_av_exp Query Route→窗口文本策略→原单模型内循环。'
                    '原193→209计划、Local三事件、步骤20、尺寸512768、H4C2、AV尾部及接缝都保持。'
                    '仅显式accepted_window_text_exp不重复tokenize已结束/未来/只在已知上下文的局部事件；'
                    '跨越当前交付窗口的事件仍保留原时间/sigma，可能仍会重复，不是逐字控制或论文all-key路线。'
                    '默认preserve_all精确Plan恒等。零/一个事件只旁路Relay attention，不旁路原生续接。'
                    'CPU图通过不代表完整GPU/对白/音色/情绪/口型/接缝验收；独立新chain，不覆盖旧图或旧样片。')
            if route == 'voice_long_dialogue_owner':
                workflow['nodes'][-1]['widgets_values'][0] += (
                    '\n\n一次台词归属单变量EXP：dialogue_start_owner_exp。只由Local事件起始的交付窗口'
                    '保留该事件<d>块；后续相交窗口保留视觉/情绪与原时间/sigma，但不重新发出字面台词。'
                    '不改变AV上下文、音轨、步数、mask、接缝；不静音或裁断正在续接的已知音频。'
                    'Global不可放<d>，标记必须成对；未标记对白不识别。不是原论文all-key方法，'
                    '不保证台词完成/逐字/不重复，CPU图不是GPU或人审资格；保存仅为artifact，不覆盖旧图。')
            # Validate actual saved frontend conversion, not just the source API.
            _, schemas = selected_frontend_schema(recipe, info)
            reread = saved_api(workflow, schemas)
            result = await execution.validate_prompt('t8-' + route + '-cpu', reread, None)
            item = dict(route=route, validation=result, execution_nodes=len(reread),
                        image=image, audio=audio, queued=False)
            expected_outputs = {node_id for node_id, node in reread.items() if node['class_type'] in {
                'SaveVideo', 'MiniMaxH3AudioSourceExplanationT8', 'MiniMaxH3DualModelLongVideoEXPT8',
                'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'}}
            if not result[0] or result[3] or len(expected_outputs) != 2 or set(result[2]) != expected_outputs:
                raise RuntimeError(json.dumps(item, ensure_ascii=False))
            for suffix, data in (('api.json', api), ('json', workflow)):
                path = args.output / ('2026-09-18_T8_' + route + '_EXP.' + suffix)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
                item[suffix] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            results.append(item)
        return results, current_ids
    cases, ids = asyncio.run(validate())
    report = dict(status='actual_Core_saved_candidate_validation_pass_not_live_UI_or_inference',
                  cases=cases, registered_nodes=len(ids), registered_node_ids=ids, previous343_prefix_preserved=True,
                  previous_provider='bound_h16_source_release_package_343', cuda_initialized=torch.cuda.is_initialized(),
                  previous349_prefix_preserved=True,
                  queued=False, browser_used=False)
    if report['cuda_initialized']:
        raise RuntimeError('CPU qualification initialized CUDA')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
