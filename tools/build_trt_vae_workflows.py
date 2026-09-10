"""Project-owned frontend examples; CPU only, never touches a live frontend."""
from copy import deepcopy
import uuid

from .api_to_frontend_workflow import convert
from .frontend_workflow_compat import normalize_native_widget_inputs

UNSELECTED = '未找到引擎：请先编译'


def recipe(source=None, mode='check'):
    if mode == 'check':
        return {'1': {'class_type':'MiniMaxH3TRTVAECheckEXPT8','inputs':{'runtime_directory':''}}}
    if mode == 'compile':
        return {'1': {'class_type':'MiniMaxH3TRTVAECompileEXPT8','inputs':{
            'kind':'decoder-flex','runtime_directory':'','timeout_seconds':1800}}}
    if mode not in ('decoder','full','compare') or source is None:
        raise ValueError('Expected an existing native short generation recipe')
    graph = deepcopy(source)
    if graph['1'] != {'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_fp16.safetensors'}}:
        raise ValueError('Do not replace an unknown video VAE')
    values = {'native_video_vae':'minimax_h3_video_vae_fp16.safetensors','decoder_engine':UNSELECTED,
              'runtime_directory':'','cpu_output_budget_mib':1024}
    if mode == 'compare':
        expected = {'class_type':'MiniMaxH3AVDecodeT8','inputs':{
            'av_latent':['10',0],'video_vae':['1',0],'audio_vae':['2',0]}}
        if graph.get('11') != expected or any(key in graph for key in ('101','102','103','104','105')):
            raise ValueError('Comparison requires the known single-sampler AV decode recipe')
        # The native AV decoder exposes that exact video latent after it has
        # completed. This data dependency orders the two decoders without a
        # second sampler, audio decoder or new global execution patch.
        graph['101'] = {'class_type':'MiniMaxH3TRTVAEDecoderEXPT8','inputs':values}
        graph['102'] = {'class_type':'VAEDecode','inputs':{'samples':['11',2],'vae':['101',0]}}
        graph['103'] = deepcopy(graph['12'])
        graph['103']['inputs']['frames'] = ['102',0]
        graph['18']['inputs']['filename_prefix'] = 'MiniMaxH3/TRT-VAE/compare_native'
        graph['104'] = deepcopy(graph['18'])
        graph['104']['inputs'].update(images=['103',0],audio=['12',1],filename_prefix='MiniMaxH3/TRT-VAE/compare_trt')
        # Both saves take the exact same trimmed AUDIO output, not a second
        # VAE decode or a separately mixed/remixed waveform.
        graph['105'] = {'class_type':'PreviewAny','inputs':{'source':['104',2]}}
        return graph
    if mode == 'full':
        values.update(image_encoder_engine=UNSELECTED,video_encoder_engine=UNSELECTED)
    graph['1'] = {'class_type':'MiniMaxH3TRTVAEFullEXPT8' if mode == 'full' else 'MiniMaxH3TRTVAEDecoderEXPT8',
                  'inputs':values}
    return graph


def build(source, info, mode, title):
    graph = recipe(source,mode)
    workflow = convert(graph,info,title)
    normalize_native_widget_inputs(workflow)
    workflow['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL,'t8:trt-vae:'+mode))
    note = ('# TRT VAE · 1.78.0 可选 EXP\n\n'
            '这不是新的生成模型，只更换视频VAE执行方式。音乐/人声仍走原音频VAE。'
            '不替换旧工作流；不需要安装上游节点。\n\n'
            '先读本目录README。准备独立TensorRT运行环境和ONNX，先检查，再单独编译；'
            '编译完成刷新模型列表，手动选择本机引擎。当前占位值不可直接运行生成。'
            'runtime_directory留空使用models/vae/h3_trt/runtime/site-packages；不要填写别人机器的路径。\n\n'
            '推荐flex解码＋原生编码。Full模式还需T1和T17两套encoder，单图未测出速度优势，'
            '编码差异可能影响后续画面。W4已审短片整体可接受，但量化差异更大，不保证任意素材。\n\n'
            '目前只做短片验证，长片测试已取消，不代表长片已通过。本次集中短片和静态文字审片已获整体认可。'
            '首次编译、引擎加载、采样、音频和保存也有耗时，不把VAE解码提速当整段生成提速。\n\n'
            '完整短片输出阶段实测：原生30.54秒、TRT31.33秒，包含加载、解码和保存，不含采样；'
            '当前没有总耗时收益。只是可选EXP，不建议为了这组结果专门安装。')
    if mode == 'compare':
        note += ('\n\n同潜空间对照：只运行一次原生8步采样。先由AV Decode完成原生视频/音频解码，'
                 '再把其video_latent交给TRT解码；两份保存共用同一条已裁剪音频。'
                 'compare_native与compare_trt是明确标注对照，不是匿名盲测。'
                 '本图是复用模板，已有审片不用重跑；不要把同seed的两次采样当同latent对照。')
    node_id = workflow['last_node_id']+1
    workflow['nodes'].append({'id':node_id,'type':'MarkdownNote','title':'先读这里 / TRT VAE',
        'pos':[0,-650],'size':[900,540],'flags':{},'order':len(graph),'mode':0,
        'inputs':[],'outputs':[],'properties':{},'widgets_values':[note]})
    workflow['last_node_id'] = node_id
    workflow['extra']['trt_vae_delivery_status'] = 'exp_release_1_78_0_short_review_accepted_no_long_qualification'
    first = workflow['nodes'][0]
    first['size'] = [560,360 if mode == 'full' else 250]
    return graph,workflow
