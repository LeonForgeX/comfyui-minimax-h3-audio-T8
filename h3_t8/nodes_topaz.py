"""Optional official Topaz postprocessing. No discovery/installation at import."""
import json
from pathlib import Path
import tempfile
import uuid

import folder_paths
from comfy_api.latest import io, InputImpl


CATEGORY = 'T8/MiniMax H3/Post FX/Topaz'
TopazRuntime = io.Custom('T8_OFFICIAL_TOPAZ_RUNTIME')
TOPAZ_REGULAR_MODEL_IDS = [
    'iris-3', 'prob-4', 'rhea-1', 'nyx-3', 'ahq-12', 'amq-13', 'alq-13',
    'thf-4', 'thd-3', 'ghq-5', 'gcg-5', 'ganim-1', 'nxf-1', 'nxl-1',
]
TOPAZ_MODEL_GUIDE = (
    'iris-3 Iris：真人脸、低清人像；prob-4 Proteus：通用且参数最全；'
    'rhea-1 Rhea：中等质量细节恢复；nyx-3/nxf-1/nxl-1 Nyx：强降噪；'
    'ahq-12/amq-13/alq-13 Artemis：分别用于高/中/低质量素材；'
    'thf-4 Theia Fidelity：偏忠实；thd-3 Theia Detail：偏细节；'
    'ghq-5 Gaia HQ：高质量实拍/CG；gcg-5 Gaia CG：CG、锯齿和摩尔纹；'
    'ganim-1 Gaia Animation：动画。不同安装版本实际可用模型以环境检查报告为准。')
TOPAZ_FI_MODEL_IDS = ['apo-8', 'apf-2', 'chr-2', 'chf-3', 'aion-1']
TOPAZ_FI_GUIDE = ('apo-8 Apollo：高质量通用，适合非线性运动；apf-2 Apollo Fast：速度优先；'
    'chr-2 Chronos：通用高质量变帧率；chf-3 Chronos Fast：速度优先；'
    'aion-1 Aion：新版通用插帧。请先在正式Topaz软件内下载对应模型。')


def _parameter_overrides(mode, auto_estimate_frames, preblur, noise, details, halo,
                         sharpen, compression, add_noise, grain, grain_size,
                         keep_color, input_blend, parameters_json,
                         supported_model_parameters=None):
    try:
        custom = json.loads(parameters_json)
    except json.JSONDecodeError as error:
        raise ValueError('高级参数JSON格式无效；普通使用请保持 {}') from error
    if not isinstance(custom, dict):
        raise ValueError('高级参数JSON必须是对象，例如 {}')
    if mode == 'model_defaults':
        values = {}
    elif mode == 'auto_estimate':
        values = {'estimate': auto_estimate_frames}
    elif mode == 'manual':
        requested = {'preblur': preblur, 'noise': noise, 'details': details,
            'halo': halo, 'blur': sharpen, 'compression': compression}
        supported = (set(requested) if supported_model_parameters is None
            else {str(name).lower() for name in supported_model_parameters})
        values = {'estimate': 0, **{name: value for name, value in requested.items()
            if name in supported}, 'prenoise': add_noise, 'grain': grain,
            'gsize': grain_size, 'kcolor': int(keep_color), 'blend': input_blend}
    else:
        raise ValueError('未知 Topaz 参数模式')
    values.update(custom)
    return values


class MiniMaxH3TopazEnvironmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazEnvironmentEXPT8',
            display_name='Official Topaz · 环境检查 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='检查正式安装、签名、滤镜，并在报告中列出模型ID、各倍率候选权重与可用参数。文件存在不代表能运行。不会安装、下载或运行增强；星光需独立官方 Neuroserver 与有效权限。',
            inputs=[io.String.Input('install_directory', default='', tooltip='正式 Topaz Video 程序目录，不是模型目录。'),
                io.String.Input('model_definitions_directory', default='', tooltip='正式安装对应的 models JSON 定义目录。'),
                io.String.Input('model_data_directory', default='', tooltip='正式 Topaz 配置的模型数据目录，可能与定义目录不同。')],
            outputs=[TopazRuntime.Output('topaz_runtime'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, install_directory, model_definitions_directory, model_data_directory):
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import audit_installation
        if not all(isinstance(p, str) and p.strip() for p in
                (install_directory, model_definitions_directory, model_data_directory)):
            raise ValueError('请填写正式 Topaz 程序、模型定义和模型数据目录。没有安装 Topaz 不影响其他 T8 节点。')
        runtime = OfficialTopaz(install_directory, model_definitions_directory, model_data_directory)
        report = audit_installation(runtime)
        handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
            'definitions': str(runtime.definitions), 'data': str(runtime.data)}
        return io.NodeOutput(handle, json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazVideoEXPT8',
            display_name='Official Topaz · 视频高清增强 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='正式 tvai_up 高清增强；默认直接输出高质量H.264 MP4并保留原音频，无需再接“保存视频”。可选超大无损审计母版。不插帧、不自动下载；插帧请用独立Topaz插帧节点。Topaz软件无需保持打开。',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_REGULAR_MODEL_IDS, default='iris-3',
                    display_name='高清模型（用途见提示）', tooltip=TOPAZ_MODEL_GUIDE),
                io.Combo.Input('scale', options=['1x', '2x', '4x'], default='2x', tooltip='不改变帧率；需在正式 Topaz 中准备对应倍率的权重。'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True),
                io.String.Input('parameters_json', default='{}', multiline=True, advanced=True,
                    display_name='高级参数JSON（通常保持{}）',
                    tooltip='仅供精确覆盖独立控件或未来参数；普通用户不要填写。'),
                io.Combo.Input('size_mode', options=['scale', 'target_dimensions'], default='scale',
                    optional=True, advanced=True, tooltip='scale 使用上方倍率。target_dimensions 使用下方宽高，忽略倍率；需保持原比例、1至4倍，不裁剪或拉伸。'),
                io.Int.Input('target_width', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；填写32至8192之间的偶数。'),
                io.Int.Input('target_height', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；宽高须同时填写并保持原视频比例。'),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='可选母版输出目录。留空写入ComfyUI/output/MiniMaxH3-Topaz；磁盘不足时可填写另一个本地目录。'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='仅用于列表中尚未收录的新正式模型ID；填写后覆盖model_id。'),
                io.Combo.Input('output_profile', options=['delivery_h264', 'lossless_master'],
                    default='delivery_h264', optional=True,
                    display_name='保存格式', tooltip='默认delivery_h264：GPU编码的高质量MP4，体积正常且节点已直接保存；不要再接SaveVideo。lossless_master只供审计归档，可能产生十几GB。'),
                io.Combo.Input('parameter_mode', options=['model_defaults', 'auto_estimate', 'manual'],
                    default='model_defaults', optional=True, display_name='参数模式',
                    tooltip='model_defaults使用模型默认；auto_estimate让Topaz抽帧估计；manual使用下方独立参数。'),
                io.Int.Input('auto_estimate_frames', default=8, min=1, max=100, optional=True,
                    advanced=True, display_name='自动分析帧数', tooltip='仅auto_estimate生效；越多越慢。'),
                io.Float.Input('preblur', default=0., min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='抗锯齿(-) / 去模糊(+)'),
                io.Float.Input('noise', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='降噪'),
                io.Float.Input('details', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='恢复细节'),
                io.Float.Input('halo', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='去光晕'),
                io.Float.Input('sharpen', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='锐化'),
                io.Float.Input('compression', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='去压缩伪影'),
                io.Float.Input('add_noise', default=0., min=0., max=.1, step=.005, optional=True,
                    advanced=True, display_name='输入预加噪'),
                io.Float.Input('grain', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='输出颗粒'),
                io.Float.Input('grain_size', default=0., min=0., max=5., step=.1, optional=True,
                    advanced=True, display_name='颗粒尺寸'),
                io.Boolean.Input('keep_color', default=True, optional=True, advanced=True,
                    display_name='模型色彩校正'),
                io.Float.Input('input_blend', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='混合原片'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='额外推理实例',
                    tooltip='0最稳。1在本机60帧仅由11.5秒降到10.5秒，收益小但占用更多显存；16GB卡不建议更高。')],
            outputs=[io.Video.Output('enhanced_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, scale, vram_fraction=.8, parameters_json='{}',
                size_mode='scale', target_width=0, target_height=0, output_directory='', custom_model_id='',
                output_profile='delivery_h264', parameter_mode='model_defaults', auto_estimate_frames=8,
                preblur=0., noise=.5, details=.5, halo=.5, sharpen=.5, compression=.5,
                add_noise=0., grain=0., grain_size=0., keep_color=True, input_blend=0., engine_instances=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_regular
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('Connect the official Topaz environment node')
        if scale not in ('1x', '2x', '4x') or not isinstance(parameters_json, str) or len(parameters_json) > 8192:
            raise ValueError('Invalid Topaz scale or parameter JSON')
        if not isinstance(custom_model_id, str):
            raise ValueError('Custom Topaz model ID must be a string')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        _, definition = runtime.model(model_id)
        declared = {str(item.get('name', '')).lower() for item in definition.get('parameters', [])}
        parameters = _parameter_overrides(parameter_mode, auto_estimate_frames, preblur,
            noise, details, halo, sharpen, compression, add_noise, grain, grain_size,
            keep_color, input_blend, parameters_json, supported_model_parameters=declared)
        manual_controls = {'preblur', 'noise', 'details', 'halo', 'blur', 'compression'}
        parameter_audit = {'mode': parameter_mode,
            'ignored_unsupported_model_controls': sorted(manual_controls - declared)
                if parameter_mode == 'manual' else []}
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz requires an untrimmed, uncropped local file VIDEO. Save edited/frame-batch VIDEO first.') from error
        if not isinstance(output_directory, str):
            raise ValueError('Topaz output directory must be a local path string')
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('Custom Topaz output directory must be an absolute non-root path')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            output = Path(folder_paths.get_output_directory()).resolve(strict=True)
            root = output / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('Topaz output directory must be a real local directory')
        progress_bar = ProgressBar(100)
        path, report = run_regular(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, width=target_width if size_mode == 'target_dimensions' else None,
            height=target_height if size_mode == 'target_dimensions' else None,
            scale=int(scale[0]), size_mode=size_mode,
            output_profile=output_profile,
            instances=engine_instances,
            parameter_audit=parameter_audit,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            vram=vram_fraction, parameters=parameters, interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazFrameInterpolationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazFrameInterpolationEXPT8',
            display_name='Official Topaz · 视频插帧 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='独立正式tvai_fi插帧：2x/4x提高帧率但保持时长和尺寸，默认直接保存高质量H.264 MP4并原样复制音轨。不是慢动作，不与高清模型混用，不需要保持Topaz软件开启。',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_FI_MODEL_IDS, default='apo-8',
                    display_name='插帧模型（用途见提示）', tooltip=TOPAZ_FI_GUIDE),
                io.Combo.Input('multiplier', options=['2x', '4x'], default='2x',
                    display_name='帧率倍数', tooltip='2x：24→48、30→60；4x：24→96。时长不变。'),
                io.Float.Input('duplicate_threshold', default=.01, min=-.01, max=.2, step=.005,
                    advanced=True, display_name='重复帧检测阈值',
                    tooltip='0或负值关闭重复帧检测；过高可能误删正常静止帧。一般保持0.01。'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='留空写入ComfyUI/output/MiniMaxH3-Topaz；也可填写其他本地盘。'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='仅用于列表未收录的正式插帧模型ID；填写后覆盖model_id。'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='额外推理实例', tooltip='0最稳；增加会占更多显存。')],
            outputs=[io.Video.Output('interpolated_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, multiplier,
                duplicate_threshold=.01, vram_fraction=.8, output_directory='', custom_model_id='',
                engine_instances=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_interpolation
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('请连接 Official Topaz 环境检查节点')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz插帧需要未裁剪的本地文件VIDEO；帧批请先保存成视频。') from error
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('自定义Topaz输出目录必须是绝对的非磁盘根目录')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            root = Path(folder_paths.get_output_directory()).resolve(strict=True) / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('Topaz输出目录必须是真实本地目录')
        progress_bar = ProgressBar(100)
        path, report = run_interpolation(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, multiplier=int(multiplier[0]),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            vram=vram_fraction, instances=engine_instances, duplicate_threshold=duplicate_threshold,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


TOPAZ_NODE_CLASSES = [MiniMaxH3TopazEnvironmentEXPT8, MiniMaxH3TopazVideoEXPT8,
                      MiniMaxH3TopazFrameInterpolationEXPT8]
