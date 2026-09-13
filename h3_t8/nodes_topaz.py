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
            description='正式 tvai_up；输出无损 MOV/MKV 母版，逐帧时间轴、音频包与解码PCM均检查。不插帧、不自动下载。1x 为原尺寸增强，2x/4x需对应正式权重；Iris2x短片已完成机械验证，画质待审。',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_REGULAR_MODEL_IDS, default='iris-3',
                    tooltip='常用正式增强模型定义ID。模型仍须先在正式Topaz中下载；星光不是此路线。'),
                io.Combo.Input('scale', options=['1x', '2x', '4x'], default='2x', tooltip='不改变帧率；需在正式 Topaz 中准备对应倍率的权重。'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True),
                io.String.Input('parameters_json', default='{}', multiline=True, advanced=True,
                    tooltip='可选模型参数，例如 {"noise":0.3,"details":0.5}。仅接受该模型/版本支持的参数。空对象使用程序默认。'),
                io.Combo.Input('size_mode', options=['scale', 'target_dimensions'], default='scale',
                    optional=True, advanced=True, tooltip='scale 使用上方倍率。target_dimensions 使用下方宽高，忽略倍率；需保持原比例、1至4倍，不裁剪或拉伸。'),
                io.Int.Input('target_width', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；填写32至8192之间的偶数。'),
                io.Int.Input('target_height', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；宽高须同时填写并保持原视频比例。'),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='可选母版输出目录。留空写入ComfyUI/output/MiniMaxH3-Topaz；磁盘不足时可填写另一个本地目录。'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='仅用于列表中尚未收录的新正式模型ID；填写后覆盖model_id。')],
            outputs=[io.Video.Output('enhanced_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, scale, vram_fraction=.8, parameters_json='{}',
                size_mode='scale', target_width=0, target_height=0, output_directory='', custom_model_id=''):
        from comfy.model_management import throw_exception_if_processing_interrupted
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
        parameters = json.loads(parameters_json)
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
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
        path, report = run_regular(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, width=target_width if size_mode == 'target_dimensions' else None,
            height=target_height if size_mode == 'target_dimensions' else None,
            scale=int(scale[0]), size_mode=size_mode,
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            vram=vram_fraction, parameters=parameters, interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


TOPAZ_NODE_CLASSES = [MiniMaxH3TopazEnvironmentEXPT8, MiniMaxH3TopazVideoEXPT8]
