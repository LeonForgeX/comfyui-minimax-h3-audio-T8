"""One isolated serial short dual-MODEL job, guarded, no automatic retry."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


FLAT_STUDIO_GLOBAL_PROMPT = (
    "One continuous locked medium shot inside a plain neutral-gray matte studio. "
    "Uniform diffuse ceiling illumination remains physically fixed for the entire video, "
    "with constant exposure, white balance, brightness and shadow direction. No windows, "
    "lamps, screens, reflections, glowing objects, sunbeams, moving shadows, light pulses, "
    "flicker or automatic-exposure changes. A woman in a matte blue coat stands against the "
    "plain wall holding one matte red book. Consistent face, clothing, book, camera position "
    "and background; no cuts, subtitles or screen text. A quiet constant room tone is audible."
)
FLAT_STUDIO_LOCAL_PROMPTS = (
    "She looks toward the camera and clearly says one Mandarin sentence: <d>测试开始</d>. "
    "Natural mouth movement; the book remains at her waist.\n"
    "She becomes silent and holds the same red book still at waist height. No speech.\n"
    "Without speaking, she slowly raises the same red book from her waist to her chest in one "
    "smooth uninterrupted movement that continues through the segment boundary.\n"
    "She holds the same book motionless at chest height and looks at the camera. No speech."
)

FLAT_CEL_GRID_GLOBAL_PROMPT = (
    "One continuous locked orthographic medium-wide shot in flat two-dimensional cel animation. "
    "A perfectly uniform pale-gray background contains five stationary dark-gray registration "
    "lines and one fixed black floor line; their positions, widths and colors never change. "
    "Lighting is not depicted: no gradients, highlights, shadows, reflections, glow, light "
    "spots, exposure shifts, color-temperature shifts or brightness breathing. A woman in a "
    "solid matte teal coat holds one solid matte orange rectangular card. Her identity, outline, "
    "colors and the background grid remain constant. Fixed framing; no cuts, camera movement, "
    "zoom, depth of field, text or subtitles. Constant quiet room tone; no speech."
)
FLAT_CEL_GRID_LOCAL_PROMPTS = (
    "She stands motionless with the orange card held to the left of the center registration line. "
    "No speech.\n"
    "She begins moving the same orange card slowly and horizontally toward the center line. The "
    "background grid and all colors remain unchanged.\n"
    "She continues the same single smooth horizontal card movement across the center registration "
    "line and through the segment boundary without pausing, reversing or restarting. No speech.\n"
    "She stops with the same orange card to the right of the center line and holds it motionless. "
    "The background grid remains unchanged. No speech."
)

BUND_KOREAN_MV_GLOBAL_PROMPT = (
    "主角是已连接首帧参考图中的年轻女性：居中分缝的金色长卷发、黑色窄框墨镜、"
    "珊瑚色唇妆、含笑妩媚的眼神，身穿黑色上衣；她是夜晚外滩场景里的演唱者。"
    "已连接首帧是 [Shot 1] 角色参考；从第一帧开始保持她的脸部身份、五官、"
    "发型、墨镜、唇妆和黑色服装一致。写实温柔的韩文 MV 质感，夜晚外滩被暖金色路灯"
    "与冷色霓虹交织照亮，浅景深、肤色柔和，镜头运动缓慢滑顺。This is one continuous "
    "eight-second, two-segment diagnostic excerpt from the longer MV; preserve the same performer, "
    "wardrobe, voice and song across the internal segment boundary without restarting."
)
BUND_KOREAN_MV_LOCAL_PROMPTS = (
    "[Shot 1] 镜头从已连接首帧开始：主角的脸部近景，居中分缝的金色长卷发、"
    "黑色窄框墨镜、珊瑚色唇妆、含笑妩媚的眼神和黑色上衣保持与首帧一致。深色夜色里"
    "浮着柔散光斑，左下角虚化前景随镜头轻移滑出画面。她睫毛轻垂再抬起，嘴唇自然张开，"
    "从第一帧起柔声唱 <d>[Korean] 아침 햇살 문을 열면</d>；镜头以极慢速度向前推近，"
    "暖光在她发梢流动。\n"
    "[Shot 2] At 00:03.500, 柔和切到中近景并缓缓后拉，露出她身后外滩步道的栏杆与"
    "对岸楼群的金色散景。主角（S1）迎着江风微抬下巴，带着浅笑唱 "
    "<d>[Korean] 작은 새가 노래해요</d>，指尖轻轻碰一下耳环。本镜头、人物、歌声和"
    "后拉运动必须连续穿过内部片段接缝，不得暂停、重启、跳位或更换身份。\n"
    "[Shot 3] At 00:07.000, 镜头绕到她的侧面半身，江面反光在她发间流动，风把几缕"
    "发丝吹过脸颊。主角（S1）目光柔媚地落向镜头，唱 "
    "<d>[Korean] 초록 바람 <scenetrans></d>，歌声在切镜后保持连贯。"
)


def apply_relay_prompt_profile(graph, profile, duration):
    """Bind an explicit diagnostic prompt without changing production node behavior."""

    if profile not in ("legacy-concert", "flat-studio", "flat-cel-grid", "bund-korean-mv-8s"):
        raise ValueError("Unknown Relay prompt profile")
    if graph.get("7", {}).get("class_type") != "MiniMaxH3PromptRelayPlanT8Advanced":
        raise ValueError("Relay prompt profile requires the expected Relay node")
    graph["7"]["inputs"].update(
        length=duration * 24 + 1,
        local_prompts="She looks toward the camera.\nShe speaks the sentence and listens.",
        time_ranges="0-50\n50-100",
    )
    if duration < 8:
        if profile != "legacy-concert":
            raise ValueError("The selected Relay profile requires the two-segment 8s probe")
        return
    if profile == "flat-studio":
        graph["7"]["inputs"].update(
            global_prompt=FLAT_STUDIO_GLOBAL_PROMPT,
            local_prompts=FLAT_STUDIO_LOCAL_PROMPTS,
            time_ranges="0-15\n15-40\n40-75\n75-100",
        )
    elif profile == "flat-cel-grid":
        graph["7"]["inputs"].update(
            global_prompt=FLAT_CEL_GRID_GLOBAL_PROMPT,
            local_prompts=FLAT_CEL_GRID_LOCAL_PROMPTS,
            time_ranges="0-15\n15-40\n40-75\n75-100",
        )
    elif profile == "bund-korean-mv-8s":
        graph["7"]["inputs"].update(
            global_prompt=BUND_KOREAN_MV_GLOBAL_PROMPT,
            local_prompts=BUND_KOREAN_MV_LOCAL_PROMPTS,
            time_ranges="0-43.75\n43.75-87.5\n87.5-100",
        )
    else:
        from tools.build_dual_model_workflows import SCENE_PROMPT, RELAY_EVENTS

        graph["7"]["inputs"].update(
            global_prompt=SCENE_PROMPT,
            local_prompts=RELAY_EVENTS,
            time_ranges="0-15\n15-40\n40-75\n75-100",
        )
    if duration == 24:
        graph["7"]["inputs"]["length"] = 583


def core_memory_options(command, disable_pinned_memory=False):
    """Explicit test-process options, never a user launcher/global setting edit."""
    result = list(command)
    if '--disable-pinned-memory' in result:
        raise ValueError('Core memory options are already configured')
    if disable_pinned_memory:
        result.append('--disable-pinned-memory')
    return result


def apply_canvas_size(graph, width, height, low_width, low_height):
    """Bind one exact-ratio LOW/HIGH canvas without implicit stretching."""

    values = (width, height, low_width, low_height)
    if any(not isinstance(value, int) or value <= 0 or value % 32 for value in values):
        raise ValueError("Canvas dimensions must be positive integer multiples of 32")
    if width * low_height != height * low_width:
        raise ValueError("LOW and HIGH canvases must use the same aspect ratio")
    graph["8"]["inputs"].update(
        width=width, height=height, low_width=low_width, low_height=low_height
    )


def apply_context_frames(graph, context_frames):
    """Select one native H3 continuation overlap without changing other controls."""

    if context_frames not in (22, 39):
        raise ValueError("Continuation context must be 22 or 39 frames")
    graph["8"]["inputs"]["context_frames"] = context_frames


def apply_low_context_source(graph, low_context_source):
    """Select the LOW continuation source as one explicit controlled variable."""

    choices = ("independent_low_x0", "accepted_picture_low_context_v1")
    if low_context_source not in choices:
        raise ValueError("Unknown LOW continuation context source")
    graph["8"]["inputs"]["low_context_source"] = low_context_source


def attach_first_frame(graph, core, filename, *, canvas_size=None):
    """Bind a real Core input file without copying it or altering the user's UI."""
    if filename is None:
        return None
    relative = Path(filename)
    input_root = (Path(core) / 'input').resolve(strict=True)
    if relative.is_absolute() or '..' in relative.parts or not filename.strip():
        raise ValueError('First frame must be a Core input-relative filename')
    source = (input_root / relative).resolve(strict=True)
    if not source.is_relative_to(input_root) or not source.is_file():
        raise ValueError('First frame must remain inside the Core input directory')
    if '23' in graph or 'first_frame' in graph['8']['inputs']:
        raise ValueError('Reference probe would overwrite an existing input')
    receipt = file_identity(source)
    if canvas_size is not None:
        from PIL import Image

        width, height = canvas_size
        with Image.open(source) as image:
            source_width, source_height = image.size
        if source_width * height != source_height * width:
            raise ValueError(
                f"First-frame aspect {source_width}:{source_height} does not match "
                f"canvas {width}:{height}; refusing implicit distortion"
            )
        receipt.update(pixel_width=source_width, pixel_height=source_height)
    graph['23'] = {'class_type': 'LoadImage', 'inputs': {'image': relative.as_posix()}}
    graph['8']['inputs']['first_frame'] = ['23', 0]
    return receipt


def attach_memory_chunks(graph, backend, head_chunks=1, ffn_chunks=1):
    if head_chunks not in (1, 4) or ffn_chunks not in (1, 2):
        raise ValueError('Only explicit tested memory chunk recipes are supported by this harness')
    if head_chunks == ffn_chunks == 1:
        return
    if backend != 'kj-memory':
        raise ValueError('Memory chunk recipe requires the authenticated KJ memory backend')
    if any(key in graph for key in ('24', '25', '26', '27')):
        raise ValueError('Memory recipe would overwrite graph nodes')
    for branch, source, head_id, ffn_id in (('model_pass1', '21', '24', '26'),
                                            ('model_pass2', '22', '25', '27')):
        if head_chunks != 1:
            graph[head_id] = {'class_type': 'MiniMaxLowVRAMAttention',
                             'inputs': {'model': [source, 0], 'head_chunks': head_chunks}}
            source = head_id
        if ffn_chunks != 1:
            graph[ffn_id] = {'class_type': 'MiniMaxChunkFeedForward',
                            'inputs': {'model': [source, 0], 'chunks': ffn_chunks, 'seq_threshold': 4096}}
            source = ffn_id
        graph['8']['inputs'][branch] = [source, 0]


def attach_t8_memory(graph, head_chunks=4, ffn_chunks=2):
    """Replace the two KJ slots with one explicit authenticated T8 recipe.

    ``head_chunks=1`` keeps the T8 early-release wrapper active without changing
    the attention-head launch grouping.  It is the deterministic control for
    the separately qualified ``head_chunks=4`` experiment.
    """

    if head_chunks not in (1, 4) or ffn_chunks not in (1, 2):
        raise ValueError('Only explicit tested T8 memory recipes are supported by this harness')
    if not all(key in graph for key in ('8', '21', '22', '30', '31')):
        raise ValueError('T8 memory recipe requires the accepted dual-model graph')
    if any(key in graph for key in ('24', '25')):
        raise ValueError('T8 memory recipe would overwrite graph nodes')
    replacement = {}
    for branch, source, attention_id, ffn_id in (
            ('model_pass1', '30', '21', '24'), ('model_pass2', '31', '22', '25')):
        replacement[attention_id] = {
            'class_type': 'MiniMaxH3LowVRAMAttentionT8Advanced',
            'inputs': {'model': [source, 0], 'head_chunks': head_chunks},
        }
        replacement[ffn_id] = {
            'class_type': 'MiniMaxH3ChunkFeedForwardT8Advanced',
            'inputs': {'model': [attention_id, 0], 'chunks': ffn_chunks,
                       'seq_threshold': 4096},
        }
        replacement.setdefault('runner', {})[branch] = [ffn_id, 0]
    runner_inputs = replacement.pop('runner')
    graph.update(replacement)
    graph['8']['inputs'].update(runner_inputs)


def attach_eav_stock20(graph, enabled=False):
    """An explicit separate recipe; never apply EAV to a four-step Turbo model."""
    if not enabled:
        return
    if (graph['1']['class_type'] != 'UNETLoader'
            or graph['2']['class_type'] != 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'
            or graph['21']['inputs']['model'] != ['2', 0]
            or graph['22']['inputs']['model'] != ['3', 0]):
        raise ValueError('Unexpected graph for the Stock20 first-model recipe')
    graph['21']['inputs']['model'] = ['1', 0]
    del graph['2']
    graph['8']['inputs'].update(coarse_steps=20, eav_mode='apply_exp')


def attach_second_base(graph, core, filename):
    """Explicit existing native base for pass2; never replace the pass1 loader."""
    if filename is None:
        return None
    relative = Path(filename)
    root = (Path(core) / 'models/diffusion_models').resolve(strict=True)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.safetensors':
        raise ValueError('Second base must be a diffusion_models-relative safetensors file')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Second base must remain in the explicit model directory')
    if (relative.as_posix() == graph['1']['inputs']['unet_name'] or '28' in graph
            or graph['3']['inputs']['model'] != ['1', 0]):
        raise ValueError('Distinct second base would overwrite an unexpected graph')
    graph['28'] = {'class_type': 'UNETLoader', 'inputs': {
        'unet_name': relative.as_posix(), 'weight_dtype': 'default'}}
    graph['3']['inputs']['model'] = ['28', 0]
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--port', type=int, default=8208)
    parser.add_argument('--relay', action='store_true')
    parser.add_argument('--relay-prompt-profile',
                        choices=('legacy-concert', 'flat-studio', 'flat-cel-grid', 'bund-korean-mv-8s'),
                        default='legacy-concert',
                        help='Explicit Relay diagnostic content; fixed-light profiles remove moving-light cues')
    parser.add_argument('--backend', choices=('kj', 'kj-memory', 't8-memory', 'sol', 'pytorch'), default='kj')
    parser.add_argument('--sol-tau', type=float, choices=(0.0, 0.5, 1.3), default=1.3,
                        help='Explicit Sol-only diagnostic threshold; never silently change connected settings')
    parser.add_argument('--duration', type=int, choices=(3, 8, 24), default=3)
    parser.add_argument('--context-frames', type=int, choices=(22, 39), default=22,
                        help='Native continuation overlap; 39 is the background-continuity diagnostic')
    parser.add_argument('--low-context-source',
                        choices=('independent_low_x0', 'accepted_picture_low_context_v1'),
                        default='independent_low_x0',
                        help='Explicit LOW continuation source; accepted picture changes only later LOW guidance')
    parser.add_argument('--video-context-mode',
                        choices=('reference_only', 'high_native_mask_exp', 'high_native_mask_ramp_exp'),
                        default='reference_only',
                        help='Explicit single-variable continuation experiment; existing workflow default remains reference_only')
    parser.add_argument('--content', choices=('portrait', 'game'), default='portrait')
    parser.add_argument('--query-rows', type=int, choices=(256, 512, 1024), default=256)
    parser.add_argument('--first-frame', help='Existing filename relative to Core input; no UI changes')
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--low-width', type=int, default=448)
    parser.add_argument('--low-height', type=int, default=224)
    parser.add_argument('--memory-head-chunks', type=int, choices=(1, 4), default=1)
    parser.add_argument('--memory-ffn-chunks', type=int, choices=(1, 2), default=1)
    parser.add_argument('--eav-stock20', action='store_true', help='Separate base20 first pass plus EMA4 second pass; Relay/KJ only')
    parser.add_argument('--second-base', help='Existing diffusion_models-relative native H3 base for pass2 only')
    parser.add_argument('--disable-pinned-memory', action='store_true',
                        help='Explicit isolated Core option; no global/user configuration change')
    parser.add_argument('--headroom-gib', type=int, choices=(0, 2), default=0,
                        help='Additional native DynamicVRAM headroom, preserving the existing5GiB reserve and resource guards')
    args = parser.parse_args()
    if args.backend in ('kj-memory', 't8-memory') and not args.relay:
        raise ValueError('This measured memory-composition probe requires Relay; plain direct-forward counting is separate')
    if not args.relay and args.relay_prompt_profile != 'legacy-concert':
        raise ValueError('A Relay prompt profile requires --relay')
    if args.backend == 't8-memory' and args.memory_ffn_chunks != 2:
        raise ValueError('This T8 dual-model acceptance probe requires the active ChunkFFN chunks=2 path')
    if args.eav_stock20 and (not args.relay or args.backend not in ('kj', 'kj-memory')):
        raise ValueError('EAV probe requires the explicit KJ/Relay combination')
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new artifact directory in this worktree')
    transport.CORE = args.core.resolve()
    transport.PROJECT = project
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        if args.backend in ('kj', 'kj-memory', 'sol'):
            idx = result.index('--whitelist-custom-nodes') + 1
            result.insert(idx, 'ComfyUI-sol-attn' if args.backend == 'sol' else 'ComfyUI-KJNodes')
        return core_memory_options(result, args.disable_pinned_memory)

    transport.server_command = command
    template = project / 'artifacts/dual-workflows-cpu-v1' / ('Relay.prompt.json' if args.relay else 'Plain.prompt.json')
    if not template.is_file():
        template = project / 'tests/fixtures/dual_picture_accepted_api.json'
    graph = json.loads(template.read_text(encoding='utf-8'))
    apply_canvas_size(graph, args.width, args.height, args.low_width, args.low_height)
    apply_context_frames(graph, args.context_frames)
    apply_low_context_source(graph, args.low_context_source)
    graph['8']['inputs'].update(total_duration_seconds=float(args.duration), chain_id='dual_short_' + root.name,
        base_seed=2609032101, filename_prefix='Dual_Model_Pilot', query_chunk_rows=args.query_rows)
    for index, model in [('21', '2'), ('22', '3')]:
        if args.backend == 'kj':
            graph[index] = {'class_type': 'PathchSageAttentionKJ', 'inputs': {
                'model': [model, 0], 'sage_attention': 'auto', 'allow_compile': False}}
        elif args.backend == 'kj-memory':
            graph[index] = {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch',
                            'inputs': {'model': [model, 0]}}
        elif args.backend == 't8-memory':
            continue
        elif args.backend == 'sol':
            graph[index] = {'class_type': 'SolAttentionPatch', 'inputs': {
                'model': [model, 0], 'enabled': True, 'tau': args.sol_tau, 'min_tokens': 4096,
                'strict': True, 'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}}
        else:
            graph[index] = {'class_type': 'ModelAttentionBackend', 'inputs': {
                'model': [model, 0], 'attention': 'pytorch attention'}}
    graph['8']['inputs'].update(model_pass1=['21', 0], model_pass2=['22', 0])
    if args.backend == 't8-memory':
        attach_t8_memory(graph, args.memory_head_chunks, args.memory_ffn_chunks)
    if args.video_context_mode != 'reference_only':
        if args.duration < 8:
            raise ValueError('Video context experiment needs a real continuation segment')
        graph['8']['inputs']['video_context_mode'] = args.video_context_mode
    attach_eav_stock20(graph, args.eav_stock20)
    if args.backend != 't8-memory':
        attach_memory_chunks(graph, args.backend, args.memory_head_chunks, args.memory_ffn_chunks)
    second_base = attach_second_base(graph, args.core, args.second_base)
    if args.relay:
        apply_relay_prompt_profile(graph, args.relay_prompt_profile, args.duration)
    if args.content == 'game':
        if args.relay:
            raise ValueError('Game recipe currently qualified for plain timeline only; do not reuse portrait events')
        graph['8']['inputs']['global_prompt'] = transport.GAME_PROMPT
    reference = attach_first_frame(
        graph, args.core, args.first_frame, canvas_size=(args.width, args.height)
    )
    root.mkdir(parents=True)
    result = {'status': 'incomplete', 'mode': args.mode, 'relay': args.relay,
        'relay_prompt_profile': args.relay_prompt_profile if args.relay else None,
        'backend': args.backend, 'duration': args.duration, 'content': args.content,
        'width': int(graph['8']['inputs']['width']),
        'height': int(graph['8']['inputs']['height']),
        'low_width': int(graph['8']['inputs']['low_width']),
        'low_height': int(graph['8']['inputs']['low_height']),
        'context_frames': int(graph['8']['inputs']['context_frames']),
        'low_context_source': str(graph['8']['inputs']['low_context_source']),
        'video_context_mode': str(graph['8']['inputs'].get('video_context_mode', 'reference_only')),
        'first_frame_file': reference, 'memory_head_chunks': args.memory_head_chunks,
        'memory_ffn_chunks': args.memory_ffn_chunks, 'eav_stock20': args.eav_stock20,
        'disable_pinned_memory': args.disable_pinned_memory,
        'reserve_vram_gib': 5, 'headroom_gib': args.headroom_gib,
        'human_review': 'pending'}
    server = transport.OwnedServer(root, args.port, args.mode == 'cpu', args.headroom_gib)
    monitor = None
    guard = ResourceGuard()
    # Match the established host-wide acceleration lease; no unrelated process
    # is closed. Only this controller's server handle/children can be stopped.
    lease = args.core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            result['second_base_file'] = file_identity(second_base) if second_base else None
            source = transport.source_snapshot()
            source['tools/run_dual_model_pilot.py'] = file_identity(Path(__file__))['sha256']
            expected = {'core': verify_core_source(args.core), 'sources': source,
                        'mode': 'cpu-smoke' if args.mode == 'cpu' else 'gpu', 'pilot_graphs': {'dual_short': graph}}
            expected['runtime_options'] = {'reserve_vram_gib': 5, 'headroom_gib': args.headroom_gib}
            transport.write_json(root / 'paths.json', probe_resource_config(args.core, project))
            transport.write_json(root / 'expected.json', expected)
            if args.mode == 'gpu':
                reason = guard.observe(reader.sample(), startup=True)
                if reason:
                    raise RuntimeError('Startup resource guard: ' + reason)
            server.start()
            if args.mode == 'gpu':
                monitor = transport.ContinuousGuard(reader, guard, root / 'resources.jsonl', server)
                # Join the observer before NVML exits, including error paths.
                cleanup.callback(monitor.close)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            transport.wait_ready(server, check)
            history, _ = transport.execute_graph(server, {
                '1': {'class_type': 'T8ProgressiveEnvironmentAudit', 'inputs': {'expected_json': json.dumps(expected)}},
                '2': {'class_type': 'PreviewAny', 'inputs': {'source': ['1', 0]}}}, root / 'environment', check)
            result['environment'] = transport.preview_report(history, '2')
            transport.write_json(root / 'object-info.json', server.request('GET', '/object_info'))
            if args.mode == 'gpu':
                started = time.perf_counter()
                history, timing = transport.execute_graph(server, graph, root / 'generation', check, timeout=2400)
                result.update(status='generation_completed_independent_media_and_human_review_pending',
                              wall_seconds=time.perf_counter()-started, timing=timing)
            else:
                result['status'] = 'actual_owned_CPU_server_and_dual_graph_validation_pass_no_generation'
            if transport.source_snapshot() != {k: v for k, v in source.items() if k != 'tools/run_dual_model_pilot.py'}:
                raise RuntimeError('Runtime sources changed during the owned run')
            if reference:
                stable_reference = {
                    key: value for key, value in reference.items()
                    if key not in ('pixel_width', 'pixel_height')
                }
                if file_identity(Path(reference['path'])) != stable_reference:
                    raise RuntimeError('Reference image changed during the owned run')
            if second_base and file_identity(second_base) != result['second_base_file']:
                raise RuntimeError('Second base file changed during the owned run')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(root / 'terminal.json', result)
        print(json.dumps({'status': result['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
