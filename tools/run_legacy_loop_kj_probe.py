"""Exercise the existing single-MODEL Relay/EAV node, not the dual runner.

Uses the same guarded isolated server lifecycle. Explicit recipe: --relay
--backend kj-memory --memory-head-chunks 4 --memory-ffn-chunks 2
--eav-stock20 --disable-pinned-memory. No Turbo, no second pass, no upscaler.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dual_model_pilot as pilot  # noqa: E402
from tools.build_dual_model_workflows import defaults  # noqa: E402

NODE = 'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'


def legacy_graph(graph, info):
    dual = graph['8']['inputs']
    if (dual['coarse_steps'] != 20 or dual['refine_steps'] != 4
            or dual['eav_mode'] != 'apply_exp' or dual['prompt_relay_mode'] != 'apply_exp'
            or dual['total_duration_seconds'] != 3 or '2' in graph or '23' in graph or '28' in graph
            or graph['21'] != {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch',
                               'inputs': {'model': ['1', 0]}}
            or dual['model_pass1'] != ['26', 0]):
        raise ValueError('Legacy probe requires explicit base20 Relay/EAV memory head4/FFN2 recipe')
    if (graph['24'] != {'class_type': 'MiniMaxLowVRAMAttention',
                       'inputs': {'model': ['21', 0], 'head_chunks': 4}}
            or graph['26'] != {'class_type': 'MiniMaxChunkFeedForward',
                              'inputs': {'model': ['24', 0], 'chunks': 2, 'seq_threshold': 4096}}):
        raise ValueError('Unexpected KJ memory recipe')
    result = {key: deepcopy(graph[key]) for key in ('1', '4', '5', '6', '7', '21', '24', '26')}
    settings = defaults(info[NODE])
    # Carry matching public options, never the dual node's stage/upscale inputs.
    settings.update({key: deepcopy(value) for key, value in dual.items() if key in settings})
    settings.update(model=['26', 0], clip=['4', 0], video_vae=['5', 0], audio_vae=['6', 0],
        width=512, height=256, steps=20, sampler_name='dual_clock_euler', scheduler='native_flow',
        chain_id='legacy_' + dual['chain_id'], prompt_relay_plan=['7', 0],
        resume_existing=False, filename_prefix='Legacy_KJ_Relay_EAV_Stock20',
        model_id='FL2VA_INT8_ConvRot_no_Turbo_legacy_KJ_memory_probe')
    result['8'] = {'class_type': NODE, 'inputs': settings}
    return result


def main():
    project = Path(__file__).resolve().parents[1]
    schema = project / 'artifacts/dual-backend-benchmark-pytorch-gpu-v1/object-info.json'
    info = json.loads(schema.read_text(encoding='utf-8'))
    original_attach = pilot.attach_first_frame
    original_snapshot = pilot.transport.source_snapshot

    def attach(graph, core, filename):
        if filename:
            raise ValueError('This legacy composition probe has no reference input')
        replacement = legacy_graph(graph, info)
        graph.clear()
        graph.update(replacement)
        return None

    def snapshot():
        return {**original_snapshot(), 'tools/run_legacy_loop_kj_probe.py':
                pilot.file_identity(Path(__file__))['sha256'],
                str(schema.relative_to(project)): pilot.file_identity(schema)['sha256']}

    pilot.attach_first_frame = attach
    pilot.transport.source_snapshot = snapshot
    try:
        pilot.main()
    finally:
        pilot.attach_first_frame = original_attach
        pilot.transport.source_snapshot = original_snapshot


if __name__ == '__main__':
    main()
