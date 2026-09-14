"""Private instrumented public executor, loaded only by the controlled harness."""
import json
import os
from pathlib import Path

from .observer import ForwardCallAudit

PROJECT = Path(__file__).resolve().parents[2]


def public_node():
    import nodes
    return nodes.NODE_CLASS_MAPPINGS['MiniMaxH3ProgressiveLongVideoEXPT8']


class ObservedPublicLong:
    @classmethod
    def INPUT_TYPES(cls):
        return public_node().INPUT_TYPES()

    RETURN_TYPES = ('VIDEO', 'STRING', 'STRING')
    RETURN_NAMES = ('video', 'video_path', 'report_json')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'T8/Probe only'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    @classmethod
    def execute(cls, **kwargs):
        from comfy.ldm.minimax.model import MiniMaxH3Model
        from comfy.ldm.modules import attention
        if kwargs['total_duration_seconds'] != 3. or kwargs['render_window_frames'] != 124:
            raise ValueError('This observer only qualifies the declared single3s window')
        output = Path(os.environ['T8_H3_ATTENTION_AUDIT_ROOT']).resolve(strict=True)
        if not output.is_relative_to(PROJECT / 'artifacts'):
            raise ValueError('Observer output must be an owned development artifact directory')
        core = Path(attention.__file__).resolve().parents[3]
        observer = ForwardCallAudit({'core_sage': attention.attention_sage,
            'sage_kernel': attention.sageattn, 'pytorch': attention.attention_pytorch},
            MiniMaxH3Model.forward, core / 'custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py')
        failure = None
        try:
            with observer:
                result = public_node().execute(**kwargs)
            if len(observer.forwards) != len(kwargs['sigmas']) - 1 or not all(row['completed'] for row in observer.forwards):
                raise RuntimeError('Observed H3 forwards do not match the actual complete sigma table')
            return result
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            raise
        finally:
            report = observer.report()
            report.update(status='failed' if failure else 'completed_public_execution_observed',
                          error=failure, low_evaluations=kwargs['low_evaluations'],
                          human_qualified=False, no_attention_or_model_replacement=True)
            with (output / 'attention-dispatch.json').open('x', encoding='utf8') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)


NODE_CLASS_MAPPINGS = {'T8ProgressiveObservedLongVideo': ObservedPublicLong}
