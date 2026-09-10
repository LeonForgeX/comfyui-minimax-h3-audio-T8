"""Predeclared long-duration/Sage qualifications; no process or GPU startup."""
from __future__ import annotations

from copy import deepcopy
import inspect
import sys


QUALIFICATION_CASES = ('long32', 'sage_short')


def qualification_recipe(graph, case, qualification):
    if qualification not in QUALIFICATION_CASES or case not in ('T2VA_native8','T2VA_progressive6plus2'):
        raise ValueError('Expected a predeclared T2VA qualification, not another task/recipe')
    result = deepcopy(graph)
    result['18']['inputs']['filename_prefix'] = f'MiniMaxH3/ProgressiveQualification/{qualification}/{case}'
    if qualification == 'long32':
        result['91']['inputs']['scene_duration_seconds'] = 32.0
    else:
        result['106'] = {'class_type':'T8ProgressiveSageBackend','inputs':{'model':['104',0]}}
        result['7']['inputs']['model'] = ['106',0]
        if case.endswith('progressive6plus2'):
            result['10']['class_type'] = 'T8ProgressiveQualifiedSampler'
    return result


def requested_frames(qualification):
    if qualification not in (*QUALIFICATION_CASES,None):
        raise ValueError('Unknown qualification')
    return 768 if qualification == 'long32' else 73


class CallAudit:
    """Thread-local CPython profile observer; no function/attention replacement.

Only used for the isolated Sage pair. Restores the prior profiler on all exits;
refuses an existing profiler rather than taking ownership from another tool.
"""
    def __init__(self, functions):
        self.codes = {}
        for name,function in functions.items():
            code = getattr(inspect.unwrap(function),'__code__',None)
            if code is None or code in self.codes:
                raise ValueError('Attention audit requires distinct Python function identities')
            self.codes[code] = name
        self.counts = {name:{'calls':0,'successful_returns':0,'empty_returns':0} for name in functions}
        self.active = False

    def __enter__(self):
        if self.active or sys.getprofile() is not None:
            raise RuntimeError('Cannot replace an existing profile observer')
        self.active = True
        sys.setprofile(self.observe)
        return self

    def observe(self, frame, event, arg):
        name = self.codes.get(frame.f_code)
        if name is None:
            return
        if event == 'call':
            self.counts[name]['calls'] += 1
        elif event == 'return':
            self.counts[name]['empty_returns' if arg is None else 'successful_returns'] += 1

    def __exit__(self,*args):
        sys.setprofile(None)
        self.active = False


def sage_audit():
    from comfy.ldm.modules import attention
    kernel = getattr(attention,'sageattn',None)
    if kernel is None:
        raise RuntimeError('Installed SageAttention kernel is unavailable; no installation or fallback')
    return CallAudit({'core_sage':attention.attention_sage,'sage_kernel':kernel,'pytorch':attention.attention_pytorch})


def require_sage(counts):
    sage, kernel, pytorch = (counts[name] for name in ('core_sage','sage_kernel','pytorch'))
    calls = sage['calls']
    if calls <= 0 or any(row['calls'] != calls or row['successful_returns'] != calls or row['empty_returns'] for row in (sage,kernel)) or pytorch['calls']:
        raise ValueError('Actual Sage route was missing, failed, or fell back to PyTorch')
    return {'status':'actual_sage_calls_completed_without_pytorch_fallback','counts':deepcopy(counts),
            'scope':'thread_local_Python_function_calls_during_sampler; not_GPU_kernel_timing'}
