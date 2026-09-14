"""Thread-local call observation gated to actual H3 forward frames; no patching."""
from copy import deepcopy
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progressive_qualification import CallAudit  # noqa: E402


class ForwardCallAudit(CallAudit):
    def __init__(self, functions, forward, ffn_file=None):
        super().__init__(functions)
        self.forward_code = inspect.unwrap(forward).__code__
        self.ffn_file = str(Path(ffn_file).resolve()) if ffn_file else None
        self.forwards = []
        self.current = None

    def observe(self, frame, event, arg):
        if frame.f_code is self.forward_code:
            if event == 'call':
                if self.current is not None:
                    raise RuntimeError('Nested H3 forward is outside this probe')
                self.current = {'index': len(self.forwards), 'completed': False,
                    'counts': {name: {'calls': 0, 'successful_returns': 0, 'empty_returns': 0} for name in self.counts},
                    'ffn_calls': 0, 'ffn_chunk_eligible_calls': 0}
                self.forwards.append(self.current)
            elif event == 'return':
                self.current['completed'] = arg is not None
                self.current = None
            return
        if self.current is None:
            return
        name = self.codes.get(frame.f_code)
        key = 'calls' if event == 'call' else ('empty_returns' if arg is None else 'successful_returns') if event == 'return' else None
        if name is not None and key is not None:
            self.current['counts'][name][key] += 1
            self.counts[name][key] += 1
        if (event == 'call' and self.ffn_file is not None
                and frame.f_code.co_name == 'minimax_mlp_chunked_forward'
                and frame.f_code.co_filename == self.ffn_file):
            self.current['ffn_calls'] += 1
            module, value = frame.f_locals['self'], frame.f_locals['x']
            if value.shape[0] > module.kj_seq_threshold and module.kj_num_chunks > 1:
                self.current['ffn_chunk_eligible_calls'] += 1

    def report(self):
        return deepcopy(dict(forwards=self.forwards, counts=self.counts,
            scope='Python function entries/returns inside actual H3.forward only; excludes text encoding/VAE/hash verification; not GPU kernel timing',
            profiler_still_owned=self.active))
