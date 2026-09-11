"""R1 incremental negative cases; no new global Core patches."""
from itertools import combinations

import pytest

from h3_audio_t8_pkg.h3_core_compat import call_h3_final_layer


@pytest.mark.parametrize('names', [item for size in (1, 2)
    for item in combinations(('sigma', 'sample_sigmas', 'shifts'), size)])
def test_each_partial_final_schedule_is_rejected_before_call(names):
    namespace = {}
    # Test-generated signatures enumerate all six partial API shapes.
    exec('def forward(self, x, t, video, audio, *, ' + ', '.join(names)
         + '):\n    raise AssertionError("partial layer must never execute")', namespace)
    layer_type = type('PartialFinal', (), {'forward': namespace['forward'],
        '__call__': lambda self, *a, **kw: self.forward(*a, **kw)})
    with pytest.raises(RuntimeError, match='Unsupported partial H3 FinalLayer schedule interface'):
        call_h3_final_layer(layer_type(), 1, 2, 3, 4, sigma=5, sample_sigmas=6, shifts=7)


def test_complete_schedule_preserves_identity_and_does_not_swallow_typeerror():
    sentinel = object()
    class CompleteFinal:
        def forward(self, x, t, video, audio, *, sigma, sample_sigmas, shifts):
            assert sigma is sample_sigmas is shifts is sentinel
            raise TypeError('internal layer error must propagate unchanged')
        __call__ = forward
    with pytest.raises(TypeError, match='internal layer error must propagate unchanged'):
        call_h3_final_layer(CompleteFinal(), 1, 2, 3, 4,
            sigma=sentinel, sample_sigmas=sentinel, shifts=sentinel)
