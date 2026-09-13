"""Execute the real worker entry with tiny CPU doubles, never H3/GPU inference."""
from contextlib import contextmanager
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from h3_audio_t8_pkg.prepared_backend.backend_files import sha


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / 'h3_t8/prepared_backend'
    monkeypatch.syspath_prepend(str(backend))
    spec = importlib.util.spec_from_file_location('tao_worker_lifecycle_test', backend / 'taomate_video_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    state = {
        'refs': {}, 'removed': 0, 'released': 0, 'postflight': 0, 'fault': None,
        # Importing the complete Comfy extension may already initialize CUDA on
        # some Core builds. This worker must not change that process-level state.
        'cuda_initialized_before': torch.cuda.is_initialized(),
    }

    def module(name, **values):
        value = ModuleType(name)
        value.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, value)

    class Handle:
        def __init__(self, block):
            self.block = block

        def remove(self):
            self.block.hook = None
            state['removed'] += 1

    class Block:
        def register_forward_hook(self, hook):
            self.hook = hook
            return Handle(self)

    class Model:
        def __init__(self):
            self.blocks = [Block() for _ in range(50)]
            self.weight = torch.nn.Parameter(torch.zeros(1))
            state['refs']['model'] = weakref.ref(self)

        @classmethod
        def allocate(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def named_parameters(self):
            return [('weight', self.weight)]

        def to_empty(self, **kwargs):
            assert kwargs == {'device': 'cpu'}

    class Execution:
        denoise_forwards = 12
        clean_forwards = phase_count = 4
        audio_teacher = {'published_clean_audio_exact_match': True}
        attention_backend = {'backend': 'torch_CUDA_SDPA_not_FA3', 'cache_transfer_bytes': 1}

        def to_dict(self):
            return {'denoise_forwards': 12, 'clean_forwards': 4}

    class Runtime:
        def __init__(self):
            self.executions = [Execution()]
            self._cache = SimpleNamespace(host=SimpleNamespace(retention_receipts=[{'layers': 50}]))
            self.last_weight_receipt = {'block_calls': [0] * 800}

        def run(self):
            pass

        def dit_timing_receipt(self):
            return {'cpu_fixture': True}

        def release_retained_state(self):
            state['released'] += 1
            self._cache = None
            if state['fault'] == 'cleanup':
                raise RuntimeError('synthetic cleanup failure')

    class Pipeline:
        def __init__(self, model):
            self.transformer = model
            # Exercise cyclic ownership: scope exit alone is not sufficient.
            self.cycle = self
            state['refs']['pipeline'] = weakref.ref(self)

        def generate(self, **kwargs):
            if state['fault'] == 'generation':
                raise RuntimeError('synthetic generation failure')
            assert kwargs['video_noise_seed'] == kwargs['audio_noise_seed'] == 8301
            assert kwargs['width'] == 864 and kwargs['height'] == 480
            for _ in range(16):
                for block in self.transformer.blocks:
                    block.hook(None, None, None)
            return SimpleNamespace(video_latents=torch.zeros(1, 24, 37, 30, 54), audio_latents=known)

    known = torch.zeros(2, 32, 207)

    @contextmanager
    def transport():
        yield object()

    module('taomate_h3.model.dit', MiniMaxH3DiT=Model)
    module('taomate_h3.model.weight_loading', load_minimax_h3_dit_weights=lambda *args: {'fixture': True})
    module('taomate_h3.model.layers', set_inference_packed_dense_flash_attn3=lambda value: None)
    module('taomate_h3.inference.lora_checkpoint', apply_h3_lora_checkpoint=lambda *args: {},
           materialize_h3_lora_bf16_buffers_=lambda *args: {})
    module('taomate_input_preflight', load_inputs=lambda *args: (
        {'hidden': torch.zeros(1), 'tags': torch.zeros(1)}, 'fixture', object(), known, {}))
    module('taomate_local_transport', local_transport=transport)
    module('taomate_local_runtime', make_local_runtime=lambda *args, **kwargs: Runtime())
    module('taomate_prepared_pipeline', prepared_pipeline=lambda model, context: Pipeline(model))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(uuid='GPU-fixture'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=128 * 1024**3))
    monkeypatch.setattr(worker.subprocess, 'check_output', lambda args, **kwargs:
                        'fixture-revision\n' if 'rev-parse' in args else '')
    cpu = tmp_path / 'cpu.json'
    cpu.write_text(json.dumps({'status': 'full_official_DiT_packed_CPU_forward_pass',
        'weight_offload': {'output_bitexact': True}, 'cuda_initialized': False, 'parameters': 33122992896}))
    download = tmp_path / 'download.json'
    download.write_text(json.dumps({'status': 'download_and_transformer_checksums_pass', 'total_bytes': 1, 'files': []}))
    asset = tmp_path / 'input.dat'
    asset.write_bytes(b'immutable tiny input')
    request = {'schema': 't8-taomate-prepared-first-request-v1', 'video_seed': 8301, 'audio_seed': 8301,
        'identities': {str(asset): sha(asset)}, 'source': str(tmp_path), 'source_revision': 'fixture-revision',
        'cpu_receipt': str(cpu), 'download_receipt': str(download), 'base': str(tmp_path),
        'adapter': str(tmp_path / 'adapter.safetensors'), 'gpu_uuid': 'GPU-fixture'}
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request))
    monkeypatch.setattr(sys, 'argv', ['worker', '--request', str(path)])
    seen = []

    def hash_file(path):
        if Path(path) == asset:
            seen.append(path)
            if len(seen) == 2:
                state['postflight'] += 1
                state['alive_at_postflight'] = {name: ref() is not None for name, ref in state['refs'].items()}
                if state['fault'] == 'postflight':
                    return '0' * 64
        return sha(path)

    monkeypatch.setattr(worker, 'sha', hash_file)
    return worker, state, tmp_path


def test_full_entry_releases_model_and_pipeline_before_postflight_hashing(worker_case, capsys):
    worker, state, root = worker_case
    worker.main()
    assert state['alive_at_postflight'] == {'model': False, 'pipeline': False}
    assert state['removed'] == 50 and state['released'] == 1
    report = json.loads((root / 'report.json').read_text())
    assert report['status'] == 'first_Tao_request_latents_pass'
    assert report['clean_audio_bitexact'] and report['retained_state_released']
    assert report['generation_scope_released_before_postflight']
    from safetensors.torch import load_file
    latent = load_file(root / 'tao-normalized-latents.safetensors')
    assert torch.count_nonzero(latent['video']) == 0 and torch.count_nonzero(latent['audio']) == 0
    assert torch.cuda.is_initialized() == state['cuda_initialized_before']
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    phases = [event['stage'] for event in events if 'stage' in event]
    assert phases[-5:] == ['building_generation_receipt', 'generation_cache_released',
                          'generation_scope_released', 'postflight_input_hashes', 'postflight_complete']
    for event in events:
        if 'stage' in event:
            assert event['memory']['process_rss_bytes'] > 0
            assert event['memory']['system_available_bytes'] == 128 * 1024**3


@pytest.mark.parametrize('fault', ['generation', 'cleanup', 'postflight'])
def test_failure_never_reports_success_or_runs_postflight_before_cleanup(worker_case, fault):
    worker, state, root = worker_case
    state['fault'] = fault
    with pytest.raises((RuntimeError, AssertionError)):
        worker.main()
    report = json.loads((root / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert state['removed'] == 50 and state['released'] == 1
    assert state['postflight'] == (1 if fault == 'postflight' else 0)
    assert torch.cuda.is_initialized() == state['cuda_initialized_before']


def test_generation_body_is_the_original_math_with_only_progress_events_added():
    root = Path(__file__).resolve().parents[1]
    before = ast.parse((root / 'tests/fixtures/tao_video_worker_before_finalization.py').read_text())
    current = ast.parse((root / 'h3_t8/prepared_backend/taomate_video_worker.py').read_text())
    main = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    guarded = next(node for node in main.body if isinstance(node, ast.Try)).body
    start = next(i for i, node in enumerate(guarded)
                 if isinstance(node, ast.Expr) and ast.unparse(node).startswith('sys.path.insert'))
    expected = guarded[start:-1]  # Postflight identity loop stays in the caller.
    generated = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == '_generate')

    class WithoutProgress(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'progress'):
                return None
            return self.generic_visit(node)

    def normalized(body):
        tree = ast.Module(body=body, type_ignores=[])
        return ast.dump(WithoutProgress().visit(tree), include_attributes=False)

    assert normalized(generated.body[1:]) == normalized(expected)
