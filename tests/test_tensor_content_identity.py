"""Byte-equivalent bounded identities; no sampled or cached weight checks."""
import hashlib

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_identity import _update_tensor_digest, value_identity


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
FLOAT8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def _original_digest(value):
    return hashlib.sha256(value.detach().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_finite_edges_keep_original_byte_identity(dtype):
    info = torch.finfo(dtype)
    values = torch.tensor([0., -0., 1., -1., info.max, -info.max, info.tiny, -info.tiny], dtype=dtype)
    subnormal = torch.tensor([info.tiny], dtype=dtype)
    subnormal.view(torch.uint8).zero_()
    subnormal.view(torch.uint8)[0] = 1
    values = torch.cat((values, subnormal))
    identity = value_identity(values)
    assert identity == {'tensor_sha256': _original_digest(values),
                        'dtype': str(dtype), 'shape': list(values.shape)}


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
def test_every_finite_half_storage_pattern_is_hashed_without_conversion(dtype):
    # Actual Torch isfinite is the classification oracle for all 65536 words.
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = words.view(dtype)
    finite = values[torch.isfinite(values)].contiguous()
    assert value_identity(finite)['tensor_sha256'] == _original_digest(finite)
    nonfinite = values[~torch.isfinite(values)].contiguous()
    with pytest.raises(ValueError, match='nonfinite'):
        value_identity(nonfinite)


@pytest.mark.parametrize('dtype', FLOAT8_DTYPES)
def test_every_float8_storage_word_agrees_with_actual_torch_finite_classification(dtype):
    # Include every sign/exponent/mantissa word, not merely NaN representatives.
    # Test one-word chunks to prove each rejection matches the old widened path.
    words = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    values = words.view(dtype)
    finite = torch.isfinite(values.float())
    for index in range(256):
        chunk = values[index:index + 1]
        if finite[index]:
            digest = hashlib.sha256()
            _update_tensor_digest(digest, chunk)
            assert digest.hexdigest() == _original_digest(chunk)
        else:
            with pytest.raises(ValueError, match='nonfinite'):
                _update_tensor_digest(hashlib.sha256(), chunk)


@pytest.mark.parametrize('dtype', FLOAT8_DTYPES)
def test_float8_content_identity_keeps_one_mib_chunks_and_fresh_byte_checks(dtype, monkeypatch):
    from h3_audio_t8_pkg.long_video_dual_identity import content_identity
    stride = 1024 * 1024
    values = torch.ones(stride + 1, dtype=torch.uint8).view(dtype)
    copied = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copied.append(tensor.numel() * tensor.element_size())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    assert content_identity(values)['tensor_sha256'] == _original_digest(values)
    assert copied == [stride, 1]
    before = content_identity(values)
    values.data.view(torch.uint8)[-1] = 2
    assert content_identity(values) != before
    all_words = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    bad_word = all_words[~torch.isfinite(all_words.view(dtype).float())][0]
    values.data.view(torch.uint8)[-1] = bad_word
    with pytest.raises(ValueError, match='nonfinite'):
        content_identity(values)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('invalid', (float('nan'), float('inf'), -float('inf')))
def test_nonfinite_in_last_chunk_is_rejected(dtype, invalid):
    stride = 4 * 1024 * 1024 // torch.empty((), dtype=dtype).element_size()
    values = torch.ones(stride + 1, dtype=dtype)
    values[-1] = invalid
    with pytest.raises(ValueError, match='nonfinite'):
        value_identity(values)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_in_place_data_mutations_are_freshly_checked_at_every_call(dtype):
    values = torch.ones(32, dtype=dtype)
    before = value_identity(values)
    version = values._version
    values.data[-1] = 2.
    assert values._version == version  # A version-token cache would miss this.
    assert value_identity(values) != before
    values.data[-1] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        value_identity(values)


@pytest.mark.parametrize('dtype', (torch.bool, torch.uint8, torch.int8, torch.int16,
                                 torch.int32, torch.int64, torch.complex64, torch.complex128))
def test_existing_nonfloating_dtypes_keep_original_hash(dtype):
    values = torch.tensor([0, 1, 0, 1], dtype=dtype)
    assert value_identity(values)['tensor_sha256'] == _original_digest(values)


def test_scalar_empty_and_multidimensional_shape_are_preserved():
    for values in (torch.tensor(1.), torch.empty(0), torch.ones(2, 3, 4)):
        identity = value_identity(values)
        assert identity['shape'] == list(values.shape)
        assert identity['tensor_sha256'] == _original_digest(values)


def test_chunks_keep_transfer_and_interruption_bound(monkeypatch):
    stride = 4 * 1024 * 1024 // 4
    values = torch.ones(2 * stride + 17)
    copied = []
    checks = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copied.append(tensor.numel() * tensor.element_size())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    result = value_identity(values, interrupt_check=lambda: checks.append(True))
    assert copied == [4 * 1024 * 1024, 4 * 1024 * 1024, 17 * 4]
    assert len(checks) == 3
    assert result['tensor_sha256'] == _original_digest(values)

    def cancel():
        raise RuntimeError('cancel identity')

    with pytest.raises(RuntimeError, match='cancel identity'):
        value_identity(values, interrupt_check=cancel)
    assert len(copied) == 3


def test_hash_update_receives_shared_readonly_bytes_not_a_copy():
    values = torch.tensor([1., 2., 3.])

    class Observer:
        def update(self, buffer):
            assert isinstance(buffer, memoryview)
            assert buffer.nbytes == values.numel() * values.element_size()
            assert buffer.obj.__array_interface__['data'][0] == values.data_ptr()
            self.digest = hashlib.sha256(buffer).hexdigest()

    observer = Observer()
    _update_tensor_digest(observer, values)
    assert observer.digest == _original_digest(values)


@pytest.mark.parametrize('value', (torch.empty(2, device='meta'), torch.ones(2, 3).T,
                                 torch.ones(2, 3).to_sparse(),
                                 torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)))
def test_unsupported_tensors_still_fail_without_materializing(value):
    with pytest.raises(ValueError, match='materializing'):
        value_identity(value)
