import pytest

from tools.run_dlss_fi_frame_probe import rgb_identity, verify_decoder


def test_alpha_only_change_cannot_fake_rgb_interpolation():
    a, pixels = rgb_identity(bytes([1,2,3,255]),1,1)
    b, _ = rgb_identity(bytes([1,2,3,0]),1,1)
    c, _ = rgb_identity(bytes([1,2,4,255]),1,1)
    assert a == b and a != c and pixels.shape == (1,1,3)


@pytest.mark.parametrize('payload', [b'',b'123',b'12345',bytearray(b'1234')])
def test_malformed_color_frame_is_rejected(payload):
    with pytest.raises(ValueError):
        rgb_identity(payload,1,1)


def test_wrong_decoder_identity_rejected_before_running(monkeypatch):
    monkeypatch.setattr('tools.run_dlss_fi_frame_probe.file_identity', lambda p: {'sha256': str(p)})
    source = {'media': {'binaries': {'ffmpeg': {'sha256': 'correct'}}}}
    verify_decoder('correct', source)
    with pytest.raises(ValueError, match='audited FFmpeg'):
        verify_decoder('ffprobe', source)
