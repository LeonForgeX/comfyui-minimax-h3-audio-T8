import pytest

from tools.run_dual_model_pilot import core_memory_options, transport


def test_optional_memory_recipe_does_not_change_default_or_source_command():
    source = ['python', 'main.py', '--port', '8208', '--reserve-vram', '5']
    assert core_memory_options(source) == source
    assert core_memory_options(source, True) == [*source, '--disable-pinned-memory']
    assert source == ['python', 'main.py', '--port', '8208', '--reserve-vram', '5']


@pytest.mark.parametrize('headroom', [1, 4, float('nan'), True, '2'])
def test_unscoped_memory_headroom_is_rejected(headroom, tmp_path):
    with pytest.raises(ValueError, match='headroom'):
        transport.server_command(tmp_path, 8208, False, headroom)


def test_existing_launch_setting_is_not_silently_overridden():
    with pytest.raises(ValueError, match='already configured'):
        core_memory_options(['python', '--disable-pinned-memory'], True)


def test_native_headroom_is_additional_to_existing_five_gib_reserve(tmp_path):
    command = core_memory_options(transport.server_command(tmp_path, 8208, False, 2), True)
    assert command.count('--reserve-vram') == 1
    assert command[command.index('--reserve-vram') + 1] == '5'
    assert command[command.index('--vram-headroom') + 1] == '2'
    assert command.count('--disable-pinned-memory') == 1
