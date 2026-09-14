"""Layout-only invariants: public namespace, workers and legacy CLI entrypoints."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

import h3_audio_t8_pkg as package

ROOT = Path(__file__).resolve().parents[1]


def test_root_stays_small_and_web_workflows_keep_their_paths():
    assert {p.name for p in ROOT.glob('*.py')} == {
        '__init__.py', 'trt_vae_compile.py', 'trt_vae_prepare_flex.py', 'trt_vae_prepare_t1.py'}
    assert package.WEB_DIRECTORY == './web'
    assert (ROOT / package.WEB_DIRECTORY).is_dir()
    assert len(list((ROOT / 'examples/workflows').rglob('*.json'))) == 254


@pytest.mark.parametrize('name', ['nodes', 'sampling', 'vdn_h3_advanced', 'trt_vae_loader',
                                 'dlss_fi_backend.process', 'video_outpaint_inspection',
                                 'skin_finish_vretoucher_runtime'])
def test_old_qualified_import_names_resolve_once_to_internal_files(name):
    first = importlib.import_module(package.__name__ + '.' + name)
    second = importlib.import_module(package.__name__ + '.' + name)
    assert first is second
    assert Path(first.__file__).resolve().is_relative_to(ROOT / 'h3_t8')


def test_internal_path_precedes_legacy_root_leftovers():
    assert list(package.__path__) == [str(ROOT / 'h3_t8'), str(ROOT)]
    assert importlib.util.find_spec(package.__name__ + '.nodes').origin == str(ROOT / 'h3_t8/nodes.py')


def test_vendored_license_and_worker_paths_still_exist():
    from h3_audio_t8_pkg.skin_finish_vretoucher_runtime import bundled_vretoucher_source_root
    from h3_audio_t8_pkg.video_outpaint_inspection import WORKER_PATH
    assert (bundled_vretoucher_source_root() / 'LICENSE').is_file()
    assert WORKER_PATH == ROOT / 'h3_t8/video_outpaint_inspection_worker.py'
    assert WORKER_PATH.is_file()


@pytest.mark.parametrize('entry', ['trt_vae_compile.py', 'trt_vae_prepare_flex.py', 'trt_vae_prepare_t1.py'])
def test_legacy_cli_help_runs_without_a_gpu_or_runtime_install(entry, tmp_path):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / entry), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
