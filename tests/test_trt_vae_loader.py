import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import trt_vae_loader as loader
from h3_audio_t8_pkg.nodes_trt_vae import TRT_VAE_NODE_CLASSES,MiniMaxH3TRTVAECheckEXPT8
from test_trt_vae_backend import bundle
from test_trt_vae_contract import core_shell


def runtime(root):
    for name in ('tensorrt/__init__.py','tensorrt_bindings/__init__.py','tensorrt_libs/nvinfer_10.dll',
                 'tensorrt_libs/nvonnxparser_10.dll'):
        p = root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b'CPU fixture, not executable')
    for name in ('tensorrt_cu13','tensorrt_cu13_bindings','tensorrt_cu13_libs'):
        p = root/f'{name}-{loader.TRT_VERSION}.dist-info/METADATA'
        p.parent.mkdir()
        p.write_text('Version: '+loader.TRT_VERSION+'\n',encoding='utf8')


def test_runtime_check_is_static_no_install_or_cuda(tmp_path):
    before = set(sys.modules)
    assert loader.inspect_runtime(tmp_path)['missing']
    runtime(tmp_path)
    report = loader.inspect_runtime(tmp_path)
    assert not report['missing'] and report['status'] == 'static_files_present_not_execution_qualified'
    assert not report['gpu_initialized'] and not report['installed_or_downloaded']
    assert not any(n.split('.')[0] in ('tensorrt','tensorrt_bindings') for n in set(sys.modules)-before)


def test_runtime_relative_paths_rejected_and_default_portable(tmp_path):
    assert loader.runtime_directory('',tmp_path) == tmp_path/'vae/h3_trt/runtime/site-packages'
    with pytest.raises(ValueError,match='绝对路径'):
        loader.runtime_directory('../user-folder',tmp_path)


def test_bundle_inventory_is_metadata_only_and_does_not_select_partial(tmp_path):
    fixture = tmp_path/'fixture'
    fixture.mkdir()
    root,_ = bundle(fixture)
    engines = tmp_path/'models/vae/h3_trt/engines'
    engines.mkdir(parents=True)
    root.rename(engines/'decoder-example')
    (engines/'unfinished').mkdir()
    assert loader.bundle_options(tmp_path/'models','decoder') == ['decoder-example']
    resolved,_ = loader.resolve_bundle('decoder-example',tmp_path/'models','decoder')
    assert resolved == engines/'decoder-example'
    with pytest.raises(ValueError):
        loader.resolve_bundle('../decoder-example',tmp_path/'models','decoder')


def test_public_loader_constructs_only_explicit_wrapper_not_engine(tmp_path,monkeypatch):
    runtime(tmp_path)
    root,_ = bundle(tmp_path)
    native_path = tmp_path/'native.safetensors'
    native_path.write_bytes(b'CPU native fixture')
    monkeypatch.setattr(loader,'NATIVE_VAE_SHA',hashlib.sha256(native_path.read_bytes()).hexdigest())
    events = []
    native = SimpleNamespace(first_stage_model=core_shell(),throw_exception_if_invalid=lambda:None,
                             encode=lambda value:value+1)
    def backend_factory(bundles,site,**kwargs):
        events.append((bundles,site))
        def never(kind,shapes):
            raise AssertionError('Loader must not allocate engine')
        return never
    vae,report = loader.load_vae(native_path=native_path,decoder=root,runtime_site=tmp_path,
                                native_factory=lambda path:native,backend_factory=backend_factory)
    assert len(events) == 1 and report['encoder'] == 'native'
    assert report['status'] == 'vae_interface_prepared_not_executed'
    assert torch.equal(vae.encode(torch.zeros(1)),torch.ones(1))
    native_path.write_bytes(b'changed VAE')
    with pytest.raises(ValueError,match='不能静默替换'):
        loader.load_vae(native_path=native_path,decoder=root,runtime_site=tmp_path,native_factory=lambda path:native)


def test_public_full_loader_requires_distinct_t1_t17_profiles_before_loading_native(tmp_path):
    runtime(tmp_path)
    root,_ = bundle(tmp_path)
    with pytest.raises(ValueError,match='T1单帧'):
        loader.load_vae(native_path='never_read',decoder=root,runtime_site=tmp_path,encoder_paths=[])


def test_four_optional_node_schemas_and_missing_runtime_readable(tmp_path):
    before = set(sys.modules)
    schemas = [node.define_schema().get_v1_info(node) for node in TRT_VAE_NODE_CLASSES]
    assert len(schemas) == 4
    assert schemas[1].output[0] == schemas[2].output[0] == 'VAE'
    result = MiniMaxH3TRTVAECheckEXPT8.execute(str(tmp_path))
    assert result.result[0] is False
    assert json.loads(result.result[2])['installed_or_downloaded'] is False
    assert not any(n.startswith('tensorrt') for n in set(sys.modules)-before)
