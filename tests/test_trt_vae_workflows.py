from copy import deepcopy
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.nodes_trt_vae import TRT_VAE_NODE_CLASSES
from tools.build_trt_vae_workflows import build,recipe
from tools.audit_progressive_workflows import audit_candidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode',['check','compile'])
def test_prepare_workflow_actual_schema_and_serialized_controls(mode):
    info = {c.define_schema().node_id:json.loads(json.dumps(c.define_schema().get_v1_info(c).__dict__))
            for c in TRT_VAE_NODE_CLASSES}
    graph,workflow = build(None,info,mode,mode)
    report = audit_candidate(graph,workflow,info)
    assert report['nodes'] == 1
    assert workflow['nodes'][0]['widgets_values'][0] == ('' if mode == 'check' else 'decoder-flex')
    assert '1.78.0' in workflow['nodes'][-1]['widgets_values'][0]


@pytest.mark.parametrize('mode',['decoder','full'])
def test_generation_changes_only_video_vae_not_sampler_or_audio(mode):
    source = {'1':{'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_fp16.safetensors'}},
              '2':{'class_type':'VAELoader','inputs':{'vae_name':'audio.safetensors'}},
              '3':{'class_type':'ExistingSampler','inputs':{'seed':23,'steps':8}}}
    before = deepcopy(source)
    result = recipe(source,mode)
    assert source == before and result['2'] == source['2'] and result['3'] == source['3']
    assert result['1']['inputs']['runtime_directory'] == ''
    assert ('image_encoder_engine' in result['1']['inputs']) == (mode == 'full')


def test_public_worker_does_not_depend_on_excluded_research_tools():
    old = (ROOT/'tools/trt_vae_build_worker.py').read_text(encoding='utf8')
    new = (ROOT/'h3_t8/trt_vae_compile_worker.py').read_text(encoding='utf8')
    assert new.strip() == old.replace('sys.path.insert(0, str(PROJECT / "h3_t8"))\n', '').replace('Path(__file__).resolve().parents[1]',
                                     'Path(__file__).resolve().parent').strip()
    for name in ('trt_vae_compile.py','trt_vae_compile_worker.py','nodes_trt_vae.py','trt_vae_loader.py'):
        text = (ROOT/'h3_t8'/name).read_text(encoding='utf8')
        assert 'from tools' not in text and 'import tools' not in text
        assert 'artifacts/acceleration-research' not in text


def test_delivered_workflows_are_portable_and_no_long_test():
    paths = list((ROOT/'examples/workflows/30-trt-vae').glob('*.json'))
    assert len(paths) == 5
    for path in paths:
        data = json.loads(path.read_text(encoding='utf8'))
        assert data['extra']['trt_vae_delivery_status'].startswith('exp_release_1_78_0')
        text = json.dumps(data)
        assert 'F:/' not in text and 'F:\\' not in text and 'G:/' not in text
        for node in data['nodes']:
            assert node['type'] != 'MiniMaxH3LongVideoInNodeLoopT8Advanced'


def test_comparison_one_sampler_audio_decode_and_ordered_exact_latent():
    source = {'1':{'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_fp16.safetensors'}},
        '10':{'class_type':'SamplerCustomAdvanced','inputs':{'noise':['8',0]}},
        '11':{'class_type':'MiniMaxH3AVDecodeT8','inputs':{'av_latent':['10',0],'video_vae':['1',0],'audio_vae':['2',0]}},
        '12':{'class_type':'MiniMaxH3OutputTrimT8','inputs':{'frames':['11',0],'audio':['11',1],'start_seconds':0,'duration_seconds':3,'fps':24}},
        '18':{'class_type':'MiniMaxH3SafeAVSaveT8Advanced','inputs':{'images':['12',0],'audio':['12',1],'crf':18,'filename_prefix':'old'}}}
    before = deepcopy(source)
    result = recipe(source,'compare')
    assert source == before and result['10'] == source['10'] and result['11'] == source['11']
    assert sum(n['class_type']=='SamplerCustomAdvanced' for n in result.values()) == 1
    assert sum(n['class_type']=='MiniMaxH3AVDecodeT8' for n in result.values()) == 1
    assert result['102']['inputs']['samples'] == ['11',2]
    assert result['104']['inputs']['audio'] == result['18']['inputs']['audio'] == ['12',1]
    assert result['104']['inputs']['images'] == ['103',0]
    assert result['103']['inputs']['frames'] == ['102',0]
