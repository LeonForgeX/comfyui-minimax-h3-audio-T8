import os

import pytest

from tests.test_topaz_contract import runtime  # noqa: F401


@pytest.mark.parametrize('key', ['TOPAZLABS_LICENSE', 'TOPAZ_MODEL_STORE', 'TOPAZ_ENGINE_MODE',
                               'topazlabs_license', 'Topaz_Model_Store'])
def test_external_engine_and_license_overrides_not_inherited(runtime, key):  # noqa: F811
    source = {key: 'reference-package-placeholder', 'UNCHANGED': 'keep'}
    child = runtime.child_environment(source)
    assert source[key] == 'reference-package-placeholder'
    assert all(k.upper() != key.upper() for k in child)
    assert child['UNCHANGED'] == 'keep'


@pytest.mark.parametrize('key', ['TVAI_MODEL_DIR', 'tvai_model_dir', 'TVAI_MODEL_DATA_DIR', 'Tvai_Model_Data_Dir'])
def test_windows_case_alias_cannot_preserve_old_model_route(runtime, key):  # noqa: F811
    child = runtime.child_environment({key: 'old-placeholder'})
    assert [k for k in child if k.upper() == key.upper()] == [key.upper()]
    assert child[key.upper()] != 'old-placeholder'


def test_official_program_precedes_inherited_path_without_parent_mutation(runtime):  # noqa: F811
    parent = {'Path': 'some-other-program', 'UNCHANGED': 'keep'}
    child = runtime.child_environment(parent)
    assert [key for key in child if key.upper() == 'PATH'] == ['PATH']
    assert child['PATH'].split(os.pathsep)[0] == str(runtime.install)
    assert parent == {'Path': 'some-other-program', 'UNCHANGED': 'keep'}


def test_conflicting_path_aliases_rejected(runtime):  # noqa: F811
    with pytest.raises(ValueError, match='PATH'):
        runtime.child_environment({'Path': 'one', 'PATH': 'two'})
