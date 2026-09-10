from copy import deepcopy

import pytest

from tools.dlss_fi_device_binding import bind_probe


def example():
    return ({"supports_native_2x_by_report": True, "stderr_tail": "[NGXCubinGeneric::SetGPUArch:400] SetGPUArch:: Gpu count = 1, luid: 0x1359b"},
            {"devices": [{"luid": "0x1359b", "node_mask": 1, "uuid": "GPU-fixture", "name": "fixture"}]})


def test_actual_luid_not_name_binds_selected_uuid_and_does_not_qualify_video():
    probe, inventory = example()
    result = bind_probe(probe, inventory, "GPU-fixture")
    assert result["device"]["uuid"] == "GPU-fixture"
    assert not result["generation_qualified"] and not result["quality_qualified"]


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "multi", "duplicate_adapter", "wrong_uuid", "wrong_luid", "mask", "unavailable"])
def test_device_ambiguity_and_mismatch_rejected(fault):
    probe, inventory = example()
    if fault == "missing":
        probe["stderr_tail"] = "GPU fixture"
    elif fault == "ambiguous":
        probe["stderr_tail"] += "\nSetGPUArch:: Gpu count = 1, luid: 0x2"
    elif fault == "multi":
        probe["stderr_tail"] = probe["stderr_tail"].replace("count = 1", "count = 2")
    elif fault == "duplicate_adapter":
        inventory["devices"].append(deepcopy(inventory["devices"][0]))
    elif fault == "wrong_uuid":
        inventory["devices"][0]["uuid"] = "other"
    elif fault == "wrong_luid":
        inventory["devices"][0]["luid"] = "0x2"
    elif fault == "mask":
        inventory["devices"][0]["node_mask"] = 3
    else:
        probe["supports_native_2x_by_report"] = False
    with pytest.raises(ValueError):
        bind_probe(probe, inventory, "GPU-fixture")
