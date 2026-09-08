import pytest
from tools import export_vdn_two_pass_workflows as exporter


def test_four_routes_have_distinct_identity_and_use_registered_raw_av_output():
    routes = exporter.candidate_graphs()
    assert len(routes) == 4
    prefixes = set()
    for name, (graph, backend) in routes.items():
        assert graph["18"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
        assert graph["73"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
        assert graph["61"]["inputs"]["width"] == ["60", 1]
        assert graph["62"]["inputs"]["second_pass_audio_strength"] == 0.
        assert graph["7"]["class_type"] == "MiniMaxH3VDNExecutionPlanT8Advanced"
        assert graph["10"]["inputs"]["sigmas"] == ["7", 2]
        assert graph["5"]["inputs"]["stage"] == "stage_dmd_8nfe"
        expected_task = "I2VA" if "_I2VA_" in name else "T2VA"
        assert graph["6"]["inputs"]["task_type"] == expected_task
        assert ("20" in graph) == (expected_task == "I2VA")
        assert all(not node["class_type"].startswith("T8VDN") for node in graph.values())
        assert "50" not in graph
        prefixes.add(graph["18"]["inputs"]["filename_prefix"])
        if backend == "native_h3":
            assert graph["81"]["inputs"]["model"] == ["80", 0]
            assert graph["66"]["inputs"]["sigmas"] == ["82", 1]
        else:
            assert "81" not in graph
    assert len(prefixes) == 4


def test_missing_schema_never_publishes_partial_drafts(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    output = tmp_path / "drafts"
    with pytest.raises(ValueError, match="missing captured schemas"):
        exporter.export_candidates([schema], output)
    assert not output.exists()


def test_existing_drafts_are_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError, match="must be new"):
        exporter.export_candidates([], tmp_path)
