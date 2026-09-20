"""Formal namespace preservation and public Director CPU integration."""

import asyncio
import json

import h3_audio_t8_pkg
from h3_audio_t8_pkg import nodes_director, nodes_h16_chunked_pass2
from h3_audio_t8_pkg.director_project import ProjectStore, new_project
from h3_audio_t8_pkg.director_routes import export_preflight_workflow

def test_registry_has_unique_ids_and_new_nodes_are_append_only_at_the_tail():
    current = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    node_ids = [node_class.define_schema().node_id for node_class in current]
    assert len(current) == 356
    assert len(node_ids) == len(set(node_ids))
    assert current[-2] is nodes_director.MiniMaxH3DirectorProjectT8
    assert current[-1] is nodes_h16_chunked_pass2.DeciiaChunkedPass2Sampler
    assert h3_audio_t8_pkg.WEB_DIRECTORY == "./web"


def test_export_and_real_D1_node_execute_without_queue_or_private_fixtures(
    tmp_path, monkeypatch
):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "中文原文\r\n第二行  🙂"
    project = store.save(project, 0)
    graph = export_preflight_workflow(project, project["current"])
    assert graph["api_snapshot"]["1"]["class_type"] == "MiniMaxH3DirectorProjectT8"
    api_inputs = graph["api_snapshot"]["1"]["inputs"]
    assert json.loads(api_inputs["project_json"]) == project
    assert api_inputs["shot_id"] == project["current"]
    monkeypatch.setattr(nodes_director, "get_store", lambda: store)
    actual = nodes_director.MiniMaxH3DirectorProjectT8.execute(
        json.dumps(project), project["current"]
    ).result
    assert actual[0] == project["doc"]["shots"][0]["simplePrompt"]
    assert actual[1:4] == (800, 448, 107)
    assert json.loads(actual[-1])["gpu_queued"] is False
