from __future__ import annotations

import asyncio
import json
import sys
import types

from h3_audio_t8_pkg.director_project import ProjectStore


def test_generate_request_id_returns_existing_prompt_without_double_queue(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_routes

    handlers = {}

    class Routes:
        def post(self, path):
            def register(handler):
                handlers[("POST", path)] = handler
                return handler
            return register

        get = post

    server = types.SimpleNamespace(routes=Routes())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, "_REGISTERED", False)
    monkeypatch.setattr(director_routes, "get_store", lambda: ProjectStore(tmp_path / "user", tmp_path / "input"))
    monkeypatch.setattr(director_routes, "build_director_generation_prompt", lambda *_args, **_kwargs: {
        "prompt": {"1": {"class_type": "Test", "inputs": {}}},
        "recipe": "director_two_pass", "seed": 1, "turbo_lora": None,
        "sampling": {"mode": "two_pass"}, "report": {"ready": True},
    })
    submitted = []

    async def queue(_prompt, _client_id, prompt_id=None):
        submitted.append(prompt_id)
        return prompt_id

    monkeypatch.setattr(director_routes, "queue_director_prompt", queue)
    director_routes.register_director_routes()
    generate = handlers[("POST", director_routes.PREFIX + "/generate")]
    assert ("POST", director_routes.PREFIX + "/sampling_ui.mjs") in handlers
    body = {
        "request_id": "e42bea1f-f961-4ef7-9c48-b181490b6f17",
        "project": {"id": "302b352d-14a3-4f90-a923-e0528e7efed0"},
        "shot_id": "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce", "seed": 1,
    }

    class Request:
        async def json(self):
            return body

    first = asyncio.run(generate(Request()))
    second = asyncio.run(generate(Request()))
    assert first.status == 202 and second.status == 200
    assert len(submitted) == 1
    assert json.loads(first.text)["prompt_id"] == json.loads(second.text)["prompt_id"]
    body["seed"] = 2
    rejected = asyncio.run(generate(Request()))
    assert rejected.status == 400
    assert len(submitted) == 1


def test_results_recover_existing_receipt_and_cache_terminal_media(tmp_path):
    from h3_audio_t8_pkg.director_routes import director_project_results

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project_id = "302b352d-14a3-4f90-a923-e0528e7efed0"
    shot_id = "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce"
    prompt_id = "e42bea1f-f961-4ef7-9c48-b181490b6f17"
    request_file = store.root / "requests" / "a.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps({
        "state": "queued", "prompt_id": prompt_id,
        "result": {"recipe": "director_two_pass", "report": {
            "project_id": project_id, "selection": {"shot_id": shot_id},
        }},
    }), encoding="utf-8")
    other = store.root / "requests" / "b.json"
    other.write_text(json.dumps({"state": "queued", "prompt_id": prompt_id,
                                 "project_id": "316378f7-18b0-456d-bc84-ac089e141fc7"}), encoding="utf-8")
    media = {"12": {"images": [{"filename": "film_00001_.mp4", "subfolder": "T8_Director\\abc", "type": "output"}]}}
    calls = []

    def status(queried):
        calls.append(queried)
        return {"state": "success", "outputs": media}

    results = director_project_results(store, project_id, status)["results"]
    assert len(results) == 1
    assert results[0]["shot_id"] == shot_id
    assert results[0]["outputs"] == media
    assert results[0]["state"] == "success"
    assert calls == [prompt_id]
    assert director_project_results(store, project_id, status)["results"] == results
    assert calls == [prompt_id]  # persisted media remains discoverable after Core restart


def test_results_recover_pre_receipt_video_from_exact_project_and_shot_prefix(tmp_path):
    from h3_audio_t8_pkg.director_routes import director_project_results

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project_id = "302b352d-14a3-4f90-a923-e0528e7efed0"
    shot_id = "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce"
    media_dir = tmp_path / "output" / "T8_Director" / project_id[:8]
    media_dir.mkdir(parents=True)
    (media_dir / f"{shot_id[:8]}_00001_.mp4").write_bytes(b"video")
    (media_dir / "ffffffff_00001_.mp4").write_bytes(b"other shot")
    records = director_project_results(store, project_id, lambda _id: {},
                                       shot_ids=[shot_id], output_root=tmp_path / "output")["results"]
    assert len(records) == 1
    assert records[0]["recovered_by"] == "project_and_shot_output_prefix"
    assert records[0]["outputs"]["legacy"]["images"][0]["filename"] == f"{shot_id[:8]}_00001_.mp4"
