from __future__ import annotations

import asyncio
import sys
import types

from h3_audio_t8_pkg.director_generation import (
    cancel_director_prompt,
    director_job_status,
    queue_director_prompt,
)


class _Queue:
    def __init__(self):
        self.items = []
        self.history = {}

    def put(self, item):
        self.items.append(item)

    def delete_queue_item(self, predicate):
        for index, item in enumerate(self.items):
            if predicate(item):
                self.items.pop(index)
                return True
        return False

    def interrupt_if_running(self, _prompt_id):
        return False

    def get_current_queue(self):
        return ([], list(self.items))

    def get_history(self, prompt_id=None):
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}


def test_queue_status_and_cancel_use_core_queue_contract(monkeypatch):
    queue = _Queue()
    prompt_server = types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(
            instance=types.SimpleNamespace(
                number=7,
                prompt_queue=queue,
                trigger_on_prompt=lambda prompt: prompt,
                node_replace_manager=types.SimpleNamespace(apply_replacements=lambda _prompt: None),
            )
        )
    )
    execution = types.SimpleNamespace(
        validate_prompt=lambda *_args: asyncio.sleep(0, result=(True, "ok", {"node_errors": {}}))
    )
    monkeypatch.setitem(sys.modules, "server", prompt_server)
    monkeypatch.setitem(sys.modules, "execution", execution)

    prompt_id = asyncio.run(queue_director_prompt({"1": {"class_type": "Test", "inputs": {}}}, "tab-1"))
    assert prompt_id
    assert queue.items[0][1] == prompt_id
    assert queue.items[0][3]["client_id"] == "tab-1"
    assert director_job_status(prompt_id)["state"] == "queued"
    assert cancel_director_prompt(prompt_id)["deleted_from_queue"] is True
    assert director_job_status(prompt_id)["state"] == "unknown"

    queue.history[prompt_id] = {
        "status": {"status_str": "success"},
        "outputs": {"12": {"videos": [{"filename": "director.mp4"}] }},
    }
    assert director_job_status(prompt_id)["state"] == "success"
