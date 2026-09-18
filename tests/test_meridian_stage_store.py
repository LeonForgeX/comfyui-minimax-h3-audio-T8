import json

import pytest
import torch

from h3_audio_t8_pkg.meridian_checkpoint_io import atomic_json, conversion_lock
from h3_audio_t8_pkg.meridian_stage_store import StageStore


def test_atomic_complete_warm_reuse_and_tuple_contract(tmp_path):
    store = StageStore(tmp_path)
    contract = dict(shape=(1, 2))
    value, first = store.run("geometry", contract, lambda: dict(x=torch.arange(3)))
    second, reused = store.run(
        "geometry", contract, lambda: pytest.fail("cache operation repeated")
    )
    assert (
        torch.equal(value["x"], second["x"])
        and reused["reused"]
        and not first["reused"]
    )
    assert not list(tmp_path.glob("*.partial-*"))


def test_corrupt_payload_never_returns_success(tmp_path):
    store = StageStore(tmp_path)
    contract = dict(source="same")
    store.run("warp", contract, lambda: torch.arange(3))
    _, data, _ = store.paths("warp", contract)
    with data.open("r+b") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        store.run("warp", contract, lambda: 0)


def test_cancel_keeps_receipt_and_next_same_contract_recovers(tmp_path):
    state = dict(cancel=False)
    store = StageStore(tmp_path, cancelled=lambda: state["cancel"])

    def cancelled():
        state["cancel"] = True
        return torch.arange(4)

    with pytest.raises(InterruptedError):
        store.run("encode", {}, cancelled)
    _, data, receipt = store.paths("encode", {})
    assert (
        not data.exists() and json.loads(receipt.read_text())["status"] == "interrupted"
    )
    state["cancel"] = False
    value, record = store.run("encode", {}, lambda: torch.arange(4))
    assert record["status"] == "complete" and len(value) == 4


def test_restart_running_receipt_preserved(tmp_path):
    store = StageStore(tmp_path)
    key, _, receipt = store.paths("sample", {})
    atomic_json(receipt, dict(status="running", identity=key, contract={}, pid=12345))
    store.run("sample", {}, lambda: torch.arange(4))
    old = list(tmp_path.glob("*.interrupted-*.json"))
    assert len(old) == 1 and json.loads(old[0].read_text())["status"] == "interrupted"


def test_budget_too_small_or_opaque_external_returns_not_cached(tmp_path):
    store = StageStore(tmp_path, budget_bytes=10)
    _, record = store.run("sample", {}, lambda: torch.arange(4))
    assert record["status"] == "complete_not_cached" and not list(tmp_path.glob("*.pt"))
    _, record = store.run("encode", {}, lambda: torch.arange(4), portable=False)
    assert record["status"] == "complete_not_cached"


@pytest.mark.parametrize("portable,budget", [(False, 100000), (True, 0)])
def test_uncached_failure_receipt_durable_and_next_call_recovers(
    tmp_path, portable, budget
):
    store = StageStore(tmp_path, budget_bytes=budget)

    def fail():
        raise RuntimeError("real failure kept")

    with pytest.raises(RuntimeError, match="real failure"):
        store.run("sample", {}, fail, portable=portable)
    failed = list(tmp_path.glob("*.uncached-*.json"))
    assert len(failed) == 1 and json.loads(failed[0].read_text())["status"] == "failed"
    _, record = store.run("sample", {}, lambda: torch.arange(3), portable=portable)
    assert record["status"] == "complete_not_cached" and record["portable"] is False
    assert len(list(tmp_path.glob("*.uncached-*.json"))) == 2
    assert not list(tmp_path.glob("*.pt"))
    assert json.loads(failed[0].read_text())["status"] == "failed"


def test_uncached_core_cancel_is_not_cached_success(tmp_path):
    import comfy.model_management as mm

    store = StageStore(tmp_path)

    def cancel():
        raise mm.InterruptProcessingException()

    with pytest.raises(mm.InterruptProcessingException):
        store.run("encode", {}, cancel, portable=False)
    receipt = next(tmp_path.glob("*.uncached-*.json"))
    assert json.loads(receipt.read_text())["status"] == "interrupted"
    assert not list(tmp_path.glob("*.pt"))


def test_eviction_only_owned_complete_payload(tmp_path):
    store = StageStore(tmp_path, budget_bytes=100000)
    store.run("warp", {"id": 1}, lambda: torch.arange(1000))
    _, old, _ = store.paths("warp", {"id": 1})
    store.budget = old.stat().st_size + 100
    store.run("warp", {"id": 2}, lambda: torch.arange(1000))
    assert (
        not old.exists()
        and sum(p.stat().st_size for p in tmp_path.glob("*.pt")) <= store.budget
    )


def test_locked_cache_no_over_budget_admission(tmp_path):
    store = StageStore(tmp_path, budget_bytes=100000)
    store.run("warp", {"id": 1}, lambda: torch.arange(1000))
    _, old, _ = store.paths("warp", {"id": 1})
    store.budget = old.stat().st_size + 100
    with conversion_lock(old.with_suffix(".lock")):
        _, record = store.run("warp", {"id": 2}, lambda: torch.arange(1000))
    assert old.exists() and record["status"] == "complete_not_cached"


def test_live_identity_writer_cannot_be_entered_twice(tmp_path):
    store = StageStore(tmp_path)
    key, _, _ = store.paths("geometry", {})
    with conversion_lock(tmp_path / f"geometry-{key}.lock"):
        with pytest.raises(RuntimeError):
            store.run("geometry", {}, lambda: torch.arange(1))
