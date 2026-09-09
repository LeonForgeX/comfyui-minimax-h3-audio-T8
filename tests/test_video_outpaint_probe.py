import json
import hashlib
import threading
from types import SimpleNamespace

import pytest

from tools import run_video_outpaint_t8_probe as probe


@pytest.mark.parametrize("status,delta,allowed", [("failed", 10, True), ("failed", -10, False), ("running", 10, False)])
def test_resume_distinguishes_reused_pid_from_active_or_unknown_server(tmp_path, monkeypatch, status, delta, allowed):
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")
    monkeypatch.setattr(probe.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: report_path.stat().st_mtime + delta))
    report = {"server_pid": 1234, "status": status}
    if allowed:
        probe.verify_prior_server_inactive(report, report_path)
    else:
        with pytest.raises(RuntimeError, match="without proof"):
            probe.verify_prior_server_inactive(report, report_path)


def test_live_journal_keeps_each_observation_before_sampler_stop(tmp_path, monkeypatch):
    sampler = probe.LiveMemorySampler.__new__(probe.LiveMemorySampler)
    sampler.journal = tmp_path / "telemetry.jsonl"
    sampler.journal_error = None
    sampler._stop = threading.Event()
    sampler.rows = []
    def observe(self):
        self.rows.append({"gpu_free_mib": 1234, "elapsed_seconds": 0.1})
        assert json.loads(self.journal.read_text()) == self.rows[0]
    monkeypatch.setattr(probe.upstream.MemorySampler, "_run", observe)
    sampler._run()
    assert sampler.journal_error is None and len(sampler.rows) == 1
    before = sampler.journal.read_bytes()
    sampler._run()
    assert sampler.journal_error and sampler._stop.is_set()
    assert sampler.journal.read_bytes() == before


def test_resume_graph_changes_only_explicit_resume_flags():
    if not probe.upstream.REFERENCE.exists():
        pytest.skip("optional pinned upstream fixture is not installed")
    before, after = probe.build_prompt(), probe.build_prompt(resume=True)
    assert not before["11"]["inputs"]["resume_audio"] and after["11"]["inputs"]["resume_audio"]
    assert not before["12"]["inputs"]["resume"] and after["12"]["inputs"]["resume"]
    after["11"]["inputs"]["resume_audio"] = False
    after["12"]["inputs"]["resume"] = False
    assert before == after


def test_native_noise_probe_changes_only_the_requested_noise_input():
    before, after = probe.build_prompt(), probe.build_prompt(native_noise=True)
    assert after["12"]["inputs"].pop("noise_algorithm") == "t8.outpaint.native_cpu_noise/v1"
    assert before == after
    assert not any(n["class_type"] == "SaveVideo" for n in after.values())


def test_geometry_probe_changes_only_actual_compose_opt_in():
    before,after=probe.build_prompt(),probe.build_prompt(geometry_align=True)
    assert after['13']['inputs'].pop('geometry_align') is True
    assert after==before
    with pytest.raises(ValueError,match='geometry_align'):
        probe.build_prompt(geometry_align='true')


def test_probe_explicitly_binds_joint_default_and_preserve_alternative():
    joint = probe.build_prompt()
    preserve = probe.build_prompt(source_mode="preserve_source")
    assert joint["13"]["inputs"]["source_mode"] == "joint_decode"
    assert preserve["13"]["inputs"].pop("source_mode") == "preserve_source"
    joint["13"]["inputs"].pop("source_mode")
    assert joint == preserve
    with pytest.raises(ValueError, match="source_mode"):
        probe.build_prompt(source_mode="joint")


def test_probe_can_increase_only_the_audited_kj_memory_chunk_counts():
    before = probe.build_prompt()
    after = probe.build_prompt(head_chunks=8, ffn_chunks=8)
    assert after["5"]["inputs"]["head_chunks"] == 8
    assert after["9"]["inputs"]["chunks"] == 8
    after["5"]["inputs"]["head_chunks"] = before["5"]["inputs"]["head_chunks"]
    after["9"]["inputs"]["chunks"] = before["9"]["inputs"]["chunks"]
    assert after == before
    with pytest.raises(ValueError, match="head_chunks"):
        probe.build_prompt(head_chunks=57)
    with pytest.raises(ValueError, match="ffn_chunks"):
        probe.build_prompt(ffn_chunks=0)


def test_compose_retry_allows_only_delivery_changes_and_complete_assets(tmp_path, monkeypatch):
    from h3_audio_t8_pkg.video_outpaint_plan import canonical
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    prior = tmp_path / "artifacts/prior"
    cache = prior / "output/T8_H3_Outpaint_Cache/run/sampling"
    cache.mkdir(parents=True)
    stable, changed = tmp_path / "video_outpaint_sampling.py", tmp_path / "video_outpaint_compose.py"
    stable.write_bytes(b"stable sampler")
    changed.write_bytes(b"corrected composition")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()
    record = {"source_sha256": probe.SOURCE_SHA, "implementation_sha256": {
        stable.name: sha(stable), changed.name: "0"*64}}
    (prior / "report.json").write_text(json.dumps(record))
    asset = b"fake sample publication fixture"
    digest = hashlib.sha256(asset).hexdigest()
    (cache / f"window-{digest}.safetensors").write_bytes(asset)
    state = {"status": "sampled", "committed": [{"sha256": digest, "bytes": len(asset)}],
             "identity": {"plan_sha256": "a"*64}}
    def write_state():
        (cache / "outpaint_windows.json").write_text(canonical({**state, "sha256": hashlib.sha256(canonical(state).encode()).hexdigest()}))
    write_state()
    assert probe.checked_resume_cache(prior, sha, compose_only=True, expected_plan_sha="a"*64)
    with pytest.raises(ValueError, match="runtime changed"):
        probe.checked_resume_cache(prior, sha)
    stable.write_bytes(b"changed sampler")
    with pytest.raises(ValueError, match="runtime changed"):
        probe.checked_resume_cache(prior, sha, compose_only=True)
    stable.write_bytes(b"stable sampler")
    state["status"] = "interrupted"
    write_state()
    with pytest.raises(ValueError, match="fully sampled"):
        probe.checked_resume_cache(prior, sha, compose_only=True)
    state["status"] = "sampled"
    write_state()
    (cache / f"window-{digest}.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="asset integrity"):
        probe.checked_resume_cache(prior, sha, compose_only=True)
