from dataclasses import replace
from fractions import Fraction
import struct

import pytest

from dlss_fi_contract import (
    FRAME_OUT_MAGIC, SETUP_MAGIC, SETUP_OUT_MAGIC, FrameLedger, TwoXTimeline,
    build_timeline, setup_packet, validate_frame_response, validate_setup_response,
)


@pytest.mark.parametrize("rate", [Fraction(24), Fraction(30), Fraction(24000, 1001), Fraction(30000, 1001)])
@pytest.mark.parametrize("origin", [Fraction(0), Fraction(7, 3), Fraction(-1, 2)])
def test_two_x_keeps_exact_rate_origin_and_duration(rate, origin):
    plan = build_timeline([origin + i/rate for i in range(73)], rate)
    slots = list(plan.slots())
    assert len(slots) == 146
    assert slots[0].pts == origin
    assert slots[-1].pts + 1/plan.target_rate == origin + plan.duration
    assert plan.duration == 73/rate
    assert [s.pts for s in slots[::2]] == [origin + i/rate for i in range(73)]
    assert slots[-1].kind == "tail_hold"
    assert sum(s.kind == "generated" for s in slots) == 72


def test_cut_is_previous_source_hold_not_cross_cut_synthesis():
    plan = build_timeline([Fraction(i, 24) for i in range(4)], Fraction(24), cuts=[2])
    slots = list(plan.slots())
    assert [s.kind for s in slots] == ["source", "generated", "source", "cut_hold",
                                      "source", "generated", "source", "tail_hold"]
    assert slots[3].source_index == 1 and slots[3].pts == Fraction(3, 48)
    assert slots[4].source_index == 2 and slots[4].pts == Fraction(2, 24)


@pytest.mark.parametrize("fault", ["float_rate", "zero_rate", "negative_rate", "one_frame", "duplicate", "gap",
                                  "float_pts", "cut_zero", "cut_end", "cut_duplicate", "cut_bool"])
def test_bad_source_timeline_rejected(fault):
    rate, stamps, cuts = Fraction(24), [Fraction(i, 24) for i in range(4)], []
    if fault == "float_rate":
        rate = 24.0
    elif fault == "zero_rate":
        rate = 0
    elif fault == "negative_rate":
        rate = -24
    elif fault == "one_frame":
        stamps = stamps[:1]
    elif fault == "duplicate":
        stamps[2] = stamps[1]
    elif fault == "gap":
        stamps[2] += Fraction(1, 1000)
    elif fault == "float_pts":
        stamps[0] = 0.0
    else:
        cuts = {"cut_zero": [0], "cut_end": [4], "cut_duplicate": [2, 2], "cut_bool": [True]}[fault]
    with pytest.raises(ValueError):
        build_timeline(stamps, rate, cuts=cuts)


def test_direct_dataclass_cannot_bypass_timeline_scalar_validation():
    with pytest.raises(ValueError):
        TwoXTimeline(1, Fraction(24), Fraction(0), frozenset())
    with pytest.raises(ValueError):
        TwoXTimeline(4, 24.0, Fraction(0), frozenset())


def test_fixed_wire_packet_and_reset_payload_accounting():
    assert setup_packet(1024, 512, 73) == struct.pack("<5I", SETUP_MAGIC, 1024, 512, 73, 1)
    assert validate_setup_response(struct.pack("<4I", SETUP_OUT_MAGIC, 0, 1, 0))["requested_multiplier"] == 2
    for count in (0, 1):
        result = validate_frame_response(struct.pack("<4I", FRAME_OUT_MAGIC, 0, count, 0), reset=True)
        assert result == {"payload_frames": count, "usable_generated_frames": 0}
    assert validate_frame_response(struct.pack("<4I", FRAME_OUT_MAGIC, 0, 1, 0), reset=False)["usable_generated_frames"] == 1


@pytest.mark.parametrize("size", [0, 1, 15, 17, 32])
def test_malformed_header_does_not_drive_payload_allocation(size):
    with pytest.raises(ValueError):
        validate_setup_response(b"\0"*size)


@pytest.mark.parametrize("magic,status,count,disabled", [(0, 0, 1, 0), (FRAME_OUT_MAGIC, 1, 1, 0),
    (FRAME_OUT_MAGIC, 0, 1, 1), (FRAME_OUT_MAGIC, 0, 0, 0), (FRAME_OUT_MAGIC, 0, 2**32-1, 0)])
def test_worker_disabled_missing_frame_error_and_unbounded_count_fail(magic, status, count, disabled):
    with pytest.raises(ValueError):
        validate_frame_response(struct.pack("<4I", magic, status, count, disabled), reset=False)


@pytest.mark.parametrize("values", [(True, 512, 73), (0, 512, 73), (1024, 99999, 73), (1024, 512, 1)])
def test_setup_rejects_bad_geometry_and_single_frame(values):
    with pytest.raises(ValueError):
        setup_packet(*values)


def append_valid(ledger, slot, *, static=False):
    if slot.kind == "generated":
        ledger.observe(slot, payload_sha256="a"*64 if static else "c"*64,
                       worker_generated=True, endpoint_hashes=("a"*64, ("a" if static else "b")*64))
    else:
        ledger.observe(slot, payload_sha256="a"*64, endpoint_hashes=("a"*64,))


def test_streaming_ledger_accounts_all_types_without_claiming_runtime_or_audio():
    plan = build_timeline([Fraction(i, 30) for i in range(4)], Fraction(30), cuts=[2])
    ledger = FrameLedger(plan)
    for slot in plan.slots():
        append_valid(ledger, slot)
    result = ledger.finish()
    assert result["counts"] == {"source": 4, "generated": 2, "cut_hold": 1, "tail_hold": 1}
    assert result["target_rate"] == "60" and result["duration"] == "2/15"
    assert not result["runtime_qualified"] and not result["audio_qualified"] and not result["quality_qualified"]
    with pytest.raises(ValueError):
        ledger.finish()


def test_all_cut_input_explicitly_has_no_generated_intervals():
    plan = build_timeline([Fraction(i, 24) for i in range(3)], Fraction(24), cuts=[1, 2])
    ledger = FrameLedger(plan)
    for slot in plan.slots():
        append_valid(ledger, slot)
    assert ledger.finish()["has_generated_intervals"] is False


def test_static_duplicate_endpoints_can_legitimately_remain_static():
    plan = build_timeline([Fraction(0), Fraction(1, 24)], Fraction(24))
    ledger = FrameLedger(plan)
    for slot in plan.slots():
        append_valid(ledger, slot, static=True)
    assert ledger.finish()["counts"]["generated"] == 1


@pytest.mark.parametrize("fault", ["copied_motion", "wrong_provenance", "wrong_pts", "bad_hash", "missing_endpoint"])
def test_generated_frame_fault_is_sticky(fault):
    plan = build_timeline([Fraction(0), Fraction(1, 24)], Fraction(24))
    slots = list(plan.slots())
    ledger = FrameLedger(plan)
    append_valid(ledger, slots[0])
    slot, payload, generated, endpoints = slots[1], "c"*64, True, ("a"*64, "b"*64)
    if fault == "copied_motion":
        payload = "a"*64
    elif fault == "wrong_provenance":
        generated = False
    elif fault == "wrong_pts":
        slot = replace(slot, pts=Fraction(0))
    elif fault == "bad_hash":
        payload = "invalid"
    else:
        endpoints = ("a"*64,)
    with pytest.raises(ValueError):
        ledger.observe(slot, payload_sha256=payload, worker_generated=generated, endpoint_hashes=endpoints)
    with pytest.raises(ValueError):
        append_valid(ledger, slots[1])
    with pytest.raises(ValueError):
        ledger.finish()


def test_incomplete_and_extra_outputs_cannot_finish():
    plan = build_timeline([Fraction(0), Fraction(1, 24)], Fraction(24))
    ledger = FrameLedger(plan)
    with pytest.raises(ValueError):
        ledger.finish()
    for slot in plan.slots():
        append_valid(ledger, slot)
    with pytest.raises(ValueError):
        append_valid(ledger, list(plan.slots())[-1])
    with pytest.raises(ValueError):
        ledger.finish()
