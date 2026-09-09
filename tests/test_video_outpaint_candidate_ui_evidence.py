from copy import deepcopy
import hashlib

from PIL import Image
import pytest

from tools.capture_outpaint_candidate_ui_evidence import capture_history


def _data(tmp_path):
    picture = Image.new("RGB", (12, 8), (0, 0, 200))
    picture.save(tmp_path / "test.png")
    fixture = {"source_file": "fixture.mp4", "preview_report": {
        "rgb8_sha256": hashlib.sha256(picture.tobytes()).hexdigest()}}
    graph = {"1": {"class_type": "LoadVideo", "inputs": {"file": "fixture.mp4"}},
        "2": {"class_type": "MiniMaxH3VideoOutpaintLoadSelectionT8", "inputs": {"selection_id": "a"*64}}}
    history = {"job": {"prompt": [0, "job", graph], "status": {"status_str": "success", "completed": True,
        "messages": [["execution_start", {"timestamp": 1}]]}, "outputs": {"2": {"images": [
            {"type": "temp", "filename": "test.png", "subfolder": ""}]}}}}
    return fixture, history


def test_actual_output_rgb_and_explicit_restore_id_are_recorded_without_mutation(tmp_path):
    fixture, history = _data(tmp_path)
    before = deepcopy(history)
    result = capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id="a"*64)
    assert history == before
    assert result["job"]["restored_selection"]
    assert result["job"]["images"][0]["rgb8_sha256"] == fixture["preview_report"]["rgb8_sha256"]


@pytest.mark.parametrize("damage", ["short_id", "other_id", "wrong_rgb", "other_source", "model", "missing_image", "unexpected_error", "path_escape"])
def test_bad_or_unrelated_execution_cannot_be_claimed_as_archive_ui_pass(tmp_path, damage):
    fixture, history = _data(tmp_path)
    entry = history["job"]
    copied_id = "a"*64
    if damage == "short_id":
        copied_id = "a"*63
    elif damage == "other_id":
        entry["prompt"][2]["2"]["inputs"]["selection_id"] = "b"*64
    elif damage == "wrong_rgb":
        fixture["preview_report"]["rgb8_sha256"] = "b"*64
    elif damage == "other_source":
        entry["prompt"][2]["1"]["inputs"]["file"] = "other.mp4"
    elif damage == "model":
        entry["prompt"][2]["3"] = {"class_type": "UNETLoader", "inputs": {}}
    elif damage == "missing_image":
        entry["outputs"] = {}
    elif damage == "unexpected_error":
        entry["status"]["status_str"] = "error"
    else:
        entry["outputs"]["2"]["images"][0]["subfolder"] = ".."
    with pytest.raises(ValueError):
        capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id=copied_id)


def test_unconfirmed_error_is_expected_but_only_at_the_selection_node(tmp_path):
    fixture, history = _data(tmp_path)
    entry = history["job"]
    entry["prompt"][2]["2"] = {"class_type": "MiniMaxH3VideoOutpaintSelectCandidateT8", "inputs": {"confirm_selection": False}}
    error = {"node_type": "MiniMaxH3VideoOutpaintSelectCandidateT8", "exception_message": "confirm first"}
    entry["status"].update(status_str="error", completed=False, messages=[["execution_error", error]])
    assert capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id="a"*64)["job"]["confirm_selection"] == [False]
    error["node_type"] = "MiniMaxH3VideoOutpaintLoadPreparedT8"
    with pytest.raises(ValueError, match="stop at Select"):
        capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id="a"*64)


def test_confirmed_preview_text_must_match_the_id_actually_copied(tmp_path):
    fixture, history = _data(tmp_path)
    entry = history["job"]
    entry["prompt"][2]["2"] = {"class_type": "MiniMaxH3VideoOutpaintSelectCandidateT8", "inputs": {"confirm_selection": True}}
    entry["outputs"]["3"] = {"text": ["b"*64]}
    with pytest.raises(ValueError, match="PreviewAny"):
        capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id="a"*64)
    entry["outputs"]["3"]["text"] = ["a"*64]
    assert capture_history(history, fixture=fixture, temp_root=tmp_path, copied_id="a"*64)["job"]["text"] == ["a"*64]
