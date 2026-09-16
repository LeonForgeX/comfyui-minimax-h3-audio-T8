"""Development package gates cannot borrow accepted legacy human qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import uuid
import zipfile

import pytest

from tools import package_fast_h3_v2_candidate as package


@pytest.fixture(scope="module")
def selected():
    return package.select_files()


def test_real_baseline_registry_and_all255_workflows_are_frozen(selected):
    files, includes, workflows, memory = selected
    features = json.loads((package.PROJECT / "features.json").read_text(encoding="utf8"))
    package.validate_registry(features["nodes"], features["nodes"])
    assert len(features["nodes"]) == 339
    assert features["nodes"][336:] == package.SUFFIX
    assert len(workflows) == 255
    assert set(package.legacy.SELF_LIFT_WORKFLOWS).issubset(workflows)
    assert package.legacy.T8_MEMORY_WORKFLOW in workflows
    assert memory["path"] == package.legacy.T8_MEMORY_WORKFLOW
    assert package.REQUIRED.issubset(files)
    assert "docs/FAST_H3_V2_EXP.md" in includes
    assert "docs/FRAME_LIMITS.md" in includes
    assert "docs/FRAME_LIMITS.md" in files
    frame_status = json.loads((package.PROJECT / "meta.json").read_text(encoding="utf8"))
    assert frame_status["frame_upper_limits_hotfix_20260917"]["status"] == "implemented_local_verified_not_published"
    assert all(package.allowed_member(name) for name in files)
    assert set(package.ACCEPTED_V2_WORKFLOWS).issubset(files)
    assert {name for name in files if name.startswith('examples/workflows/') and name.endswith('.json')} == set(workflows) | set(package.ACCEPTED_V2_WORKFLOWS)


@pytest.mark.parametrize("mutation", ["old_prefix", "suffix_order", "missing", "duplicate", "feature_mismatch"])
def test_registry_rejects_any_old_or_new_order_drift(mutation):
    ids = json.loads((package.PROJECT / "features.json").read_text(encoding="utf8"))["nodes"]
    features = list(ids)
    if mutation == "old_prefix":
        ids[24], ids[25] = ids[25], ids[24]
        features = list(ids)
    elif mutation == "suffix_order":
        ids[-1], ids[-2] = ids[-2], ids[-1]
        features = list(ids)
    elif mutation == "missing":
        ids, features = ids[:-1], ids[:-1]
    elif mutation == "duplicate":
        ids[-1] = ids[-2]
        features = list(ids)
    else:
        features[-1] = "different"
    with pytest.raises(ValueError, match="schema|registry"):
        package.validate_registry(ids, features)


@pytest.mark.parametrize("name", ["roadmap.md", "RoAdMaP.Md", "SKILL.md", "h3_t8/SKILL.md",
                                  "tests/test.py", "tools/probe.py", "artifacts/review.json", ".git/index",
                                  "models/model.json", "h3_t8/weights.safetensors", "h3_t8/model.gguf",
                                  "h3_t8/runtime.dll", "h3_t8/sample.mp4", "h3_t8/reference.png",
                                  "h3_t8/__pycache__/model.pyc"])
def test_runtime_model_media_and_local_handoff_are_excluded(name):
    assert package.allowed_member(name) is False


@pytest.mark.parametrize("name", ["../escape.py", "h3_t8/../../escape.py", "/absolute.py",
                                  "C:/drive.py", "h3_t8\\nodes.py", "h3_t8//nodes.py", "./nodes.py"])
def test_unsafe_archive_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        package.allowed_member(name)


def test_candidate_status_is_unpublished_pending_and_not_the_legacy_qualification():
    status = package.candidate_status()
    package.validate_candidate_receipt(status)
    assert status["human_qualified"] is False
    assert status["v2_human_review"] == "pending"
    assert status["published"] is False
    assert status["publication_status"] == "not_published"
    assert status["v2_candidate_workflows_packaged"] == 0
    assert status["media_or_gpu_qualification_inferred_from_cpu"] is False


def test_candidate_receipt_binds_actual_source_version_without_claiming_publication():
    import tomllib
    version = tomllib.loads((package.PROJECT / 'pyproject.toml').read_text(encoding='utf8'))['project']['version']
    status = package.candidate_status(bound_controls=True)
    assert version in {'1.82.0', '1.83.0'}
    assert status['source_release_version'] == version
    assert status['published'] is False
    package.validate_candidate_receipt(status)


def test_unaudited_source_version_cannot_build_candidate(tmp_path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nversion="99.0.0"\n', encoding='utf8')
    with pytest.raises(ValueError, match='audited'):
        package.select_files(tmp_path)


@pytest.mark.parametrize("mutation", ["accepted", "published", "version", "prefix", "node_count", "numeric_false"])
def test_old_human_qualification_or_mutated_receipt_cannot_pass_v2_gate(mutation):
    receipt = deepcopy(package.candidate_status())
    if mutation == "accepted":
        receipt["human_qualified"] = True
        receipt["v2_human_review"] = "accepted"
    elif mutation == "published":
        receipt["published"] = True
    elif mutation == "version":
        receipt["source_release_version"] = "0.0.0"
    elif mutation == "prefix":
        receipt["legacy_registry_prefix"] = 331
    elif mutation == "node_count":
        receipt["registered_nodes"] = 336
    else:
        receipt["human_qualified"] = 0
    with pytest.raises(ValueError, match="human-pending"):
        package.validate_candidate_receipt(receipt)


def test_legacy_workflow_byte_change_is_rejected_even_if_new_hash_matches(tmp_path, selected):
    files, _, workflows, _ = selected
    files = {k: v for k, v in files.items() if k not in package.ACCEPTED_V2_WORKFLOWS}
    for name, payload in workflows.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    changed = sorted(workflows)[0]
    path = tmp_path / changed
    path.write_bytes(path.read_bytes() + b"\n")
    changed_files = {**files, changed: package.sha(path)}
    with pytest.raises(ValueError, match="legacy workflow bytes"):
        package.validate_legacy_artifacts(tmp_path, changed_files, workflows)


def test_extra_unreviewed_workflow_cannot_enter_recommended_examples(selected):
    files, _, workflows, _ = selected
    files = {**files, "examples/workflows/FastH3_V2_Unreviewed.json": "0" * 64}
    with pytest.raises(ValueError, match="workflow membership|fixed255"):
        package.validate_legacy_artifacts(package.PROJECT, files, workflows)


def test_six_accepted_controls_are_pinned_without_a_universal_quality_claim(selected):
    files, _, workflows, _ = selected
    package.validate_legacy_artifacts(package.PROJECT, files, workflows)
    status = package.candidate_status(bound_controls=True)
    package.validate_candidate_receipt(status)
    assert status['human_qualified'] is True
    assert status['v2_human_review'] == 'accepted_only_in_six_bound_control_scopes'
    assert status['v2_candidate_workflows_packaged'] == 6
    assert status['published'] is False and status['universal_quality_claim'] is False
    altered = dict(files)
    altered.pop(next(iter(package.ACCEPTED_V2_WORKFLOWS)))
    with pytest.raises(ValueError, match='complete'):
        package.validate_legacy_artifacts(package.PROJECT, altered, workflows)
    altered = dict(files)
    altered[next(iter(package.ACCEPTED_V2_WORKFLOWS))] = '0' * 64
    with pytest.raises(ValueError, match='bytes changed'):
        package.validate_legacy_artifacts(package.PROJECT, altered, workflows)


def test_archive_tampering_duplicate_and_extra_members_are_rejected(tmp_path):
    archive = tmp_path / "candidate.zip"
    data = b"candidate fixture"
    digest = __import__("hashlib").sha256(data).hexdigest()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("__init__.py", data)
    package.check_archive(archive, {"__init__.py": digest})
    with pytest.raises(ValueError, match="changed"):
        package.check_archive(archive, {"__init__.py": "0" * 64})
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("roadmap.md", b"must not ship")
    with pytest.raises(ValueError, match="membership"):
        package.check_archive(archive, {"__init__.py": digest})


def test_git_environment_cannot_redirect_snapshot_staging_to_main_index(monkeypatch):
    monkeypatch.setenv("GIT_INDEX_FILE", "main-index-must-not-be-used")
    monkeypatch.setenv("GIT_DIR", "main-git-must-not-be-used")
    monkeypatch.setenv("GIT_WORK_TREE", "main-tree-must-not-be-used")
    env = package.environment(cpu=True)
    assert all(not key.startswith("GIT_") for key in env)
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_build_requires_new_owned_artifact_directory(tmp_path):
    with pytest.raises(ValueError, match="artifact output"):
        package.build(tmp_path)
    with pytest.raises(ValueError, match="artifact output"):
        package.build(package.PROJECT / "artifacts")


@pytest.mark.skipif(os.environ.get("T8_FASTH3V2_PACKAGE_INTEGRATION") != "1",
                    reason="Opt-in actual official comfy-cli zip + extracted Core CPU import; no GPU")
def test_actual_official_zip_and_new_interpreter_cpu_import_keep_v2_pending(monkeypatch):
    import folder_paths

    cli_python = Path(os.environ["T8_COMFY_CLI_PYTHON"]).resolve(strict=True)
    core = Path(folder_paths.__file__).resolve().parent
    root = package.PROJECT / "artifacts" / ("fasth3-v2-package-cpu-test-" + uuid.uuid4().hex)
    index = Path(package.git(package.PROJECT, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    index_sha = package.sha(index)
    monkeypatch.setenv("GIT_INDEX_FILE", str(index))
    receipt = package.build(root, zip_python=cli_python)
    result = package.verify(root, core)
    assert receipt["human_qualified"] is result["human_qualified"] is True
    assert result["nodes"] == 339 and result["gpu_initialized"] is False
    assert result["publication_status"] == "not_published"
    assert result["v2_human_review"] == "accepted_only_in_six_bound_control_scopes"
    assert result["package_origins"]
    assert package.sha(index) == index_sha
    implementation = json.loads((root / "official-zip.stdout.log").read_text(encoding="utf8").splitlines()[-1])
    assert implementation["official_function"] == "comfy_cli.file_utils.zip_files"
    assert (root / "actual-CPU-import.stdout.log").is_file()
    assert (root / "actual-CPU-import.stderr.log").is_file()
    assert all(Path(path).is_relative_to(root / package.EXTRACTED) for path in result["package_origins"].values())
    print(json.dumps({"candidate_root": str(root), "archive_sha256": receipt["archive_sha256"],
                      "nodes": result["nodes"], "V2_human_review": result["v2_human_review"],
                      "published": result["published"], "main_index_unchanged": True}))
