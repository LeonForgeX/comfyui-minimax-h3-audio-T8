import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.tools import vdn_probe_environment as env
from h3_audio_t8_pkg.tools import run_nfe_resume_real_probe as shared


@pytest.fixture
def source_tree(tmp_path):
    repo = tmp_path / "core"
    repo.mkdir()
    names = [*env.CORE_MODULES.values(), "README.md"]
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# tracked fixture\n", encoding="utf-8")
    env._git(repo, "init")
    env._git(repo, "-c", "core.autocrlf=false", "add", ".")
    env._git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "-m", "fixture")
    revision = env._git(repo, "rev-parse", "HEAD").decode().strip()
    archive = repo / "nested" / "archive"
    for name in names:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / name).read_bytes())
    return repo, revision, archive


def test_nested_archive_cannot_inherit_parent_git_commit(source_tree):
    repo, revision, archive = source_tree
    with pytest.raises(ValueError, match="explicit repository and revision"):
        env.verify_core_source(archive)
    report = env.verify_core_source(archive, repository=repo, revision=revision)
    assert report["kind"] == "verified_archive"
    assert report["archive_verified_file_count"] == len(env.CORE_MODULES) + 1
    assert report["core_commit"] == revision and report["core_root"] == str(archive.resolve())


@pytest.mark.parametrize("corruption", ["missing", "changed"])
def test_archive_every_tracked_file_is_checked_not_just_module_names(source_tree, corruption):
    repo, revision, archive = source_tree
    path = archive / "README.md"
    if corruption == "missing":
        path.unlink()
    else:
        path.write_text("different bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="README.md"):
        env.verify_core_source(archive, repository=repo, revision=revision)


def test_clean_checkout_records_head_but_dirty_checkout_is_not_exact(source_tree):
    repo, revision, _ = source_tree
    assert env.verify_core_source(repo)["core_commit"] == revision
    (repo / "README.md").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        env.verify_core_source(repo)


def test_runtime_resource_paths_do_not_follow_core_archive(tmp_path):
    runtime = tmp_path / "installed"
    project = runtime / "custom_nodes" / "minimax-h3-audio-T8"
    result = env.probe_resource_config(runtime, project)
    assert result["t8_runtime_models"]["diffusion_models"] == str(runtime / "models" / "diffusion_models")
    assert result["t8_project_nodes"]["custom_nodes"] == str(project.parent)
    assert result["t8_probe_nodes"]["custom_nodes"] == str(project / "tools")
    assert not runtime.exists(), "configuration must not copy/install/create model paths"


def test_live_imports_verify_path_and_preflight_bytes(source_tree, monkeypatch):
    repo, revision, archive = source_tree
    expected = env.verify_core_source(archive, repository=repo, revision=revision)
    monkeypatch.setattr(env.importlib, "import_module",
                        lambda name: SimpleNamespace(__file__=str(archive / env.CORE_MODULES[name])))
    assert env.verify_live_core(expected)["status"] == "pass"
    changed = copy.deepcopy(expected)
    changed["core_files_sha256"]["folder_paths.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        env.verify_live_core(changed)
    monkeypatch.setattr(env.importlib, "import_module",
                        lambda name: SimpleNamespace(__file__=str(repo / env.CORE_MODULES[name])))
    with pytest.raises(RuntimeError, match="wrong live Core import"):
        env.verify_live_core(expected)


def test_isolated_server_uses_explicit_input_without_changing_default(tmp_path):
    args = SimpleNamespace(python=Path("python.exe"), host="127.0.0.1", port=8205, comfy_root=tmp_path / "archive")
    original = shared._server_command(args, tmp_path / "run")
    assert original[original.index("--input-directory") + 1] == str((args.comfy_root / "input").resolve())
    args.input_directory = tmp_path / "installed" / "input"
    explicit = shared._server_command(args, tmp_path / "run")
    assert explicit[explicit.index("--input-directory") + 1] == str(args.input_directory.resolve())


def test_failed_isolated_start_cleans_its_own_process(monkeypatch):
    server = shared.IsolatedServer(SimpleNamespace(), Path("not-created"), "fixture")
    stopped = []
    def fail():
        raise RuntimeError("controlled start failure")
    monkeypatch.setattr(server, "start", fail)
    monkeypatch.setattr(server, "stop", lambda: stopped.append(True))
    with pytest.raises(RuntimeError, match="controlled start failure"):
        server.__enter__()
    assert stopped == [True]
