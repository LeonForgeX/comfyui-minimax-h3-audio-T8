"""Read-only Core provenance and isolated probe resource configuration."""

import hashlib
import importlib
from pathlib import Path
import subprocess


CORE_MODULES = {
    "comfy.ldm.minimax.model": "comfy/ldm/minimax/model.py",
    "comfy.model_patcher": "comfy/model_patcher.py",
    "comfy.samplers": "comfy/samplers.py",
    "comfy.model_management": "comfy/model_management.py",
    "folder_paths": "folder_paths.py",
}


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def verify_core_source(core_root, *, repository=None, revision=None):
    core = Path(core_root).resolve()
    if not (core / "comfy/ldm/minimax/model.py").is_file():
        raise ValueError("selected source tree has no native H3 Core")
    try:
        top = Path(_git(core, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    except subprocess.CalledProcessError:
        top = None
    is_checkout = top == core
    if not is_checkout and (repository is None or revision is None):
        raise ValueError("Core archive needs explicit repository and revision; enclosing Git repository is not Core")
    repo = core if is_checkout else Path(repository).resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise ValueError("Core repository must name its exact Git root")
    commit = _git(repo, "rev-parse", "--verify", f"{revision or 'HEAD'}^{{commit}}").decode().strip()
    entries = _git(repo, "ls-tree", "-rz", commit).split(b"\0")
    tracked = {}
    for entry in entries:
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        name = raw_path.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"unsupported Core source entry: {name}")
        tracked[name] = blob
    if not all(name in tracked for name in CORE_MODULES.values()):
        raise ValueError("revision does not contain the required H3 Core files")
    if is_checkout:
        head = _git(repo, "rev-parse", "HEAD").decode().strip()
        if commit != head or _git(repo, "status", "--porcelain", "--untracked-files=no").strip():
            raise ValueError("Core checkout must be clean and match the requested revision")
    else:
        for name, blob in tracked.items():
            path = (core / name).resolve()
            if not path.is_relative_to(core) or not path.is_file():
                raise ValueError(f"Core archive file missing or escaped: {name}")
            contents = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
            if actual != blob:
                raise ValueError(f"Core archive differs from declared revision: {name}")
    files = {name: hashlib.sha256((core / name).read_bytes()).hexdigest() for name in CORE_MODULES.values()}
    return {"core_root": str(core), "core_commit": commit, "repository": str(repo),
            "kind": "clean_checkout" if is_checkout else "verified_archive",
            "archive_verified_file_count": len(tracked) if not is_checkout else None,
            "core_files_sha256": files,
            "dependency_scope": "current host packages, not historical dependency pins"}


def probe_resource_config(runtime_root, project_root):
    runtime, project = Path(runtime_root).resolve(), Path(project_root).resolve()
    models = runtime / "models"
    return {
        "t8_runtime_models": {key: str(models / key) for key in
                              ("diffusion_models", "text_encoders", "vae", "loras", "latent_upscale_models")},
        "t8_project_nodes": {"custom_nodes": str(project.parent)},
        "t8_probe_nodes": {"custom_nodes": str(project / "tools")},
    }


def verify_live_core(expected):
    root = Path(expected["core_root"]).resolve()
    observed = {}
    for module, relative in CORE_MODULES.items():
        path = Path(importlib.import_module(module).__file__).resolve()
        if path != root / relative:
            raise RuntimeError(f"wrong live Core import for {module}: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["core_files_sha256"][relative]:
            raise RuntimeError(f"live Core source changed: {relative}")
        observed[module] = {"path": str(path), "sha256": digest}
    return {"status": "pass", "core_commit": expected["core_commit"], "modules": observed,
            "dependency_scope": expected["dependency_scope"]}
