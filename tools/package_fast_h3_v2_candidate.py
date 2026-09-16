"""Official local development archive + isolated CPU import, never publication.

Only six independently accepted V2 control recipes may carry scoped human
qualification. Legacy acceptance is never reused as V2 evidence. Build only
in a new project-owned artifacts directory; this tool never publishes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import package_progressive_candidate as legacy  # noqa: E402
from tools.verify_progressive_package import (  # noqa: E402
    validate_registry as validate_legacy_registry,
    validate_workflows as validate_legacy_workflows,
)

BASELINE = "8036c66034af4eea6160ed98d5299a2bae79ecd8"
PREFIX_SHA256 = "1ce8dbd0c6666b5696d1c0f672f02ea70318ee24e9ffab454bbc38f920ffaf8e"
SUFFIX = ["MiniMaxH3FastH3V2SetupEXPT8", "MiniMaxH3FastH3V2RuntimeAuditEXPT8",
          "MiniMaxH3FastH3V2DualModelLongVideoEXPT8"]
SCHEMA = "t8.fasth3_v2.development_package/v1"
STATUS_MEMBER = "FAST_H3_V2_CANDIDATE_STATUS.json"
ARCHIVE = "minimax-h3-audio-T8-FastH3-V2-dev339-human-pending.zip"
SCOPED_ARCHIVE = "minimax-h3-audio-T8-FastH3-V2-dev339-bound-controls-accepted.zip"
EXTRACTED = "unpacked/minimax-h3-audio-T8"
BANNED_SUFFIXES = {".safetensors", ".onnx", ".ckpt", ".pt", ".pth", ".bin", ".dll", ".exe",
                   ".gguf", ".engine", ".plan", ".npz", ".npy", ".mp4", ".mkv", ".mov",
                   ".webm", ".wav", ".mp3", ".flac", ".png", ".jpg", ".jpeg", ".gif",
                   ".webp", ".pyc", ".pyo"}
REQUIRED = {"__init__.py", "pyproject.toml", "features.json", "meta.json", "h3_t8/nodes.py",
            "h3_t8/fast_h3_v2_advanced.py", "h3_t8/nodes_fast_h3_v2_advanced.py",
            "h3_t8/h3_memory_advanced.py", "h3_t8/long_video_dual_identity.py",
            "h3_t8/long_video_dual_model_runner.py", "docs/FAST_H3_V2_EXP.md", "docs/FRAME_LIMITS.md",
            "docs/H3_MEMORY_NODES_EXP.md", "docs/RELEASE_1.82.0.md", legacy.T8_MEMORY_WORKFLOW}
SECRETS = re.compile(rb"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
ACCEPTED_V2_WORKFLOWS = {
    # Canonical LF deployment. Original artifact CRLF receipts are preserved;
    # tests prove the only byte difference is CRLF -> LF, then re-audit controls.
    'examples/workflows/10-speed/FastH3_V2_Trained_VSA_73f_h1c1_EXP.json': 'af70d29bc8d47c78e4231bd64ff225d036927a2a730121397b014e81e571864c',
    'examples/workflows/10-speed/FastH3_V2_Trained_VSA_73f_h4c2_EXP.json': '3b9debad068b9f20fe96fc12a518dd9fc695bb8e118d431c9248ed18ead22ae8',
    'examples/workflows/10-speed/FastH3_V2_Official_Comfy_Template_124f_EXP.json': 'f70f7221fbc5ff5c09c4db154b1765e1fbd50298dfc309f8a5dcf89670c14c18',
    'examples/workflows/10-speed/FastH3_V2_Dense_Relay_Dual_4plus4_8s_EXP.json': '02fea4061af2c597d27e1c1494fc8f2ea40cd06593d68232ff5ccfe551c51926',
    'examples/workflows/10-speed/FastH3_V2_Trained_VSA_Dual_4plus4_Resume_8s_EXP.json': '4bea7f387d9bc6d436d8931f1c95f270faec12317105d15822a607f418c58835',
    'examples/workflows/10-speed/FastH3_V2_Dense_Sol_Audio_Protected_73f_EXP.json': '4cccd2d2ec64dfc94b40b80df67f9f33848cfe5ff3dd40f53e30ebba64b1f4a1',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(*, cpu=False):
    # A user's GIT_INDEX_FILE/GIT_DIR must never redirect snapshot staging to
    # the live index. All git writes below target only the new snapshot repo.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    if cpu:
        env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def git(project, *args):
    return subprocess.check_output(["git", *args], cwd=project, env=environment())


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe candidate member: " + name)
    return path


def allowed_member(name):
    path = safe_member(name)
    if (path.name.lower() in {"roadmap.md", "skill.md"}
            or path.parts[0].lower() in {".git", ".github", "tests", "tools", "artifacts", "models"}
            or any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in path.parts)
            or path.suffix.lower() in BANNED_SUFFIXES):
        return False
    return True


def validate_registry(ids, feature_ids):
    validate_legacy_registry(ids[:336], feature_ids[:336])
    digest = hashlib.sha256(json.dumps(ids[:336], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if (ids != feature_ids or len(ids) != 339 or len(set(ids)) != 339
            or digest != PREFIX_SHA256 or ids[336:] != SUFFIX):
        raise ValueError("V2 candidate registry differs from the complete fixed336 prefix plus exact3 suffix")


def baseline(project=PROJECT):
    payload = git(project, "archive", "--format=tar", BASELINE, "features.json", "examples/workflows")
    workflows, features = {}, None
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            safe_member(member.name)
            if member.name == "features.json":
                features = json.loads(archive.extractfile(member).read())
            elif member.name.startswith("examples/workflows/") and member.name.endswith(".json"):
                workflows[member.name] = archive.extractfile(member).read()
    if features is None or len(workflows) != 255:
        raise ValueError("Fixed v1.82.0 baseline does not contain336 nodes and255 workflows")
    validate_registry(features["nodes"] + SUFFIX, features["nodes"] + SUFFIX)
    return workflows


def validate_legacy_artifacts(project, files, workflows):
    names = {name for name in files if name.startswith("examples/workflows/") and name.endswith(".json")}
    accepted_names = names & set(ACCEPTED_V2_WORKFLOWS)
    if accepted_names and accepted_names != set(ACCEPTED_V2_WORKFLOWS):
        raise ValueError('Accepted V2 workflow set must be complete, not partially promoted')
    legacy_names = names - accepted_names
    validate_legacy_workflows(list(legacy_names), {k: v for k, v in files.items() if k not in accepted_names})
    if legacy_names != set(workflows):
        raise ValueError("Candidate changed the fixed255 workflow membership")
    for name in accepted_names:
        expected = ACCEPTED_V2_WORKFLOWS[name]
        path = Path(project) / name
        if files[name] != expected or sha(path) != expected:
            raise ValueError('Accepted V2 control workflow bytes changed: ' + name)
        review = json.loads(path.read_text(encoding='utf8')).get('extra', {}).get('t8_bound_review', {})
        if review.get('status') != 'accepted_in_this_review_scope' or review.get('not_universal_quality_claim') is not True:
            raise ValueError('Accepted V2 scoped review missing: ' + name)
    for name, payload in workflows.items():
        if (Path(project) / name).read_bytes() != payload or files[name] != hashlib.sha256(payload).hexdigest():
            raise ValueError("Candidate changed legacy workflow bytes: " + name)
        json.loads(payload)
    # Reuse the accepted nine SHA bindings, pending-marker and scoped-review
    # contracts. The original helper has a global PROJECT, so read this exact
    # snapshot instead of changing the helper's global or borrowing another tree.
    for name, digest in legacy.SELF_LIFT_WORKFLOWS.items():
        text = (Path(project) / name).read_text(encoding="utf8")
        review = json.loads(text).get("extra", {}).get("t8_bound_review", {})
        if (files.get(name) != digest or any(token.lower() in text.lower() for token in legacy.PENDING_REVIEW_TOKENS)
                or review.get("status") != "accepted_in_this_review_scope"
                or review.get("not_universal_quality_claim") is not True):
            raise ValueError("Accepted SelfLift workflow binding differs: " + name)
        if "_EAV_" in name:
            node = next(node for node in json.loads(text)["nodes"]
                        if node["type"] == "MiniMaxH3ProgressiveLongVideoEXPT8")
            values = node["widgets_values_named"]
            if tuple(values[key] for key in ("eav_tau", "eav_start_video_progress", "eav_end_video_progress",
                                             "eav_g_hard_limit")) != (8.0, .15, .90, 1.5):
                raise ValueError("Accepted EAV workflow recipe differs: " + name)
    return legacy.validate_t8_memory_promotion(files, project=project)


def select_files(project=PROJECT):
    from pathspec import PathSpec

    project = Path(project).resolve(strict=True)
    config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf8"))
    if config["project"]["version"] not in {"1.82.0", "1.83.0"}:
        raise ValueError("V2 candidate requires the audited1.82.0 development or1.83.0 source version")
    includes = config["tool"]["comfy"]["includes"]
    for name in includes:
        if not allowed_member(name) or not (project / name).is_file():
            raise ValueError("Unsafe or missing explicit candidate include: " + name)
    ignore = PathSpec.from_lines("gitwildmatch", (project / ".comfyignore").read_text().splitlines())
    tracked = git(project, "ls-files", "-z").decode("utf8").split("\0")
    extras = [path.relative_to(project).as_posix() for path in project.glob("*.py")]
    for directory in ("h3_t8", "examples"):
        extras.extend(path.relative_to(project).as_posix() for path in (project / directory).rglob("*") if path.is_file())
    files = {}
    for name in sorted(set(tracked + extras + includes) - {""}):
        if not allowed_member(name) or (name not in includes and ignore.match_file(name)):
            continue
        path = project / name
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(project) or not path.is_file() or path.is_symlink():
            raise ValueError("Candidate source escaped its project: " + name)
        payload = path.read_bytes()
        if len(payload) > 40 * 1024**2 or SECRETS.search(payload):
            raise ValueError("Oversized candidate member or possible secret: " + name)
        files[name] = hashlib.sha256(payload).hexdigest()
    if not REQUIRED.issubset(files) or STATUS_MEMBER in files:
        raise ValueError("V2 candidate is missing runtime/guide files or shadows generated qualification metadata")
    features = json.loads((project / "features.json").read_text(encoding="utf8"))
    validate_registry(features["nodes"], features["nodes"])
    workflows = baseline(project)
    memory = validate_legacy_artifacts(project, files, workflows)
    return files, includes, workflows, memory


def candidate_status(*, bound_controls=False):
    source_version = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf8"))["project"]["version"]
    status = {"schema": SCHEMA, "baseline": BASELINE, "source_release_version": source_version,
            "registered_nodes": 339, "legacy_registry_prefix": 336, "legacy_workflow_json_count": 255,
            "v2_nodes": SUFFIX, "v2_candidate_workflows_packaged": 0,
            "human_qualified": False, "v2_human_review": "pending", "published": False,
            "publication_status": "not_published", "universal_quality_claim": False,
            "legacy_review_scope": "Preserved exact accepted legacy artifacts only; does not qualify FastH3 V2",
            "media_or_gpu_qualification_inferred_from_cpu": False}
    if bound_controls:
        status.update(bound_controls_accepted=True, human_qualified=True,
            v2_human_review='accepted_only_in_six_bound_control_scopes', v2_candidate_workflows_packaged=6,
            accepted_control_workflow_sha256s=ACCEPTED_V2_WORKFLOWS,
            human_qualification_scope='Bound trained/template/Sol audio-protected samples and two8s loops only; no universal quality or speed claim. Diagnostic observers removed while preserving native Core sampling controls; not a fresh GPU run.')
    return status


def validate_candidate_receipt(receipt):
    bound = receipt.get('bound_controls_accepted', False)
    if type(bound) is not bool:
        raise ValueError('Invalid bound-control qualification flag')
    status = candidate_status(bound_controls=bound)
    if any(receipt.get(key) != value or type(receipt.get(key)) is not type(value) for key, value in status.items()):
        raise ValueError("Receipt is not a human-pending unpublished FastH3 V2 development candidate")


def run_logged(command, root, label, *, cwd, env=None):
    with (root / (label + ".stdout.log")).open("xb") as stdout, (root / (label + ".stderr.log")).open("xb") as stderr:
        subprocess.run(command, cwd=cwd, env=environment(cpu=True) if env is None else env,
                       stdout=stdout, stderr=stderr, check=True, timeout=180)


def check_archive(archive, files):
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError("Official candidate archive membership differs")
        for member in package.infolist():
            if (not allowed_member(member.filename) or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16) or member.file_size > 40 * 1024**2
                    or hashlib.sha256(package.read(member)).hexdigest() != files[member.filename]):
                raise ValueError("Unsafe or changed candidate archive member: " + member.filename)
        if package.testzip() is not None:
            raise ValueError("Candidate zip integrity failed")


def build(root, *, zip_python=None, project=PROJECT):
    project, root = Path(project).resolve(strict=True), Path(root).resolve()
    if root.exists() or root == project / "artifacts" or not root.is_relative_to(project / "artifacts"):
        raise ValueError("New dedicated project-owned artifact output required")
    files, includes, workflows, memory = select_files(project)
    index = Path(git(project, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    index_sha = sha(index)
    root.mkdir(parents=True)
    snapshot = root / "snapshot"
    snapshot.mkdir()
    for name, digest in files.items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, destination)
        if sha(destination) != digest:
            raise ValueError("Source changed while copying: " + name)
    status = candidate_status(bound_controls=set(ACCEPTED_V2_WORKFLOWS).issubset(files))
    (snapshot / STATUS_MEMBER).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf8")
    source_files = dict(files)
    files[STATUS_MEMBER] = sha(snapshot / STATUS_MEMBER)
    git(snapshot, "init", "-q", "-b", "codex/package-fasth3-v2-dev")
    names = sorted(files)
    for start in range(0, len(names), 40):
        git(snapshot, "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
            "add", "--", *names[start:start + 40])
    archive = root / (SCOPED_ARCHIVE if status.get('bound_controls_accepted') else ARCHIVE)
    zip_code = (
        "import hashlib,inspect,json,sys; from comfy_cli.file_utils import zip_files; "
        "zip_files(sys.argv[1],includes=json.loads(sys.argv[2])); "
        "print(json.dumps({'implementation':inspect.getsourcefile(zip_files),"
        "'source_sha256':hashlib.sha256(inspect.getsource(zip_files).encode()).hexdigest(),"
        "'official_function':'comfy_cli.file_utils.zip_files'}))")
    run_logged([str(zip_python or sys.executable), "-B", "-c", zip_code, str(archive), json.dumps(includes)],
               root, "official-zip", cwd=snapshot)
    check_archive(archive, files)
    extracted = root / EXTRACTED
    with zipfile.ZipFile(archive) as package:
        package.extractall(extracted)
    if sha(index) != index_sha or any(sha(project / name) != digest for name, digest in source_files.items()):
        raise ValueError("Main index or frozen candidate source changed during build")
    validate_legacy_artifacts(extracted, files, workflows)
    receipt = {**status, "status": "official_fast_h3_v2_dev_archive_verified_CPU_import_pending",
               "archive": str(archive), "archive_sha256": sha(archive), "extracted": str(extracted),
               "files": files, "source_files": source_files, "main_index_sha256": index_sha,
               "index_path": str(index), "main_index_unchanged": True,
               "selflift_workflows": legacy.SELF_LIFT_WORKFLOWS, "t8_memory_workflow": memory,
               "legacy_workflow_sha256s": {name: hashlib.sha256(data).hexdigest() for name, data in workflows.items()},
               "candidate_graphs": "Only six exact accepted control workflows included when qualified; older unreviewed graphs remain separate local artifacts"}
    with (root / "receipt.json").open("x", encoding="utf8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
    return receipt


def cpu_worker(root, core):
    """Only called in a new interpreter with Core CPU arguments before import."""
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf8"))
    validate_candidate_receipt(receipt)
    package = root / EXTRACTED
    sys.path.insert(0, str(core))
    sys.argv = ["fasth3-v2-development-package-import", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise ValueError("Development archive import must be CPU-only")
    name = "_t8_fasth3_v2_extracted_candidate"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    nodes = asyncio.run(module.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in nodes]
    validate_registry(ids, json.loads((package / "features.json").read_text(encoding="utf8"))["nodes"])
    origins = {}
    for key, loaded in list(sys.modules.items()):
        location = getattr(loaded, "__file__", None)
        if key.startswith(name) and location:
            if not Path(location).resolve().is_relative_to(package.resolve()):
                raise ValueError("Actual development package import escaped extraction")
            origins[key] = str(location)
    if torch.cuda.is_initialized():
        raise ValueError("Extracted candidate initialized CUDA")
    result = {**candidate_status(bound_controls=receipt.get('bound_controls_accepted', False)), "status": "actual_extracted_dev339_CPU_import_pass_scoped_qualification_only",
              "nodes": len(ids), "prefix_sha256": PREFIX_SHA256, "package_origins": origins,
              "gpu_initialized": False, "archive_sha256": receipt["archive_sha256"]}
    with (root / "import-receipt.json").open("x", encoding="utf8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)


def verify(root, core, *, python=None):
    root, core = Path(root).resolve(strict=True), Path(core).resolve(strict=True)
    if not (core / "comfy/cli_args.py").is_file():
        raise ValueError("Actual ComfyUI checkout required for isolated CPU import")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf8"))
    validate_candidate_receipt(receipt)
    archive = root / (SCOPED_ARCHIVE if receipt.get('bound_controls_accepted') else ARCHIVE)
    if (Path(receipt["archive"]) != archive or Path(receipt["extracted"]) != root / EXTRACTED
            or sha(archive) != receipt["archive_sha256"]):
        raise ValueError("Candidate archive ownership or SHA changed")
    check_archive(archive, receipt["files"])
    package = root / EXTRACTED
    for name, digest in receipt["files"].items():
        if not allowed_member(name) or not (package / name).resolve().is_relative_to(package) or sha(package / name) != digest:
            raise ValueError("Extracted candidate membership or bytes changed")
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
    if actual != set(receipt["files"]):
        raise ValueError("Extra files in extracted development candidate")
    validate_candidate_receipt(json.loads((package / STATUS_MEMBER).read_text(encoding="utf8")))
    validate_legacy_artifacts(package, receipt["files"], baseline())
    env = environment(cpu=True)
    env["PYTHONPATH"] = str(core)
    run_logged([str(python or sys.executable), "-B", str(Path(__file__).resolve()), "--cpu-worker",
                "--root", str(root), "--core", str(core)], root, "actual-CPU-import", cwd=root, env=env)
    result = json.loads((root / "import-receipt.json").read_text(encoding="utf8"))
    validate_candidate_receipt(result)
    if result.get("nodes") != 339 or result.get("gpu_initialized") is not False or sha(receipt["index_path"]) != receipt["main_index_sha256"]:
        raise ValueError("Actual CPU import registry/CUDA/main-index gate failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--zip-python", type=Path, help="Interpreter with the installed official comfy-cli; no installation")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_worker:
        if args.core is None:
            parser.error("--core is required")
        cpu_worker(args.root.resolve(strict=True), args.core.resolve(strict=True))
        return
    if args.core is None:
        parser.error("--core is required; building and actual extracted CPU import are separate qualifications")
    if not args.verify_only:
        build(args.root, zip_python=args.zip_python)
    result = verify(args.root, args.core)
    print(json.dumps({key: value for key, value in result.items() if key != "package_origins"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
