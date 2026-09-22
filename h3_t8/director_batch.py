"""Durable, frozen, sequential Director batches; never submits work on read.

The request receipt is the authority for a submitted shot.  A batch only
coordinates immutable shot inputs and records output-media evidence; it never
interprets an old output filename as proof that a shot finished.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

from .director_project import atomic_json, contained, file_sha, identity, referenced_assets, sha


SCHEMA = "t8.minimax_h3.director_batch.v1"


def _request_id(batch_id, index, shot_id, attempt):
    suffix = f"{index}:{shot_id}" if attempt == 0 else f"{index}:{shot_id}:retry:{attempt}"
    return str(uuid.uuid5(uuid.UUID(batch_id), suffix))


def _path(store, batch_id):
    return contained(store.root, f"batches/{identity(batch_id)}.json")


def _fingerprint(project, seed, built, resources):
    return sha({"project": project, "seed": seed, "built": built, "resources": resources})


_MODEL_INPUTS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "VAELoader": ("vae_name", "vae"),
    "CLIPLoader": ("clip_name", "clip"),
    "MiniMaxH3LoRACompatibilityLoaderT8Advanced": ("lora_name", "loras"),
    "MiniMaxH3LearnedLatentUpscaleT8Advanced": ("model_name", "latent_upscale_models"),
    "MiniMaxH3SemanticBridgeConfigT8": ("model_name", "semantic_bridge"),
    "MiniMaxH3HyperFlowLoaderT8Advanced": ("hyperflow_file", "hyperflow_selection"),
}


def capture_resources(store, project, built, resolve_model):
    """Hash actual selected weight files and referenced assets once per batch.

    resolve_model(folder,name) must use the same configured Core search roots as
    the node, not an invented models_dir or a filename heuristic.
    """
    models = {}
    for plan in built:
        for node in plan["prompt"].values():
            spec = _MODEL_INPUTS.get(node.get("class_type"))
            if spec is None:
                continue
            field, folder = spec
            name = node["inputs"][field]
            key = f"{folder}:{name}"
            if key in models:
                continue
            path = Path(resolve_model(folder, name)).resolve()
            if not path.is_file():
                raise ValueError(f"冻结批次缺少所选模型：{folder}/{name}")
            models[key] = {"folder": folder, "name": name, "path": str(path),
                           "size": path.stat().st_size, "sha256": file_sha(path)}
    assets = {}
    for aid in sorted(referenced_assets(project)):
        item = store.asset(aid, verify=True)
        assets[aid] = {"sha256": item["sha256"], "server_path": item["server_path"]}
    return {"models": models, "assets": assets}


def verify_resources(store, resources, resolve_model):
    """Fail closed if selected bytes or resolution changed after batch start."""
    for item in resources["models"].values():
        path = Path(resolve_model(item["folder"], item["name"])).resolve()
        if (str(path) != item["path"] or not path.is_file() or
                path.stat().st_size != item["size"] or file_sha(path) != item["sha256"]):
            raise ValueError(f"冻结批次模型已变化：{item['folder']}/{item['name']}；不会提交下一镜")
    for aid, frozen in resources["assets"].items():
        item = store.asset(aid, verify=True)
        if (item["sha256"] != frozen["sha256"] or item["server_path"] != frozen["server_path"]):
            raise ValueError(f"冻结批次素材已变化：{aid}；不会提交下一镜")


def create_batch(store, batch_id, project, seed, built, resources):
    """Persist a validated project's exact inputs before any GPU submission."""
    batch_id = identity(batch_id)
    identity(project["id"])
    shots = project["doc"]["shots"]
    if not isinstance(shots, list) or not shots or len(shots) > 200:
        raise ValueError("批次镜头数量必须为 1–200")
    if len({identity(shot["id"]) for shot in shots}) != len(shots):
        raise ValueError("批次镜头身份不能重复")
    if len(built) != len(shots) or not all(isinstance(plan.get("prompt"), dict) for plan in built):
        raise ValueError("每个冻结镜头都需要正式 Core 生成图")
    seed = int(seed)
    if seed < 0 or seed >= 2**64 - len(shots):
        raise ValueError("批次种子超出范围")
    # Never accept a mutable client-supplied item list, receipt or job state.
    batch = {
        "schema": SCHEMA,
        "id": batch_id,
        "project_id": project["id"],
        "fingerprint": _fingerprint(project, seed, built, resources),
        "project": project,
        "resources": resources,
        "items": [
            {
                "shot_id": shot["id"],
                "seed": seed + index,
                "request_id": _request_id(batch_id, index, shot["id"], 0),
                "attempt": 0,
                "previous_request_ids": [],
                "built": built[index],
            }
            for index, shot in enumerate(shots)
        ],
    }
    path = _path(store, batch_id)
    if path.exists():
        existing = load_batch(store, batch_id)
        if existing != batch:
            raise ValueError("批次身份已用于不同的项目或生成配置")
        return existing
    atomic_json(path, batch)
    return batch


def load_batch(store, batch_id):
    batch = json.loads(_path(store, batch_id).read_text(encoding="utf-8"))
    if batch.get("schema") != SCHEMA or batch.get("id") != identity(batch_id):
        raise ValueError("批次身份或格式已损坏")
    project = batch.get("project")
    if not isinstance(project, dict) or project.get("id") != batch.get("project_id"):
        raise ValueError("批次项目身份已损坏")
    items = batch.get("items")
    shots = project.get("doc", {}).get("shots", [])
    if not isinstance(items, list) or not items or len(items) != len(shots):
        raise ValueError("批次镜头列表已损坏")
    for index, (shot, item) in enumerate(zip(shots, items)):
        if shot.get("id") != identity(item.get("shot_id")):
            raise ValueError("批次镜头顺序已损坏")
        attempt = item.get("attempt")
        if type(attempt) is not int or not 0 <= attempt <= 50:
            raise ValueError("批次尝试次数已损坏")
        previous = [_request_id(batch["id"], index, shot["id"], number) for number in range(attempt)]
        if item.get("previous_request_ids") != previous:
            raise ValueError("批次旧尝试身份已损坏")
        expected_request_id = _request_id(batch["id"], index, shot["id"], attempt)
        if item.get("request_id") != expected_request_id:
            raise ValueError("批次请求身份已损坏")
        if type(item.get("seed")) is not int or item["seed"] < 0 or item["seed"] >= 2**64:
            raise ValueError("批次种子已损坏")
    if batch.get("fingerprint") != _fingerprint(
        project, items[0]["seed"], [item.get("built") for item in items], batch.get("resources"),
    ):
        raise ValueError("批次快照身份已损坏")
    if [item["seed"] for item in items] != list(range(items[0]["seed"], items[0]["seed"] + len(items))):
        raise ValueError("批次种子顺序已损坏")
    return batch


def retry_batch_item(store, batch_id, index):
    """Explicitly abandon one terminal/verified-stale attempt, not past successes."""
    batch = load_batch(store, batch_id)
    item = batch["items"][index]
    if item["attempt"] >= 50:
        raise ValueError("此镜头已经达到安全重试上限")
    item["previous_request_ids"].append(item["request_id"])
    item["attempt"] += 1
    item["request_id"] = _request_id(batch_id, index, item["shot_id"], item["attempt"])
    atomic_json(_path(store, batch_id), batch)
    return load_batch(store, batch_id)


def _video_paths(outputs, output_root):
    """Resolve only Core output MP4 metadata under the configured output root."""
    result = []
    for value in outputs.values():
        if not isinstance(value, dict):
            continue
        for key in ("videos", "gifs", "video", "images"):
            for item in value.get(key) or []:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not isinstance(filename, str) or not filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                    continue
                if item.get("type", "output") != "output":
                    continue
                folder = item.get("subfolder") or ""
                if not isinstance(folder, str):
                    continue
                path = contained(output_root, str(Path(folder.replace("\\", "/")) / filename))
                result.append(path)
    return result


def _delivery_contract(item):
    """Use the frozen compiler shot, never the mutable current project."""
    try:
        shot = item["built"]["shot"]
        expected = {
            "frames": shot["time"]["delivery_trim_frames"],
            "width": shot["canvas"]["width"],
            "height": shot["canvas"]["height"],
            "seconds": shot["time"]["requested_seconds"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("冻结镜头缺少音画交付规格") from error
    if (any(type(expected[key]) is not int or expected[key] <= 0 for key in ("frames", "width", "height"))
            or type(expected["seconds"]) not in (int, float)
            or not math.isfinite(expected["seconds"]) or expected["seconds"] <= 0):
        raise ValueError("冻结镜头音画交付规格已损坏")
    return expected


def _check_delivery(evidence, expected):
    if any(evidence[key] != expected[key] for key in ("frames", "width", "height")):
        raise ValueError("成片帧数或画幅与冻结镜头不一致")
    samples = evidence.get("audio_samples")
    rate = evidence.get("audio_sample_rate")
    if (type(samples) is not int or samples <= 0 or type(rate) is not int or rate <= 0):
        raise ValueError("成片音频长度与冻结镜头不一致")
    # Allow at most two AAC-sized priming/padding frames, not an arbitrary
    # fraction of a short scene. The small capped relative term covers longer
    # non-AAC Core outputs without accepting materially truncated audio.
    tolerance = max(2048 / rate, min(0.05, 0.02 * expected["seconds"]))
    if abs(samples / rate - expected["seconds"]) > tolerance:
        raise ValueError("成片音频长度与冻结镜头不一致")


def _media_evidence(paths, output_root, expected):
    """Decode complete Director video and mandatory AV audio once."""
    import av

    evidence = []
    for path in paths:
        try:
            with av.open(str(path)) as container:
                video = next((stream for stream in container.streams if stream.type == "video"), None)
                if video is None:
                    raise ValueError("没有视频轨道")
                width, height = video.width, video.height
                frames = sum(1 for _ in container.decode(video))
                if frames == 0 or width <= 0 or height <= 0:
                    raise ValueError("没有完整的可解码视频帧")
            with av.open(str(path)) as container:
                audio = next((stream for stream in container.streams if stream.type == "audio"), None)
                if audio is None:
                    raise ValueError("导演台成片缺少音频轨道")
                audio_frames = 0
                audio_samples = 0
                for frame in container.decode(audio):
                    audio_frames += 1
                    audio_samples += frame.samples
                if audio_frames == 0:
                    raise ValueError("音频轨道无法解码")
                audio_sample_rate = audio.rate
        except Exception as error:
            raise ValueError(f"输出视频无法完整解码：{path.name}: {error}") from error
        item = {"file": str(path.relative_to(Path(output_root).resolve())),
                "sha256": file_sha(path), "frames": frames,
                "width": width, "height": height,
                "audio_stream": True, "audio_frames": audio_frames,
                "audio_samples": audio_samples, "audio_sample_rate": audio_sample_rate}
        _check_delivery(item, expected)
        evidence.append(item)
    return evidence


def _refresh_cached_media(evidence, paths, output_root, expected):
    """Check the persisted decoder receipt without decoding every video on poll.

    The receipt digest catches accidental damage to evidence fields. File hashes
    are still rechecked because the output itself may have changed on disk.
    """
    old_fields = {"file", "sha256", "frames", "width", "height", "audio_stream", "audio_frames"}
    required = old_fields | {"audio_samples", "audio_sample_rate"}
    if not isinstance(evidence, list) or len(evidence) != len(paths):
        raise ValueError("已核对视频数量或形状已损坏")
    refreshed = []
    root = Path(output_root).resolve()
    for item, path in zip(evidence, paths):
        if not isinstance(item, dict) or set(item) not in (old_fields, required):
            raise ValueError("已核对视频字段已损坏")
        if item["file"] != str(path.relative_to(root)):
            raise ValueError("已核对视频路径与输出不一致")
        for field in ("frames", "width", "height", "audio_frames"):
            if type(item[field]) is not int or item[field] <= 0:
                raise ValueError("已核对视频尺寸或帧数已损坏")
        if item["audio_stream"] is not True:
            raise ValueError("已核对视频缺少音频轨道")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError("已核对视频哈希已损坏")
        if set(item) == old_fields:
            # Old receipts predate audio-duration evidence. Re-decode once,
            # compare every old field, then persist the enriched receipt.
            decoded = _media_evidence([path], output_root, expected)[0]
            if {key: decoded[key] for key in old_fields} != item:
                raise ValueError("旧音画回执与实际输出不一致")
            refreshed.append(decoded)
        else:
            _check_delivery(item, expected)
            refreshed.append({**item, "sha256": file_sha(path)})
    return refreshed


def batch_status(store, batch_id, status_lookup, output_root, *, core_epoch=None):
    """Reconcile receipts against Core and media, with no queue side effect."""
    batch = load_batch(store, batch_id)
    records = []
    for item in batch["items"]:
        receipt_path = contained(store.root, f"requests/{item['request_id']}.json")
        row = {key: value for key, value in item.items() if key != "built"}
        row.update(state="not_submitted", prompt_id=None, retry_available=False)
        if receipt_path.exists():
            receipt = {}
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if not isinstance(receipt, dict):
                    raise ValueError("生成回执顶层形状已损坏")
                expected = sha({"project": {**batch["project"], "current": item["shot_id"]}, "seed": item["seed"], "shot_id": item["shot_id"]})
                if receipt.get("fingerprint") != expected:
                    raise ValueError("生成回执与冻结批次不一致")
                prompt_id = identity(receipt["prompt_id"])
                row["prompt_id"] = prompt_id
                if receipt.get("state") not in {"queued", "submitting"}:
                    raise ValueError("生成回执状态已损坏")
                terminal = receipt.get("terminal")
                observed = terminal or status_lookup(prompt_id)
                if not isinstance(observed, dict):
                    raise ValueError("生成回执终态形状已损坏")
                state = observed.get("state", "unknown")
                if state == "success":
                    expected_delivery = _delivery_contract(item)
                    outputs = observed.get("outputs") or {}
                    if not isinstance(outputs, dict):
                        raise ValueError("生成输出形状已损坏")
                    paths = _video_paths(outputs, output_root)
                    if not paths or not all(path.is_file() and path.stat().st_size > 0 for path in paths):
                        row["state"] = "needs_review"
                    else:
                        frozen = receipt.get("media_evidence")
                        digest = receipt.get("media_evidence_digest")
                        if frozen is None or digest is None:
                            # A legacy receipt without a digest cannot vouch
                            # for its stored decode fields: validate once.
                            media = _media_evidence(paths, output_root, expected_delivery)
                        else:
                            if not isinstance(digest, str) or digest != sha(frozen):
                                raise ValueError("已核对视频回执摘要已损坏")
                            media = _refresh_cached_media(frozen, paths, output_root, expected_delivery)
                        old_fields = {"file", "sha256", "frames", "width", "height", "audio_stream", "audio_frames"}
                        legacy_upgrade = (isinstance(frozen, list) and len(frozen) == len(media)
                                          and all(isinstance(old, dict) and set(old) == old_fields
                                                  and {key: new[key] for key in old_fields} == old
                                                  for old, new in zip(frozen, media)))
                        if frozen is not None and frozen != media and not legacy_upgrade:
                            row["state"] = "needs_review"
                        else:
                            row["state"] = "success"
                            row["media"] = media
                            if terminal is None or frozen is None or digest is None or legacy_upgrade:
                                receipt["terminal"] = {"state": "success", "outputs": observed.get("outputs") or {}}
                                receipt["media_evidence"] = media
                                receipt["media_evidence_digest"] = sha(media)
                                atomic_json(receipt_path, receipt)
                elif state in {"queued", "running", "error", "cancelled"}:
                    row["state"] = state
                    if state in {"error", "cancelled"} and terminal is None:
                        receipt["terminal"] = {"state": state, "outputs": observed.get("outputs") or {}}
                        atomic_json(receipt_path, receipt)
                else:
                    # Queue/history can disappear after a Core restart; do not
                    # turn an ambiguous submission into a new request.
                    row["state"] = "unknown"
                    row["retry_available"] = bool(
                        core_epoch and receipt.get("core_epoch") and receipt["core_epoch"] != core_epoch
                    )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                row["state"] = "needs_review"
            if row["state"] in {"error", "cancelled"}:
                row["retry_available"] = True
            elif row["state"] == "needs_review":
                # A damaged receipt must not permit an in-flight job to be
                # queued again merely because the parser could not read it.
                try:
                    old_id = identity(receipt["prompt_id"])
                    current = status_lookup(old_id)
                    current_state = current.get("state") if isinstance(current, dict) else "unknown"
                    row["retry_available"] = (
                        current_state not in {"running", "queued"}
                        and (current_state != "unknown" or bool(
                            core_epoch and receipt.get("core_epoch") and receipt["core_epoch"] != core_epoch
                        ))
                    )
                except (ValueError, KeyError, TypeError, OSError):
                    row["retry_available"] = False
        records.append(row)
    next_index = next((i for i, row in enumerate(records) if row["state"] != "success"), None)
    return {
        "schema": SCHEMA,
        "id": batch["id"],
        "project_id": batch["project_id"],
        "fingerprint": batch["fingerprint"],
        "items": records,
        "next_index": next_index,
        "complete": next_index is None,
    }
