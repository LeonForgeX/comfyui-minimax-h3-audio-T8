"""Same-origin director services.

D1 owns the persistent project/asset contract. D2a–D2c add explicit,
validated native-generation routes that use Core's normal in-process queue.
D3 either compiles the four compatible H3 route patches or hands the user an
exact allow-listed native workflow; it never disguises a handoff as a queue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import wraps
from pathlib import Path
import time
import uuid

from .director_project import (
    ProjectConflict,
    ProjectStore,
    compile_project,
    contained,
    identity,
    new_project,
    validate_project,
    atomic_json,
    sha,
)
from .director_generation import (
    build_director_generation_prompt,
    cancel_director_prompt,
    director_model_catalog,
    director_job_status,
    queue_director_prompt,
)
from .director_capabilities import inspect_director_capabilities
from .director_d3 import export_d3_package, handoff_d3_route, inspect_d3_routes
from .director_batch import (
    batch_status, capture_resources, create_batch, load_batch,
    retry_batch_item, verify_resources,
)

PREFIX = "/minimax_h3_t8/director"
_REGISTERED = False
_GENERATE_LOCK = asyncio.Lock()
_CORE_EPOCH = str(uuid.uuid4())


def _record_deleted_queue_receipt(store, prompt_id):
    """Persist only a confirmed queue deletion, never a running interruption."""
    matches = []
    request_dir = contained(store.root, "requests")
    for path in request_dir.glob("*.json") if request_dir.is_dir() else ():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (isinstance(receipt, dict) and receipt.get("prompt_id") == prompt_id
                and receipt.get("state") == "queued" and receipt.get("terminal") is None):
            matches.append((path, receipt))
    if len(matches) != 1:
        return False
    path, receipt = matches[0]
    receipt["terminal"] = {"state": "cancelled", "outputs": {}}
    receipt["cancelled_at"] = time.time()
    atomic_json(path, receipt)
    return True


def _resolve_director_resource(folder, name):
    if folder == "hyperflow_selection":
        from .nodes_hyperflow_advanced import _resolve

        return _resolve(name)
    if folder == "semantic_bridge":
        from .nodes_semantic_bridge import resolve_model

        return resolve_model(name)
    import folder_paths

    path = folder_paths.get_full_path(folder, name)
    if not path:
        raise ValueError(f"冻结批次缺少所选模型：{folder}/{name}")
    return path


async def _submit_director_request_locked(store, project, shot_id, seed, request_id, client_id=None, *, built=None):
    """Exactly-once receipt and Core queue submission under _GENERATE_LOCK."""
    request_id = identity(request_id)
    shot_id = identity(shot_id)
    seed = int(seed)
    fingerprint = hashlib.sha256(json.dumps(
        {"project": project, "shot_id": shot_id, "seed": seed},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    receipt_path = contained(store.root, f"requests/{request_id}.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("fingerprint") != fingerprint:
            raise ValueError("同一个请求 ID 对应不同配置；请新建生成请求")
        if receipt.get("state") == "queued":
            return receipt["result"], 200
        return {"error": "提交状态尚未确认；请先按任务 ID 查询，不会自动重发", "prompt_id": receipt.get("prompt_id")}, 409
    if built is None:
        built = await asyncio.to_thread(
            build_director_generation_prompt, project, shot_id, store, seed=seed,
        )
    prompt_id = str(uuid.uuid4())
    atomic_json(receipt_path, {"state": "submitting", "fingerprint": fingerprint,
                               "prompt_id": prompt_id, "core_epoch": _CORE_EPOCH})
    try:
        prompt_id = await queue_director_prompt(built["prompt"], client_id, prompt_id=prompt_id)
    except Exception:
        # Unknown history cannot prove a queue write did not happen; retaining
        # the reservation prevents an automatic duplicate on a lost response.
        raise
    result = {
        "prompt_id": prompt_id, "recipe": built["recipe"],
        "d3_routes": built.get("d3_routes", []), "seed": built["seed"],
        "turbo_lora": built["turbo_lora"], "sampling": built["sampling"],
        "report": built["report"],
    }
    shot_source = next((shot for shot in project.get("doc", {}).get("shots", []) if shot.get("id") == shot_id), {})
    atomic_json(receipt_path, {
        "state": "queued", "fingerprint": fingerprint, "prompt_id": prompt_id,
        "core_epoch": _CORE_EPOCH,
        "project_id": project["id"], "shot_id": shot_id,
        "shot_rev": shot_source.get("rev"), "submitted_at": time.time(), "result": result,
    })
    return result, 202


def director_project_results(store, project_id, status_lookup=director_job_status, *, shot_ids=(), output_root=None):
    """Recover a project's queued and finished shots from durable request receipts.

    Core history may disappear after a restart, so terminal output metadata is
    copied into the receipt the first time it is observed.  The media remains
    in Core's output directory and is served by Core's normal /view route.
    """
    project_id = identity(project_id)
    records = []
    for path in (store.root / "requests").glob("*.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            result = receipt.get("result") or {}
            report = result.get("report") or {}
            owner = receipt.get("project_id") or report.get("project_id")
            if owner != project_id or receipt.get("state") != "queued":
                continue
            shot_id = receipt.get("shot_id") or (report.get("selection") or {}).get("shot_id")
            if not shot_id:
                continue
            prompt_id = identity(receipt["prompt_id"])
            terminal = receipt.get("terminal")
            if terminal is None:
                current = status_lookup(prompt_id)
                if current.get("state") in {"success", "error"}:
                    terminal = {"state": current["state"], "outputs": current.get("outputs") or {}}
                    receipt["terminal"] = terminal
                    atomic_json(path, receipt)
            records.append({
                "shot_id": shot_id,
                "prompt_id": prompt_id,
                "state": terminal["state"] if terminal else "pending",
                "outputs": terminal.get("outputs", {}) if terminal else {},
                "recipe": result.get("recipe", "director"),
                "submitted_at": receipt.get("submitted_at") or path.stat().st_mtime,
                "shot_rev": receipt.get("shot_rev"),
            })
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A damaged unrelated receipt must not hide the remaining films.
            continue
    if output_root is not None:
        # Older Director tasks predate request receipts. Their SafeAVSave prefix
        # is deterministic: T8_Director/<project UUID first 8>/<shot UUID first 8>.
        # Only recover files for shot identities supplied by this project page.
        root = Path(output_root).resolve()
        folder = root / "T8_Director" / project_id[:8]
        if not folder.resolve().is_relative_to(root):
            raise ValueError("成片目录越界")
        known_media = {
            (item.get("subfolder"), item.get("filename"))
            for record in records
            for output in record["outputs"].values()
            if isinstance(output, dict)
            for item in (output.get("images") or [])
            if isinstance(item, dict)
        }
        for raw_shot_id in shot_ids:
            shot_id = identity(raw_shot_id)
            for media in folder.glob(f"{shot_id[:8]}_*.mp4"):
                subfolder = f"T8_Director\\{project_id[:8]}"
                if (subfolder, media.name) in known_media:
                    continue
                records.append({
                    "shot_id": shot_id, "prompt_id": None, "state": "success",
                    "outputs": {"legacy": {"images": [{
                        "filename": media.name,
                        "subfolder": subfolder,
                        "type": "output",
                    }]}},
                    "recipe": "既有成片", "submitted_at": media.stat().st_mtime,
                    "shot_rev": None, "recovered_by": "project_and_shot_output_prefix",
                })
    records.sort(key=lambda item: (item["submitted_at"], item["prompt_id"]), reverse=True)
    return {"project_id": project_id, "results": records}


def get_store():
    import folder_paths

    return ProjectStore(
        folder_paths.get_user_directory(), folder_paths.get_input_directory()
    )


def register_director_routes():
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        # CPU/schema tools do not necessarily have the Core server module initialized.
        return False
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return False
    routes = server.routes

    def guarded(function):
        @wraps(function)
        async def call(request):
            try:
                return await function(request)
            except ProjectConflict as error:
                return web.json_response(
                    {"error": str(error), "code": "revision_conflict"}, status=409
                )
            except FileNotFoundError:
                return web.json_response(
                    {"error": "项目或素材不存在，请重连；当前草稿不会被覆盖"},
                    status=404,
                )
            except (ValueError, KeyError, TypeError) as error:
                return web.json_response({"error": str(error)}, status=400)
            except Exception as error:
                import logging

                logging.exception("Director service failed")
                return web.json_response(
                    {
                        "error": f"保存／读取失败，可保留草稿后重试：{type(error).__name__}"
                    },
                    status=500,
                )

        return call

    @routes.get(PREFIX + "/projects")
    @guarded
    async def projects(_request):
        return web.json_response(
            {"projects": await asyncio.to_thread(get_store().list)}
        )

    @routes.get(PREFIX + "/projects/{project_id}")
    @guarded
    async def load(request):
        project = await asyncio.to_thread(
            get_store().load, request.match_info["project_id"]
        )
        return web.json_response(project)

    @routes.post(PREFIX + "/projects/{project_id}")
    @guarded
    async def save(request):
        body = await request.json()
        if identity(request.match_info["project_id"]) != body["project"]["id"]:
            raise ValueError("项目路径与内容身份不一致")
        result = await asyncio.to_thread(
            get_store().save, body["project"], body["expected_revision"]
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/compile")
    @guarded
    async def compile(request):
        body = await request.json()
        result = await asyncio.to_thread(
            compile_project, body["project"], get_store(), shot_id=body.get("shot_id")
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/validate")
    @guarded
    async def validate(request):
        body = await request.json()
        return web.json_response({"project": validate_project(body["project"])})

    @routes.post(PREFIX + "/export")
    @guarded
    async def export(request):
        body = await request.json()
        result = await asyncio.to_thread(
            compile_project, body["project"], get_store(), shot_id=body["shot_id"]
        )
        if not result["ready"]:
            return web.json_response(
                {
                    "error": "预检未通过，保留项目但不能导出可执行预检图",
                    "report": result,
                },
                status=422,
            )
        return web.json_response(
            {
                **export_preflight_workflow(body["project"], body["shot_id"]),
                "report": result,
            }
        )

    @routes.post(PREFIX + "/generate")
    @guarded
    async def generate(request):
        """Queue the validated D2a–D2c recipe selected by the current shot."""
        body = await request.json()
        store = get_store()
        request_id = identity(body.get("request_id") or str(uuid.uuid4()))
        async with _GENERATE_LOCK:
            result, status = await _submit_director_request_locked(
                store, body["project"], body["shot_id"],
                body.get("seed", 26091901), request_id, body.get("client_id"),
            )
        return web.json_response(result, status=status)

    @routes.post(PREFIX + "/batches")
    @guarded
    async def start_batch(request):
        body = await request.json()
        store = get_store()
        project = body["project"]
        batch_id = identity(body["batch_id"])
        async with _GENERATE_LOCK:
            try:
                existing = load_batch(store, batch_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if sha(existing["project"]) != sha(project) or existing["items"][0]["seed"] != int(body["seed"]):
                    raise ValueError("批次身份已用于不同的项目或生成配置")
                return web.json_response({"id": existing["id"], "project_id": existing["project_id"], "fingerprint": existing["fingerprint"]})
        report = await asyncio.to_thread(compile_project, project, store)
        if not report["ready"]:
            return web.json_response({"error": "全部生成前检查未通过", "report": report}, status=422)
        async with _GENERATE_LOCK:
            try:
                existing = load_batch(store, batch_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if sha(existing["project"]) != sha(project) or existing["items"][0]["seed"] != int(body["seed"]):
                    raise ValueError("批次身份已用于不同的项目或生成配置")
                return web.json_response({"id": existing["id"], "project_id": existing["project_id"], "fingerprint": existing["fingerprint"]})
            seed = int(body["seed"])
            prepared = []
            for index, shot in enumerate(project["doc"]["shots"]):
                shot_project = {**project, "current": shot["id"]}
                prepared.append(await asyncio.to_thread(
                    build_director_generation_prompt, shot_project, shot["id"],
                    store, seed=seed + index,
                ))
            resources = await asyncio.to_thread(
                capture_resources, store, project, prepared, _resolve_director_resource,
            )
            batch = create_batch(store, batch_id, project, seed, prepared, resources)
        return web.json_response({"id": batch["id"], "project_id": batch["project_id"], "fingerprint": batch["fingerprint"]})

    @routes.get(PREFIX + "/batches/{batch_id}")
    @guarded
    async def read_batch(request):
        import folder_paths

        store = get_store()
        async with _GENERATE_LOCK:
            state = await asyncio.to_thread(
                batch_status, store, request.match_info["batch_id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
        return web.json_response(state)

    @routes.post(PREFIX + "/batches/{batch_id}/continue")
    @guarded
    async def continue_batch(request):
        import folder_paths

        body = await request.json()
        store = get_store()
        async with _GENERATE_LOCK:
            batch = load_batch(store, request.match_info["batch_id"])
            status = await asyncio.to_thread(
                batch_status, store, batch["id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
            index = status["next_index"]
            if index is None:
                return web.json_response({"complete": True, "batch": status})
            row = status["items"][index]
            if row["state"] != "not_submitted":
                return web.json_response({"error": "当前镜头任务尚未确认，不能重复提交", "batch": status}, status=409)
            item = batch["items"][index]
            project = {**batch["project"], "current": item["shot_id"]}
            await asyncio.to_thread(
                verify_resources, store, batch["resources"], _resolve_director_resource,
            )
            result, code = await _submit_director_request_locked(
                store, project, item["shot_id"], item["seed"],
                item["request_id"], body.get("client_id"), built=item["built"],
            )
        return web.json_response({**result, "shot_id": item["shot_id"], "batch_id": batch["id"]}, status=code)

    @routes.post(PREFIX + "/batches/{batch_id}/retry")
    @guarded
    async def retry_batch(request):
        import folder_paths

        body = await request.json()
        if body.get("confirm_abandoned") is not True:
            raise ValueError("请明确确认弃用旧尝试，不能由刷新页面自动重试")
        store = get_store()
        async with _GENERATE_LOCK:
            state = await asyncio.to_thread(
                batch_status, store, request.match_info["batch_id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
            index = state["next_index"]
            if index is None:
                return web.json_response({"error": "批次全部完成，不可重试"}, status=409)
            row = state["items"][index]
            if not row["retry_available"]:
                return web.json_response({"error": "旧任务仍在当前 Core 中或提交状态未确认；不能重试", "batch": state}, status=409)
            batch = retry_batch_item(store, state["id"], index)
        return web.json_response({"batch_id": batch["id"], "index": index,
                                  "attempt": batch["items"][index]["attempt"],
                                  "previous_request_ids": batch["items"][index]["previous_request_ids"]})

    @routes.post(PREFIX + "/d3/compile")
    @guarded
    async def d3_compile(request):
        """Compile the selected D3 graph without submitting it to Core."""
        body = await request.json()
        built = await asyncio.to_thread(
            build_director_generation_prompt,
            body["project"],
            body["shot_id"],
            get_store(),
            seed=int(body.get("seed", 26091901)),
        )
        return web.json_response(
            {
                "schema": "t8.minimax_h3.director_d3_compiled_graph.v1",
                "recipe": built["recipe"],
                "d3_routes": built.get("d3_routes", []),
                "seed": built["seed"],
                "nodes": {
                    str(node_id): {
                        "class_type": node.get("class_type"),
                        "inputs": node.get("inputs", {}),
                    }
                    for node_id, node in built["prompt"].items()
                },
                "report": built["report"],
                "warning": "只编译图，不排队、不加载模型、不代表 GPU 或感知质量通过。",
            }
        )

    @routes.get(PREFIX + "/jobs/{prompt_id}")
    @guarded
    async def job(request):
        return web.json_response(
            director_job_status(request.match_info["prompt_id"])
        )

    @routes.get(PREFIX + "/results/{project_id}")
    @guarded
    async def results(request):
        import folder_paths

        shot_ids = request.query.getall("shot", [])
        if len(shot_ids) > 200:
            raise ValueError("一次最多查询 200 个镜头结果")
        async with _GENERATE_LOCK:
            records = await asyncio.to_thread(
                director_project_results, get_store(), request.match_info["project_id"],
                shot_ids=shot_ids, output_root=folder_paths.get_output_directory(),
            )
        return web.json_response(records)

    @routes.post(PREFIX + "/jobs/{prompt_id}/cancel")
    @guarded
    async def cancel(request):
        prompt_id = identity(request.match_info["prompt_id"])
        async with _GENERATE_LOCK:
            result = cancel_director_prompt(prompt_id)
            if result["deleted_from_queue"]:
                result["receipt_recorded"] = _record_deleted_queue_receipt(get_store(), prompt_id)
        return web.json_response(result)

    @routes.post(PREFIX + "/assets")
    @guarded
    async def upload(request):
        store = get_store()
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file" or not part.filename:
            raise ValueError("上传必须包含 file")
        asset_id = str(uuid.uuid4())
        suffix = Path(part.filename).suffix.lower()
        if suffix not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
        }:
            raise ValueError("请选择标准图片、视频或音频文件")
        path = contained(store.input_root, f"t8_director/{asset_id}/source{suffix}")
        path.parent.mkdir(parents=True, exist_ok=False)
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await part.read_chunk(1024 * 1024):
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise ValueError(
                            "单素材上传上限1GiB，请分段或压缩后重试；未截断保存"
                        )
                    await asyncio.to_thread(stream.write, chunk)
            asset = await asyncio.to_thread(
                store.register_asset, path, asset_id, part.filename
            )
            return web.json_response(asset, status=201)
        except BaseException:
            path.unlink(missing_ok=True)  # Only this incomplete, server-created upload.
            path.parent.rmdir()
            raise

    @routes.get(PREFIX + "/assets/{asset_id}")
    @guarded
    async def asset(request):
        store = get_store()
        asset = await asyncio.to_thread(store.asset, request.match_info["asset_id"])
        return web.FileResponse(contained(store.input_root, asset["server_path"]))

    @routes.get(PREFIX + "/assets/{asset_id}/input-preview")
    @guarded
    async def input_preview(request):
        store = get_store()
        prepared = await asyncio.to_thread(
            store.prepare_image,
            request.match_info["asset_id"],
            int(request.query["width"]),
            int(request.query["height"]),
        )
        return web.FileResponse(contained(store.input_root, prepared["server_path"]))

    @routes.get(PREFIX + "/ui")
    async def ui(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "index.html"
        )

    @routes.get(PREFIX + "/session.mjs")
    async def session(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "session.mjs"
        )

    @routes.get(PREFIX + "/sampling_ui.mjs")
    async def sampling_ui(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "sampling_ui.mjs"
        )

    @routes.get(PREFIX + "/default")
    async def default(_request):
        return web.json_response(new_project())

    @routes.get(PREFIX + "/capabilities")
    @guarded
    async def capabilities(_request):
        """Report D3 native entry points without pretending to queue them."""
        import nodes

        return web.json_response(
            inspect_director_capabilities(nodes.NODE_CLASS_MAPPINGS.keys())
        )

    @routes.get(PREFIX + "/models")
    @guarded
    async def models(_request):
        """Return only installed H3-compatible models for the Director selectors."""
        return web.json_response(director_model_catalog())

    @routes.get(PREFIX + "/d3/routes")
    @guarded
    async def d3_routes(_request):
        """List D3 native hand-off routes without touching a project or queue."""
        import nodes

        return web.json_response(inspect_d3_routes(node_ids=nodes.NODE_CLASS_MAPPINGS.keys()))

    @routes.post(PREFIX + "/d3/preflight")
    @guarded
    async def d3_preflight(request):
        """Preflight one saved Director shot before handing it to a D3 route."""
        body = await request.json()
        return web.json_response(
            await asyncio.to_thread(
                inspect_d3_routes,
                body.get("project"),
                body.get("shot_id"),
                get_store(),
                capability=body.get("capability"),
            )
        )

    @routes.post(PREFIX + "/d3/handoff")
    @guarded
    async def d3_handoff(request):
        """Return an exact allow-listed native workflow/README without queuing."""
        body = await request.json()
        result = await asyncio.to_thread(
            handoff_d3_route,
            str(body.get("capability", "")),
            body.get("file"),
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/d3/package")
    @guarded
    async def d3_package(request):
        """Export the current project plus an exact native route hand-off."""
        body = await request.json()
        result = await asyncio.to_thread(
            export_d3_package,
            str(body.get("capability", "")),
            body.get("project"),
            body.get("shot_id"),
            get_store(),
        )
        return web.json_response(result)

    _REGISTERED = True
    return True


def export_preflight_workflow(project, shot_id):
    """A real native CPU contract graph, not a mislabeled runnable GPU recipe."""
    from .director_project import canonical, validate_project

    project = validate_project(project)
    if shot_id not in {s["id"] for s in project["doc"]["shots"]}:
        raise ValueError("镜头身份不存在")
    values = [canonical(project), shot_id]
    api = {
        "1": {
            "class_type": "MiniMaxH3DirectorProjectT8",
            "inputs": {"project_json": values[0], "shot_id": shot_id},
            "_meta": {"title": "曜石导演台 · D1 CPU预检（不生成）"},
        }
    }
    workflow = {
        "id": str(uuid.uuid4()),
        "version": 0.4,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "MiniMaxH3DirectorProjectT8",
                "pos": [160, 140],
                "size": [520, 260],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": name, "type": kind, "links": None}
                    for name, kind in (
                        ("compiled_prompt", "STRING"),
                        ("width", "INT"),
                        ("height", "INT"),
                        ("length", "INT"),
                        ("media_map_json", "STRING"),
                        ("report_json", "STRING"),
                    )
                ],
                "properties": {
                    "Node name for S&R": "MiniMaxH3DirectorProjectT8",
                    "cnr_id": "minimax-h3-audio-t8",
                },
                "widgets_values": values,
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "t8_director": {
                "project_id": project["id"],
                "scope": "D1 CPU preflight only; not generation",
            }
        },
    }
    return {"workflow": workflow, "api_snapshot": api}
