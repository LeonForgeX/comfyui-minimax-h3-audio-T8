"""Real D1 recovery/failure/interaction checks on an owned isolated CPU Core.

Never submits /prompt. The optional missing-file fixture is renamed recoverably
only below the explicitly supplied isolated input root, then restored in finally.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/minimax_h3_t8/director"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8852")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, errors, queues = [], [], []
    failures = {"save": False, "upload": False}
    original_path = backup_path = None
    picture = Path("F:/ComfyUI_00110_iphsk_1789210737 (1).png").read_bytes()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100}, accept_downloads=True
        )
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda e: errors.append(str(e)))

        def gateway(route):
            request = route.request
            if request.method == "POST" and request.url.rstrip("/").endswith("/prompt"):
                queues.append(request.url)
                route.abort()
            elif request.method == "POST" and (
                (failures["save"] and PREFIX + "/projects/" in request.url)
                or (
                    failures["upload"]
                    and request.url.rstrip("/").endswith(PREFIX + "/assets")
                )
            ):
                route.fulfill(
                    status=503,
                    content_type="application/json",
                    body='{"error":"QA模拟服务暂时失败，请重试"}',
                )
            else:
                route.continue_()

        context.route("**/*", gateway)

        def compilation():
            page.locator('[data-action="check"]').click()
            expect(page.locator("[data-dialog]")).to_be_visible()
            expect(page.locator("[data-dialog-title]")).to_contain_text("真实编译预检")
            return json.loads(page.locator("[data-dialog] pre").inner_text())

        try:
            saved = context.request.get(
                args.url + PREFIX + "/projects/" + args.project_id
            ).json()
            shot = next(s for s in saved["doc"]["shots"] if s["id"] == saved["current"])
            before_queue = context.request.get(args.url + "/queue").json()
            page.goto(args.url + PREFIX + "/ui?project_id=" + args.project_id)
            simple = page.locator('[data-field="simplePrompt"]')
            expect(simple).to_have_value(shot["simplePrompt"])
            expect(page.locator("[data-tray-id]")).to_have_count(4)
            expect(page.locator("[data-stage] img")).to_be_visible()
            expect(page.locator("[data-stage] img")).to_have_js_property(
                "naturalWidth", 1024
            )
            expect(page.locator("[data-stage] img")).to_have_js_property(
                "naturalHeight", 1536
            )
            expect(page.locator("[data-stage] img")).to_have_js_property(
                "complete", True
            )
            checks.append(
                "fresh_browser_after_actual_Core_restart_restores_server_project_and_original_media"
            )
            legacy = deepcopy(saved)
            legacy["version"] = 0
            with page.expect_file_chooser() as chooser:
                page.locator('[data-service="import"]').click()
            chooser.value.set_files(
                {
                    "name": "project-v0.json",
                    "mimeType": "application/json",
                    "buffer": json.dumps(legacy, ensure_ascii=False).encode("utf-8"),
                }
            )
            expect(page.locator("[data-notice]")).to_contain_text("项目已导入草稿")
            with page.expect_download() as download:
                page.locator('[data-service="project"]').click()
            download.value.save_as(args.output / "migrated-project.json")
            migrated = json.loads(
                (args.output / "migrated-project.json").read_text(encoding="utf-8")
            )
            assert migrated["version"] == 1 and migrated["doc"] == saved["doc"]
            assert (
                context.request.get(
                    args.url + PREFIX + "/projects/" + args.project_id
                ).json()["revision"]
                == saved["revision"]
            )
            checks.append(
                "server_authoritative_v0_to_v1_import_migration_preserves_drafts_without_saving_or_overwriting"
            )
            simple.fill("失败重试保留原文。@image1 @audio1")
            failures["save"] = True
            page.locator('[data-service="save"]').click()
            expect(page.locator("[data-save]")).to_contain_text("保存未完成")
            assert simple.input_value() == "失败重试保留原文。@image1 @audio1"
            assert (
                context.request.get(
                    args.url + PREFIX + "/projects/" + args.project_id
                ).json()["revision"]
                == saved["revision"]
            )
            failures["save"] = False
            page.locator('[data-service="save"]').click()
            expect(page.locator("[data-save]")).to_contain_text("已真实保存")
            assert (
                context.request.get(
                    args.url + PREFIX + "/projects/" + args.project_id
                ).json()["revision"]
                == saved["revision"] + 1
            )
            checks.append(
                "save_503_keeps_local_draft_and_server_revision_then_real_retry_succeeds"
            )

            def upload(name):
                with page.expect_file_chooser() as chooser:
                    page.locator('.o-stage [data-action="add"]').last.click()
                chooser.value.set_files(
                    {"name": name, "mimeType": "image/png", "buffer": picture}
                )

            failures["upload"] = True
            upload("retry.png")
            expect(page.locator("[data-notice]")).to_contain_text("失败")
            assert page.locator("[data-tray-id]").count() == 4
            failures["upload"] = False
            upload("retry.png")
            expect(page.locator("[data-tray-id]")).to_have_count(5)
            checks.append("upload_503_adds_no_phantom_asset_then_real_retry_succeeds")
            raw = b'\xef\xbb\xbf{  "nodes" : [],\r\n "links": [], "note":"raw bytes" }\r\n'
            before_text = simple.input_value()
            with page.expect_file_chooser() as chooser:
                page.locator('[data-service="import"]').click()
            chooser.value.set_files(
                {
                    "name": "unknown-original.json",
                    "mimeType": "application/json",
                    "buffer": raw,
                }
            )
            expect(page.locator("[data-dialog-title]")).to_contain_text("未知工作流")
            with page.expect_download() as download:
                page.locator('[data-service="original"]').click()
            download.value.save_as(args.output / "unknown-original.json")
            assert (args.output / "unknown-original.json").read_bytes() == raw
            page.locator('[data-action="close"]').click()
            assert simple.input_value() == before_text
            checks.append(
                "unknown_workflow_import_read_only_current_draft_untouched_download_exact_BOM_CRLF_bytes"
            )
            old = next(a for a in saved["assets"] if a["id"] == shot["first"])
            original_path = (args.input_root.resolve() / old["server_path"]).resolve()
            assert (
                original_path.is_relative_to(args.input_root.resolve())
                and original_path.is_file()
            )
            backup_path = original_path.with_name(
                original_path.name + ".qa-missing-backup"
            )
            assert not backup_path.exists()
            original_path.rename(backup_path)
            compiled = compilation()
            assert any("缺失" in e["message"] for e in compiled["errors"])
            page.locator('[data-action="close"]').click()
            page.locator('[data-service="reconnect"]').click()
            with page.expect_file_chooser() as chooser:
                page.locator('[data-reconnect-asset="' + old["id"] + '"]').click()
            chooser.value.set_files(
                {"name": "replacement.png", "mimeType": "image/png", "buffer": picture}
            )
            expect(page.locator("[data-notice]")).to_contain_text("重连完成")
            compiled = compilation()
            assert not compiled["errors"], compiled["errors"]
            new = next(
                m["asset_id"]
                for m in compiled["current_shot"]["media_map"]
                if m["role"] == "first_frame"
            )
            assert new != old["id"]
            page.locator('[data-action="close"]').click()
            page.locator('[data-action="undo"]').click()
            assert (
                page.locator('[data-bind="first"]')
                .first.locator("..")
                .get_attribute("class")
            )
            assert compilation()["errors"]
            page.locator('[data-action="close"]').click()
            page.locator('[data-action="redo"]').click()
            page.locator('[data-service="save"]').click()
            expect(page.locator("[data-save]")).to_contain_text("已真实保存")
            restored = context.request.get(
                args.url + PREFIX + "/projects/" + args.project_id
            ).json()
            assert restored["doc"]["shots"][0]["first"] == new
            assert old["id"] in {a["id"] for a in restored["assets"]}
            checks.append(
                "missing_asset_detected_explicit_same_kind_reconnect_new_identity_undo_redo_save_and_old_library_retained"
            )
            backup_path.rename(original_path)
            backup_path = None
            cards = page.locator("[data-thumbs] [data-tray-id]")
            simple.fill("前后")
            simple.evaluate(
                'e=>{e.focus();e.setSelectionRange(1,2);e.dispatchEvent(new Event("select",{bubbles:true}));}'
            )
            cards.first.locator("[data-insert]").click()
            assert simple.input_value() == "前@image1后"
            page.locator("[data-global-zone] summary").click()
            global_text = page.locator("[data-global]")
            global_text.fill("全片：结束")
            global_text.evaluate(
                'e=>{e.focus();e.setSelectionRange(3,3);e.dispatchEvent(new Event("select",{bubbles:true}));}'
            )
            second_id = cards.nth(1).get_attribute("data-tray-id")
            cards.nth(1).locator("[data-insert]").click()
            assert cards.first.get_attribute("data-tray-id") == second_id
            assert (
                global_text.input_value() == "全片：@image1结束"
                and simple.input_value() == "前@image2后"
            )
            page.locator('[data-action="undo"]').click()
            assert (
                simple.input_value() == "前@image1后"
                and page.locator("[data-global-media] article").count() == 0
            )
            global_text.locator("..").locator("[data-expand]").click()
            page.locator("[data-dialog-body] [data-insert]").nth(1).click()
            page.locator('[data-dialog-body] [data-action="close"]').click()
            assert page.locator("[data-global-media] article").count() == 0
            checks.append(
                "caret_insertion_never_replaces_selection_global_promotion_reindexes_identity_undo_modal_cancel_no_leak"
            )
            page.locator('[data-writing="advanced"]').click()
            page.locator("[data-events]").locator("..").locator("summary").click()
            page.locator('[data-action="event"]').click()
            page.locator('[data-key="text"]').fill("前段动作")
            page.locator('[data-key="end"]').fill("3")
            page.locator('[data-action="event"]').click()
            assert page.locator('[data-key="start"]').nth(1).input_value() == "3"
            assert page.locator('[data-field="duration"]').input_value() == "5"
            page.locator('[data-key="text"]').nth(1).fill("尾段闭嘴微笑，只有环境声")
            page.locator('[data-action="collapse-events"]').click()
            assert not page.locator("[data-events]").locator("..").evaluate("e=>e.open")
            page.locator('[data-action="duplicate"]').click()
            with page.expect_download() as download:
                page.locator('[data-service="project"]').click()
            download.value.save_as(args.output / "edge-project.json")
            current = json.loads(
                (args.output / "edge-project.json").read_text(encoding="utf-8")
            )
            first_events, second_events = [s["events"] for s in current["doc"]["shots"]]
            assert {e["id"] for e in first_events}.isdisjoint(
                e["id"] for e in second_events
            )
            page.locator('[data-field="ratio"]').select_option("1:1")
            page.locator("[data-shot]").first.click()
            assert page.locator('[data-field="ratio"]').input_value() == "1:1"
            page.locator("[data-shared-ratio]").uncheck()
            assert (
                page.locator('[data-field="ratio"]').input_value()
                == current["doc"]["shots"][0]["ownRatio"]
            )
            page.locator("[data-shared-ratio]").check()
            page.locator('[data-writing="simple"]').click()
            assert simple.input_value() == "前@image1后"
            checks.append(
                "new_event_previous_end_auto_duration_bottom_collapse_duplicate_fresh_UUIDs_shared_ratio_draft_restore"
            )
            with page.expect_file_chooser() as chooser:
                page.locator('[data-action="global-add"]').click()
            chooser.value.set_files(
                [
                    {
                        "name": "shared-first.png",
                        "mimeType": "image/png",
                        "buffer": picture,
                    },
                    {
                        "name": "shared-second.jpg",
                        "mimeType": "image/jpeg",
                        "buffer": Path("F:/gao3/10A.jpg").read_bytes(),
                    },
                ]
            )
            expect(page.locator("[data-global-media] article")).to_have_count(2)
            expect(cards.nth(0)).to_contain_text("shared-first.png")
            expect(cards.nth(1)).to_contain_text("shared-second.jpg")
            expect(cards.nth(0)).to_contain_text("@image1")
            expect(cards.nth(1)).to_contain_text("@image2")
            local_image = cards.filter(has=page.locator('[data-bind="first"]')).nth(2)
            local_next = cards.filter(has=page.locator('[data-bind="first"]')).nth(3)
            original_id = local_image.get_attribute("data-tray-id")
            before_prompt = simple.input_value()
            local_image.drag_to(local_next)
            assert (
                cards.filter(has=page.locator('[data-bind="first"]'))
                .nth(3)
                .get_attribute("data-tray-id")
                == original_id
            )
            assert simple.input_value() != before_prompt
            cards.nth(0).locator("[data-delete-asset]").click()
            expect(page.locator("[data-dialog-body]")).to_contain_text("所有镜头")
            page.locator("[data-confirm-delete]").click()
            assert page.locator("[data-global-media] article").count() == 1
            page.locator('[data-action="undo"]').click()
            assert page.locator("[data-global-media] article").count() == 2
            checks.append(
                "two_shared_images_prefix_upload_order_independent_aliases_real_drag_identity_remap_shared_delete_scope_undo"
            )
            page.screenshot(path=str(args.output / "edge-ui.png"), full_page=True)
            after_queue = context.request.get(args.url + "/queue").json()
            assert before_queue == after_queue and not queues and not errors
            report = {
                "status": "passed",
                "checks": checks,
                "page_errors": errors,
                "queue_posts": queues,
                "before_queue": before_queue,
                "after_queue": after_queue,
                "project_id": args.project_id,
                "scope": "CPU/browser owned isolated Core; no GPU, generation, publication or user queue changes",
            }
        except BaseException as error:
            page.screenshot(path=str(args.output / "failure.png"), full_page=True)
            report = {
                "status": "failed",
                "checks": checks,
                "error": repr(error),
                "page_errors": errors,
                "queue_posts": queues,
            }
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise
        finally:
            if backup_path and backup_path.exists():
                backup_path.rename(original_path)
            browser.close()
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
