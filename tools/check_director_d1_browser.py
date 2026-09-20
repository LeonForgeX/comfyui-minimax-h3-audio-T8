"""Real CPU Core/browser D1 checks. Never queues a prompt; keeps failure receipts."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import wave

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8852")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/director-d1-20260919/browser-v1",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, page_errors, queue_posts = [], [], []
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x01" * 32000)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100}, accept_downloads=True
        )
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        def prohibit(route):
            if route.request.method == "POST" and route.request.url.rstrip(
                "/"
            ).endswith("/prompt"):
                queue_posts.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", prohibit)
        try:
            page.goto(args.url)
            expect(page.locator("#t8-director-open")).to_be_visible(timeout=90000)
            # ComfyUI 0.36 can expose extension buttons before its startup
            # splash releases pointer events.  Visibility alone is therefore
            # not a usable UI gate; wait for the overlay to become inert.
            expect(page.locator("#splash-loader")).to_be_hidden(timeout=120000)
            # Fresh isolated Core shows its own template welcome dialog; dismiss it normally.
            page.keyboard.press("Escape")
            before_queue = context.request.get(args.url + "/queue").json()
            page.locator("#t8-director-open").click()
            f = page.frame_locator('iframe[title="曜石导演台"]')
            expect(f.locator('[data-field="simplePrompt"]')).to_be_visible()
            checks.append("real_Core355_frontend_native_entry_and_iframe_open")
            # Keep one shot, exercising real deletion confirmation and undo.
            f.locator("[data-shot]").nth(2).click()
            f.locator('[data-action="delete-shot"]').click()
            f.locator("[data-confirm-delete]").click()
            f.locator("[data-shot]").nth(1).click()
            f.locator('[data-action="delete-shot"]').click()
            f.locator("[data-confirm-delete]").click()
            f.locator('[data-action="undo"]').click()
            expect(f.locator("[data-shot]")).to_have_count(2)
            f.locator('[data-action="redo"]').click()
            expect(f.locator("[data-shot]")).to_have_count(1)
            f.locator('[data-mode="text"]').click()
            simple = f.locator('[data-field="simplePrompt"]')
            original = "夜晚江边。\n0–2秒她说你好。\n2–7秒微笑闭嘴，不再说话。"
            simple.fill(original)
            f.locator('[data-field="duration"]').fill("7")
            f.locator('[data-writing="advanced"]').click()
            assert f.locator('[data-field="prompt"]').input_value() == original
            f.locator('[data-writing="simple"]').click()
            assert simple.input_value() == original
            simple.locator("..").locator("[data-expand]").click()
            f.locator("[data-editor]").fill("cancel")
            f.locator('[data-dialog-body] [data-action="close"]').click()
            assert simple.input_value() == original
            checks.append(
                "v4_default_simple_advanced_drafts_large_editor_cancel_delete_undo"
            )
            f.locator("[data-global]").locator("xpath=ancestor::details").locator(
                "summary"
            ).click()
            f.locator("[data-global]").fill("全片光线稳定。")
            image = ROOT / "artifacts/director-plan-20260919/ui-test-v3.mp4"
            with page.expect_file_chooser() as chooser:
                f.locator('.o-stage [data-action="add"]').last.click()
            picture = Path("F:/ComfyUI_00110_iphsk_1789210737 (1).png").read_bytes()
            chooser.value.set_files(
                [
                    {"name": "same.png", "mimeType": "image/png", "buffer": picture},
                    {"name": "same.png", "mimeType": "image/png", "buffer": picture},
                    {
                        "name": "test.mp4",
                        "mimeType": "video/mp4",
                        "buffer": image.read_bytes(),
                    },
                    {
                        "name": "same.wav",
                        "mimeType": "audio/wav",
                        "buffer": buf.getvalue(),
                    },
                ]
            )
            expect(f.locator("[data-tray-id]")).to_have_count(4, timeout=30000)
            assert (
                len(
                    set(
                        f.locator("[data-tray-id]").evaluate_all(
                            "els=>els.map(e=>e.dataset.trayId)"
                        )
                    )
                )
                == 4
            )
            assert f.locator('[data-audio-use="voice"]').count() == 1
            f.locator('[data-bind="first"]').first.click()
            f.locator('[data-field="ratio"]').select_option("2:3")

            def compilation():
                f.locator('[data-action="check"]').click()
                expect(f.locator("[data-dialog]")).to_be_visible()
                expect(f.locator("[data-dialog-title]")).to_contain_text("真实编译预检")
                return json.loads(f.locator("[data-dialog] pre").inner_text())

            # These are real native UI/project compiler contracts, not D2 generation.
            cards = f.locator("[data-thumbs] [data-tray-id]")
            cards.nth(1).locator('[data-bind="last"]').click()
            simple.fill("@image1 是开始，@image2 是结束；衣着与背景连续。")
            pair = compilation()
            assert not pair["errors"] and pair["current_shot"]["task_type"] == "fl2va"
            assert [m["role"] for m in pair["current_shot"]["media_map"]] == [
                "first_frame",
                "last_frame",
            ]
            f.locator('[data-action="close"]').click()
            f.locator('[data-action="pair"]').click()
            expect(f.locator("[data-stage] .o-pairview img")).to_have_count(2)
            cards.nth(1).locator('[data-bind="refs"]').click()
            hybrid = compilation()
            assert (
                not hybrid["errors"] and hybrid["current_shot"]["task_type"] == "hybrid"
            )
            assert [m["role"] for m in hybrid["current_shot"]["media_map"]] == [
                "first_frame",
                "ref_image",
            ]
            f.locator('[data-action="close"]').click()
            f.locator('[data-mode="refs"]').click()
            expect(f.locator("[data-ref-state]")).to_contain_text("1 项")
            simple.fill("@image2 是人物参考，背景与服装保持一致。")
            ref = compilation()
            assert not ref["errors"] and ref["current_shot"]["task_type"] == "ref2va"
            assert "<Picture 1>" in ref["current_shot"]["prompt"]
            assert ref["current_shot"]["aliases"][1]["native"] == "<Picture 1>"
            f.locator('[data-action="close"]').click()
            second_id = cards.nth(1).get_attribute("data-tray-id")
            cards.nth(1).locator(".o-thumb").click()
            expect(f.locator("[data-stage] img")).to_have_attribute(
                "src",
                re.compile(
                    r"/minimax_h3_t8/director/assets/" + re.escape(second_id) + "$"
                ),
            )
            # Restore the accepted single-first-frame audio recipe explicitly.
            cards.nth(1).locator('[data-bind="first"]').click()
            cards.nth(0).locator('[data-bind="first"]').click()
            f.locator('[data-audio-use="voice"]').click()
            simple.fill("@image1 人物说新的台词，结束后闭嘴微笑。@audio1")
            compiled = compilation()
            assert compiled["errors"] == [], compiled["errors"]
            current = compiled["current_shot"]
            assert current["drive_audio"] is None and current["final_audio"] is None
            assert (
                current["delivery_audio"] == "generated"
                and "<Audio 1>" in current["prompt"]
            )
            assert current["prompt"].count("全片光线稳定") == 1
            f.locator('[data-action="close"]').click()
            checks.append(
                "actual_mixed_upload_UUID_first_last_hybrid_ref_preview_aliases_and_native_voice_no_source_delivery"
            )
            f.locator('[data-service="save"]').click()
            expect(f.locator("[data-save]")).to_contain_text(
                "已真实保存", timeout=30000
            )
            with page.expect_download() as dl:
                f.locator('[data-service="project"]').click()
            dl.value.save_as(args.output / "project.json")
            project = json.loads(
                (args.output / "project.json").read_text(encoding="utf-8")
            )
            assert project["revision"] == 1 and all(
                "blob:" not in json.dumps(a) for a in project["assets"]
            )
            remote_asset = context.request.get(
                args.url
                + "/minimax_h3_t8/director/assets/"
                + project["assets"][0]["id"]
            )
            assert (
                remote_asset.ok
                and len(remote_asset.body()) == project["assets"][0]["size"]
            )
            checks.append("real_server_CAS_save_remote_asset_bytes_and_project_export")
            # Separate page has no File/Blob objects; imports use only server UUIDs.
            other = context.new_page()
            other.goto(args.url + "/minimax_h3_t8/director/ui")
            expect(other.locator('[data-field="simplePrompt"]')).to_have_value(
                simple.input_value()
            )
            expect(other.locator("[data-tray-id]")).to_have_count(4)
            other.locator('[data-field="simplePrompt"]').fill("另一标签编辑")
            f.locator('[data-service="save"]').click()
            expect(f.locator("[data-save]")).to_contain_text("版本 2")
            other.locator('[data-service="save"]').click()
            expect(other.locator("[data-notice]")).to_contain_text("另一标签已保存")
            assert (
                other.locator('[data-field="simplePrompt"]').input_value()
                == "另一标签编辑"
            )
            other.locator('[data-service="copy"]').click()
            expect(other.locator("[data-save]")).to_contain_text("已真实保存")
            other.close()
            checks.append("cross_tab_conflict_keeps_draft_and_save_as_copy")
            # Native export opens and round-trips in actual Core, without /prompt submission.
            with page.expect_download() as native:
                f.locator('[data-service="workflow"]').click()
            native.value.save_as(args.output / "preflight.workflow.json")
            with page.expect_download() as api:
                f.locator('[data-service="api"]').click()
            api.value.save_as(args.output / "preflight.api.json")
            workflow = json.loads(
                (args.output / "preflight.workflow.json").read_text(encoding="utf-8")
            )
            snapshot = json.loads(
                (args.output / "preflight.api.json").read_text(encoding="utf-8")
            )
            frame = page.frames[1]
            frame.locator('[data-field="simplePrompt"]').fill("刷新保护")
            frame.evaluate("location.reload()")
            expect(
                page.frame_locator('iframe[title="曜石导演台"]').locator(
                    '[data-field="simplePrompt"]'
                )
            ).to_have_value("刷新保护")
            page.evaluate(
                """async workflow => { const {app}=await import('/scripts/app.js'); await app.loadGraphData(workflow); }""",
                workflow,
            )
            roundtrip = page.evaluate(
                """async()=>{const {app}=await import('/scripts/app.js');return await app.graphToPrompt();}"""
            )
            assert roundtrip["output"]["1"]["inputs"] == snapshot["1"]["inputs"], (
                roundtrip
            )
            page.get_by_role("button", name="返回画布", exact=True).click()
            with page.expect_response(
                lambda r: r.request.method in {"POST", "PUT"}
                and "userdata" in r.url
                and "workflows" in r.url
            ) as saved_native:
                page.keyboard.press("Control+s")
                # Every run owns a new project/file: avoid a real overwrite-confirmation
                # dialog from an earlier QA run, rather than overwriting its evidence.
                page.locator("input:visible").last.fill(
                    "director-d1-native-qa-" + project["id"]
                )
                page.get_by_role("button", name="确认", exact=True).click()
            assert saved_native.value.ok
            native_response = context.request.get(saved_native.value.url)
            assert native_response.ok
            native_saved = native_response.json()
            (args.output / "native-saved.workflow.json").write_text(
                json.dumps(native_saved, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            page.evaluate(
                """async w=>{const {app}=await import('/scripts/app.js');await app.loadGraphData(w);}""",
                native_saved,
            )
            roundtrip2 = page.evaluate(
                """async()=>{const {app}=await import('/scripts/app.js');return await app.graphToPrompt();}"""
            )
            assert roundtrip2["output"]["1"]["inputs"] == snapshot["1"]["inputs"]
            checks.append(
                "native_workflow_load_CtrlS_reload_API_exact_and_refresh_draft_protection"
            )
            page.locator("#t8-director-open").click()
            expect(f.locator("[data-stage] img")).to_have_js_property(
                "naturalWidth", 1024
            )
            expect(f.locator("[data-stage] img")).to_have_js_property("complete", True)
            page.frames[1].evaluate(
                "async()=>{await document.querySelector('[data-stage] img').decode();}"
            )
            page.screenshot(path=str(args.output / "native-ui.png"))
            # Narrow viewport true UI layout, not fake DOM.
            for width in (1440, 1060, 750, 390, 320):
                page.set_viewport_size({"width": width, "height": 1000})
                overflow = page.frames[1].evaluate(
                    "document.documentElement.scrollWidth > innerWidth + 1"
                )
                assert not overflow, width
            checks.append("five_width_layout_no_horizontal_overflow")
            after_queue = context.request.get(args.url + "/queue").json()
            assert before_queue == after_queue and not queue_posts
            assert not page_errors, page_errors
            report = {
                "status": "passed",
                "checks": checks,
                "page_errors": page_errors,
                "queue_posts": queue_posts,
                "before_queue": before_queue,
                "after_queue": after_queue,
                "project_id": project["id"],
                "scope": "CPU D1 actual Core frontend; no GPU, no generation, no publication",
            }
        except BaseException as e:
            page.screenshot(path=str(args.output / "failure.png"))
            report = {
                "status": "failed",
                "checks": checks,
                "error": repr(e),
                "page_errors": page_errors,
                "queue_posts": queue_posts,
            }
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise
        finally:
            browser.close()
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
