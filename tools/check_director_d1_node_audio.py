"""Actual native node snapshot handshake and real audio controls; no /prompt."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8852")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    snapshot = json.loads(workflow["nodes"][0]["widgets_values"][0])
    selected = next(
        s for s in snapshot["doc"]["shots"] if s["id"] == snapshot["current"]
    )
    checks, errors, queues = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        context.add_init_script(
            "Object.defineProperty(Crypto.prototype,'randomUUID',{value:undefined,configurable:true})"
        )
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda e: errors.append(str(e)))

        def prohibit(route):
            if route.request.method == "POST" and route.request.url.rstrip(
                "/"
            ).endswith("/prompt"):
                queues.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", prohibit)
        try:
            before = context.request.get(args.url + "/queue").json()
            page.goto(args.url)
            expect(page.locator("#t8-director-open")).to_be_visible(timeout=90000)
            page.keyboard.press("Escape")
            page.evaluate(
                """async w=>{const {app}=await import('/scripts/app.js');await app.loadGraphData(w);
                const node=app.graph.getNodeById(1);node.widgets.find(w=>w.name==='打开曜石导演台').callback();}""",
                workflow,
            )
            f = page.frame_locator('iframe[title="曜石导演台"]')
            expect(f.locator('[data-field="simplePrompt"]')).to_have_value(
                selected["simplePrompt"]
            )
            expect(f.locator("[data-notice]")).to_contain_text("当前节点项目快照")
            checks.append(
                "actual_native_node_button_exact_widget_snapshot_with_randomUUID_unavailable_portable_UUIDs"
            )
            f.locator('[data-audio-use="record"]').click()
            expect(f.locator('[data-field="sound"]')).to_have_value("record")
            expect(f.locator("[data-wave]")).to_be_visible()
            f.locator('input[type="number"][data-field="start"]').fill("0.5")
            f.locator('input[type="number"][data-field="end"]').fill("1.5")
            f.locator('[data-action="listen"]').click()
            frame = page.frames[1]
            frame.wait_for_function(
                "document.querySelector('[data-audio]').paused && document.querySelector('[data-audio]').currentTime >= 1.5",
                timeout=5000,
            )
            assert f.locator('[data-field="duration"]').input_value() == "1"

            def compilation():
                f.locator('[data-action="check"]').click()
                expect(f.locator("[data-dialog]")).to_be_visible()
                expect(f.locator("[data-dialog-title]")).to_contain_text("真实编译预检")
                return json.loads(f.locator("[data-dialog] pre").inner_text())

            compiled = compilation()
            assert not compiled["errors"], compiled["errors"]
            out = compiled["current_shot"]
            assert out["recipe"] == "avatar_single_segment_progressive_lock_source"
            assert out["drive_audio"] == out["final_audio"] == selected["audio"]
            assert out["audio_selection"] == {
                "asset_id": selected["audio"],
                "start": 0.5,
                "end": 1.5,
            }
            assert out["delivery_audio"] == "original_selected_recording"
            f.locator('[data-action="close"]').click()
            f.locator('[data-action="whole"]').click()
            assert f.locator('[data-field="duration"]').input_value() == "2"
            f.locator('[data-audio-use="voice"]').click()
            out = compilation()["current_shot"]
            assert (
                out["recipe"] == "native_ref_voice_stock20"
                and out["drive_audio"] is None
                and out["final_audio"] is None
            )
            assert (
                out["delivery_audio"] == "generated"
                and out["media_map"][-1]["role"] == "ref_audio"
            )
            f.locator('[data-action="close"]').click()
            checks.append(
                "real_server_WAV_waveform_selection_playback_stops_at_end_whole_audio_and_distinct_compiled_drive_voice_delivery"
            )
            page.get_by_role("button", name="返回画布", exact=True).click()
            # Opening again from the same native node replays the original saved snapshot.
            page.evaluate(
                """async()=>{const {app}=await import('/scripts/app.js');app.graph.getNodeById(1).widgets.find(w=>w.name==='打开曜石导演台').callback();}"""
            )
            expect(f.locator('[data-field="sound"]')).to_have_value("voice")
            expect(f.locator('[data-field="simplePrompt"]')).to_have_value(
                selected["simplePrompt"]
            )
            checks.append(
                "close_returns_to_canvas_without_cancel_or_mutating_node_unsaved_snapshot_reopen_restores"
            )
            expect(f.locator("[data-stage] img")).to_have_js_property(
                "naturalWidth", 1024
            )
            expect(f.locator("[data-stage] img")).to_have_js_property("complete", True)
            page.frames[1].evaluate(
                "async()=>{await document.querySelector('[data-stage] img').decode();}"
            )
            page.screenshot(path=str(args.output / "node-audio-ui.png"))
            after = context.request.get(args.url + "/queue").json()
            assert before == after and not queues and not errors
            report = {
                "status": "passed",
                "checks": checks,
                "page_errors": errors,
                "queue_posts": queues,
                "before_queue": before,
                "after_queue": after,
                "project_id": snapshot["id"],
                "scope": "real CPU Core native node UI/audio playback, no generation/GPU/publication",
            }
        except BaseException as e:
            page.screenshot(path=str(args.output / "failure.png"))
            report = {
                "status": "failed",
                "checks": checks,
                "error": repr(e),
                "page_errors": errors,
                "queue_posts": queues,
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
