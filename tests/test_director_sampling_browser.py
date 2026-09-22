"""Actual Chromium interaction for the Director's transactional sampling dialog."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import os

from playwright.sync_api import sync_playwright
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_dialog_cancel_and_apply_do_not_change_legacy_until_committed():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "web" / "director")))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/models", lambda route: route.fulfill(json={
                "unet": [], "clip": [], "video_vae": [], "audio_vae": [],
                "lora": [{"value": "style.safetensors", "label": "style.safetensors"}],
                "upscaler": [{"value": "minimax_h3_latent_upscaler_3d_fp16.safetensors", "label": "3D upscaler"}],
            }))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            delivered = page.evaluate("""async () => { const module = await import('./session.mjs'); return module.directorOutputVideos({ outputs: { '12': { images: [{ filename: 'result.mp4' }, { filename: 'still.png' }] } } }); }""")
            assert delivered == [{"filename": "result.mp4"}]
            assert page.locator('.w-header-actions [data-action="model-settings"]').count(), (errors, page.locator("body").inner_text()[:1000])
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            page.locator('[data-sampling-mode="two_pass"]').click()
            assert page.locator(".o-sampling-columns .o-sampling-stage").count() == 2
            page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
            page.locator('[data-sampling-search="low_loras"]').fill("style")
            page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').select_option("style.safetensors")
            page.locator('[data-stage="low_loras"] [data-sampling-field="enabled"]').check()
            page.locator('[data-action="cancel-model-settings"]').click()
            cancelled = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return key ? JSON.parse(localStorage.getItem(key)) : null; }""")
            assert cancelled["doc"]["sampling"]["mode"] == "single"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-mode="single"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-sampling-mode="two_pass"]').click()
            page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
            page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').select_option("style.safetensors")
            page.locator('[data-stage="low_loras"] [data-sampling-field="enabled"]').check()
            page.locator('[data-sampling-action="copy-low"]').click()
            assert page.locator('[data-stage="high_loras"] [data-sampling-field="name"]').input_value() == "style.safetensors"
            page.locator('[data-action="apply-model-settings"]').click()
            applied = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            assert applied["version"] == 2 and applied["doc"]["sampling"]["mode"] == "two_pass"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-mode="two_pass"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').input_value() == "style.safetensors"
            page.locator('[data-sampling-scope="local"]').click()
            page.locator('[data-sampling-mode="single"]').click()
            page.locator('[data-action="apply-model-settings"]').click()
            independent = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            assert independent["doc"]["sampling"]["mode"] == "two_pass"
            assert independent["doc"]["shots"][0]["samplingInherit"] is False
            assert independent["doc"]["shots"][0]["sampling"]["mode"] == "single"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-scope="local"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-sampling-mode="single"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-sampling-scope="global"]').click()
            assert page.locator('[data-sampling-mode="two_pass"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-action="cancel-model-settings"]').click()
            project = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            shot_id = project["current"]
            page.route("**/results/*", lambda route: route.fulfill(json={"project_id": project["id"], "results": [{
                "shot_id": shot_id, "prompt_id": "e42bea1f-f961-4ef7-9c48-b181490b6f17", "state": "success",
                "recipe": "director_two_pass", "outputs": {"12": {"images": [{
                    "filename": "film.mp4", "subfolder": "T8_Director\\abc", "type": "output",
                }]}},
            }]}))
            page.evaluate("project => window.postMessage({ type: 't8-director:init', project }, location.origin)", project)
            page.wait_for_selector("video.o-output-player")
            assert page.locator('[data-view="output"]').get_attribute("aria-pressed") == "true"
            assert "film.mp4" in page.locator("video.o-output-player").get_attribute("src")
            assert "有成片" in page.locator('[data-shot]').first.inner_text()
            page.locator('[data-view="input"]').click()
            page.locator('[data-shot]').first.click()
            assert page.locator('[data-view="output"]').get_attribute("aria-pressed") == "true"
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_live_director_sampling_route_when_requested():
    url = os.getenv("T8_DIRECTOR_LIVE_URL")
    if not url:
        pytest.skip("Set T8_DIRECTOR_LIVE_URL for the running Core UI smoke")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.locator('.w-header-actions [data-action="model-settings"]').click()
        assert page.locator('[data-model-zone]').get_attribute("open") is not None
        page.locator('[data-sampling-mode="two_pass"]').click()
        assert page.locator(".o-sampling-columns .o-sampling-stage").count() == 2
        page.locator('[data-action="cancel-model-settings"]').click()
        assert not errors, errors
        browser.close()


def test_live_director_recovers_completed_video_when_requested():
    url = os.getenv("T8_DIRECTOR_LIVE_URL")
    if not url:
        pytest.skip("Set T8_DIRECTOR_LIVE_URL for the running Core UI smoke")
    import json
    from urllib.request import urlopen

    project_id = "2018d9a3-47ef-4af1-b4b4-769fb9ff58da"
    shot_id = "76c6edd1-6b51-4b78-bc5e-12a78f583022"
    with urlopen(url.replace("/ui", "/default"), timeout=20) as response:
        project = json.load(response)
    project["id"] = project_id
    project["current"] = shot_id
    project["doc"]["shots"][0]["id"] = shot_id
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.evaluate("project => window.postMessage({ type: 't8-director:init', project }, location.origin)", project)
        page.wait_for_selector("video.o-output-player", timeout=20000)
        assert "76c6edd1_00001_.mp4" in page.locator("video.o-output-player").get_attribute("src")
        assert "有成片" in page.locator("[data-shot]").first.inner_text()
        assert not errors, errors
        browser.close()
