from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_director_ui_keeps_the_frozen_beginner_and_advanced_controls():
    html = (ROOT / "web" / "director" / "index.html").read_text(encoding="utf-8")
    session = (ROOT / "web" / "director" / "session.mjs").read_text(encoding="utf-8")
    for marker in (
        "data-action=\"delete-shot\"",
        "data-delete-asset",
        "data-shared-ratio",
        "data-writing=\"simple\"",
        "data-writing=\"advanced\"",
        "data-audio-use=\"record\"",
        "data-action=\"global-add\"",
        "data-action=\"generate-all\"",
        "data-d3-inherit",
        "data-global-d3-toggle",
        "data-generation-field",
        "resolution_mp",
    ):
        assert marker in html, marker
    assert "data-d3-package" in session
    assert 'request("d3/package"' in session
    assert 'request("models")' in session
    assert 'generateAll' in session


def test_director_ui_is_local_and_does_not_embed_remote_runtime_or_media():
    html = (ROOT / "web" / "director" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert "data-picker" in html


def test_director_scope_toolbar_and_compact_preview_contract():
    html = (ROOT / "web/director/index.html").read_text(encoding="utf-8")
    session = (ROOT / "web/director/session.mjs").read_text(encoding="utf-8")
    assert 'first.closest(\'label\').insertAdjacentHTML(\'beforebegin\'' in html
    assert "el.disabled=false" in html
    assert "const d=effectiveD3(s);" in html
    assert "s.d3=clone(effectiveD3(s))" in html
    assert "doc.d3=clone(d3cfg(s))" in html
    assert 'data-model-zone open' not in html
    assert '<dialog data-model-zone' in html
    assert html.count('<div data-model-settings>') == 1
    assert 'data-action="model-settings" aria-haspopup="dialog"' in session
    assert 'data-local-zone open' in html
    assert 'class="o-media-workbench"' in html
    assert 'height:clamp(180px,25vh,260px);min-height:0;aspect-ratio:auto' in html
    assert 'data-action="toggle-preview"' in html
    assert 'root.dataset.previewCollapsed=String(hidden)' in html
    assert "shots:[makeShot('镜头 1','text')]" in html
    assert "doc.shots[2].sound='record'" not in html
    assert 'latest = await request("compile", { project: p, shot_id: ctx.current() })' in session
    assert '全部生成前检查 · 尚未提交任务' in session
    assert session.index('const report = await request("compile", { project: envelope() })') < session.index('for (const shot of shots)')


def test_director_d4_frontend_keeps_reconnect_and_large_asset_feedback_contract():
    session = (ROOT / "web" / "director" / "session.mjs").read_text(encoding="utf-8")
    host = (ROOT / "web" / "director.js").read_text(encoding="utf-8")
    assert 'xhr.upload.onprogress' in session
    assert '服务端未确认注册' in session
    assert 'window.addEventListener("offline"' in session
    assert '已有任务不会重复提交' in session
    assert 'xhr.onabort' in session
    assert 'button.dataset.service = "cancel-upload"' in session
    assert 't8director.activeJob:' in session
    assert 'setTimeout(() => watchJob' in session
    assert 'cancelledJobId' in session
    assert 'unknownPolls >= 20' in session
    assert '已连接当前 Core' in host
    assert '页面加载失败，返回画布后可重试' in host


def test_director_uses_a_dedicated_left_sidebar_without_covering_canvas_controls():
    host = (ROOT / "web" / "director.js").read_text(encoding="utf-8")
    assert "registerSidebarTab" in host
    assert 'id: "t8-obsidian-director"' in host
    assert 'title: "导演台"' in host
    assert 'type: "custom"' in host
    assert "renderDirectorSidebar" in host
    assert 'dataset.action = "open-t8-director"' in host
    assert 'id = "t8-director-open"' not in host
    assert "position:fixed;bottom:" not in host
