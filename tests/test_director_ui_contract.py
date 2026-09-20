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
    ):
        assert marker in html, marker
    assert "data-d3-package" in session
    assert 'request("d3/package"' in session


def test_director_ui_is_local_and_does_not_embed_remote_runtime_or_media():
    html = (ROOT / "web" / "director" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert "data-picker" in html


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
