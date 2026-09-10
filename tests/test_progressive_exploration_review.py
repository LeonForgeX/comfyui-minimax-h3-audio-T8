from pathlib import Path
import re
import shutil
import subprocess

import pytest

from tools.build_progressive_exploration_review import render_sections


def item(key="pair-0", accepted=False):
    return {"id": key, "title": '<script>alert("x")</script>', "description": "<img src=x>", "accepted": accepted}


def test_inline_pair_has_both_original_players_and_crops_and_no_autoplay():
    html = render_sections([item()])
    assert html.count("<video ") == 2 and html.count("<canvas ") == 2
    assert 'src="pair-0-A.mp4"' in html and 'src="pair-0-B.mp4"' in html
    assert 'autoplay' not in html and '<iframe' not in html
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert '<img' not in html


@pytest.mark.parametrize("items", ([], [item(), item()], [item('../x')], [item('pair-a')]))
def test_unsafe_or_duplicate_section_identifiers_rejected(items):
    with pytest.raises(ValueError):
        render_sections(items)


def test_accepted_reuse_has_no_new_review_questions():
    html = render_sections([item(accepted=True)])
    assert 'data-review="false"' in html and 'data-answer=' not in html
    assert '不要求重审' in html


def test_template_javascript_parses_without_browser_or_frontend_actions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js syntax checker is unavailable")
    path = Path(__file__).resolve().parents[1] / "tools/progressive_review_hub_template.html"
    html = path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    result = subprocess.run([node, '--check'], input=scripts[0], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert html.count('REVIEW_ID_PLACEHOLDER') == 1
    assert "review_id:document.querySelector" in scripts[0]
