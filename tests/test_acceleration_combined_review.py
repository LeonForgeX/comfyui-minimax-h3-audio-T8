import json

import pytest

from tools.build_acceleration_review import add_lipsync_questions, render_fi, verified_pair
from tools.build_progressive_exploration_review import render_sections


def test_fi_review_is_labeled_not_blind_and_uses_its_own_duration():
    html = render_fi({'id': 'pair-9', 'title': '<unsafe>', 'description': 'CPU合成',
        'duration': 2.0, 'source_fps': '30', 'target_fps': '60', 'has_audio': False})
    assert html.count('<video ') == 2 and html.count('<canvas ') == 2
    assert 'A：原片 30fps' in html and 'B：插帧 60fps' in html
    assert 'max="2.000000000"' in html and 'max="3.0416667"' not in html
    assert 'data-answer="audio"' not in html and '不评价音频' in html
    assert '<unsafe>' not in html and '&lt;unsafe&gt;' in html
    assert 'autoplay' not in html and '<iframe' not in html


def test_audio_source_keeps_audio_question():
    html = render_fi({'id': 'pair-8', 'title': 'game', 'description': 'original audio',
        'duration': 73/24, 'source_fps': '24', 'target_fps': '48', 'has_audio': True})
    assert 'data-answer="audio"' in html


def test_failed_run_cannot_enter_review(tmp_path):
    for name, data in [('terminal', {'status': 'failed'}), ('source', {}), ('postflight-v1', {})]:
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    with pytest.raises(ValueError, match='complete'):
        verified_pair(tmp_path, 'actual_boundary_complete')


def test_dialogue_question_does_not_invent_lipsync_review_for_silent_cases():
    html = render_sections([{'id': f'pair-{i}', 'title': 'fixture', 'description': '', 'accepted': False}
                            for i in range(3)])
    result = add_lipsync_questions(html, ('pair-2',))
    assert result.count('data-answer="lipsync"') == 1
    assert 'data-answer="lipsync"' not in result.split('<section id="pair-2"')[0]
    assert '看不清／无法判断' in result
    with pytest.raises(ValueError):
        add_lipsync_questions(result, ('pair-2',))
