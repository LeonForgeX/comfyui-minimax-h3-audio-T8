from types import SimpleNamespace

from tools.audit_native_voice_words_cpu import observe


def test_real_generator_is_consumed_without_target_hints():
    consumed, calls = [], []

    class Model:
        def transcribe(self, path, **options):
            calls.append((path, options))

            def generate():
                consumed.append(True)
                yield SimpleNamespace(text=' observed words ', start=0., end=1.,
                                      avg_logprob=-.2, no_speech_prob=.01)

            return generate(), SimpleNamespace(language='zh', language_probability=.98, duration=1.1)

    row = observe(Model(), 'owned.mp4', None)
    assert consumed == [True] and row['text'] == 'observed words'
    assert calls[0][1]['initial_prompt'] is None and calls[0][1]['prefix'] is None
    assert calls[0][1]['hotwords'] is None and calls[0][1]['language'] is None
    assert calls[0][1]['condition_on_previous_text'] is False
    assert row['segments'][0]['no_speech_prob'] == .01


def test_forced_language_is_explicit_not_expected_text():
    class Model:
        def transcribe(self, path, **options):
            assert options['language'] == 'zh'
            assert options['initial_prompt'] is None and options['hotwords'] is None
            return iter([]), SimpleNamespace(language='zh', language_probability=1., duration=3.)

    row = observe(Model(), 'owned.mp4', 'zh')
    assert row['requested_language'] == 'zh' and row['text'] == '' and row['segments'] == []
