"""Unit tests for the voice/ subsystem without GPU or live model dependencies."""

import asyncio
import threading
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

import app
from voice import config, sentences, session, stt, text_clean, tts
from voice.phonon_engine import PhononEngine


# --- 1. Text Cleaner Tests ---

def test_text_cleaner_markdown_and_code():
    raw = (
        "# Title\n\n"
        "Here is a `console.log('hi')` command.\n\n"
        "```python\nprint('secret code')\n```\n"
        "And a link to [OpenAI](https://openai.com)."
    )
    cleaned = text_clean.clean_text_for_speech(raw)
    assert "Title" in cleaned
    assert "#" not in cleaned
    assert "console.log('hi')" in cleaned
    assert "`" not in cleaned
    assert "secret code" not in cleaned
    assert "https://openai.com" not in cleaned
    assert "OpenAI" in cleaned


def test_text_cleaner_symbols_and_emojis():
    raw = "Inflation reached 5% while profits dropped by $10! Good luck 🚀🎉."
    cleaned = text_clean.clean_text_for_speech(raw)
    assert "5 percent" in cleaned
    assert "10 dollars" in cleaned
    assert "🚀" not in cleaned
    assert "🎉" not in cleaned


def test_text_cleaner_think_block():
    raw = "<think>Let me think about how to answer this.</think>The capital of France is Paris."
    cleaned = text_clean.clean_text_for_speech(raw)
    assert "think" not in cleaned.lower()
    assert "The capital of France is Paris." in cleaned


def test_text_cleaner_tables():
    raw = (
        "| Name | Age |\n"
        "|---|---|\n"
        "| Alice | 30 |\n"
        "| Bob | 25 |"
    )
    cleaned = text_clean.clean_text_for_speech(raw)
    assert "|" not in cleaned
    assert "Alice" in cleaned
    assert "Bob" in cleaned


# --- 2. Sentence Splitter Tests ---

def test_sentence_splitter_empty():
    assert sentences.split_sentences("") == []
    assert sentences.split_sentences("   \n\t  ") == []


def test_sentence_splitter_batches_short_sentences():
    # Sentences under 25 characters should be batched together
    text = "Yes. Sure. I can do that for you right now."
    res = sentences.split_sentences(text, min_chars=25, max_chars=220)
    assert len(res) == 1
    assert "Yes. Sure. I can do that for you right now." in res[0]


def test_sentence_splitter_splits_oversized():
    # Long sentence over max_chars (50) split cleanly
    long_text = "This is a very long sentence designed to test the splitting algorithm, and ensure it cuts nicely at commas or spaces."
    res = sentences.split_sentences(long_text, min_chars=10, max_chars=50)
    assert len(res) > 1
    for s in res:
        assert len(s) <= 55  # Allows boundary padding


# --- 3. VoiceSession Lifecycle & Reset Tests ---

def test_voice_session_reset_wipes_transient_buffers():
    async def _async_test():
        sess = session.VoiceSession("test-client")
        sess.state = session.VState.SPEAKING
        sess.pcm.extend(b"\x00\x01" * 100)
        sess.partial_text = "partial transcript"
        await sess.tts_queue.put(("chunk", b"wavbytes"))

        async def dummy_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_task())
        sess.tasks.add(task)

        # Call reset
        sess.reset()
        await asyncio.sleep(0)

        assert sess.state == session.VState.IDLE
        assert len(sess.pcm) == 0
        assert sess.partial_text == ""
        assert sess.tts_queue.empty()
        assert len(sess.tasks) == 0
        assert task.cancelled() or task.cancelling() > 0
        assert not sess.cancel.is_set()  # Fresh token created

    asyncio.run(_async_test())


# --- 4. TTS Graceful Degradation & Engine Fallback Tests ---

def test_phonon_invalid_voice_uses_first_catalog(monkeypatch):
    class FakeTTS:
        voices = ["Easton", "Other"]

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(__import__("sys").modules, "ptts", MagicMock(TTS=FakeTTS))
    engine = PhononEngine("fake/model", "en", "q8", "Missing").load()
    assert engine.voice == "Easton"


def test_tts_falls_back_to_sapi_when_phonon_raises():
    phonon = MagicMock(ready=True)
    phonon.synth_pcm.side_effect = RuntimeError("phonon failure")
    sapi = MagicMock(available=True, device="windows-sapi", dtype="pcm", voice="")
    sapi.synthesize.return_value = np.ones(4, dtype=np.float32)
    service = tts.TTSService(phonon=phonon, sapi=sapi)
    assert service.synthesize("Hello") is not None
    sapi.synthesize.assert_called_once()


def test_tts_output_is_wav():
    audio = np.zeros(240, dtype=np.float32)
    wav = tts.audio_to_wav_bytes(audio)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_phonon_cancellation_stops_synthesis():
    engine = PhononEngine("fake/model", "en", "q8", "Easton")
    engine._tts = MagicMock()
    cancel = threading.Event()
    cancel.set()
    assert engine.synth_pcm("Do not synthesize", cancel=cancel) is None
    engine._tts.synth.assert_not_called()


# --- 5. Voice Router & Settings Round-Trip Tests ---

def test_voice_settings_round_trip(temp_db):
    client = TestClient(app.app)

    # 1. Fetch initial settings
    res = client.get("/api/voice/settings")
    assert res.status_code == 200
    data = res.json()
    assert "speak_enabled" in data
    assert "engine_ready" in data

    # 2. Update speak_enabled to True
    res_update = client.put("/api/voice/settings", json={"speak_enabled": True, "live_enabled": True})
    assert res_update.status_code == 200
    updated_data = res_update.json()
    assert updated_data["speak_enabled"] is True
    assert updated_data["live_enabled"] is True

    # 3. Read back
    res_get = client.get("/api/voice/settings")
    assert res_get.status_code == 200
    assert res_get.json()["speak_enabled"] is True


def test_voice_tts_endpoint_unavailable_engine():
    client = TestClient(app.app)
    # Ensure engine is marked unavailable
    fake_engine = tts.TTSService(sapi=MagicMock(available=False))
    fake_engine.load = MagicMock()
    tts.set_tts_engine(fake_engine)

    res = client.post("/api/voice/tts", json={"text": "Hello world"})
    assert res.status_code == 503
    assert "TTS engine is not available" in res.json()["error"]


def test_stt_whisper_fallback_and_silence():
    class FakeUnknownValueError(Exception):
        pass

    fake_sr = MagicMock()
    fake_sr.UnknownValueError = FakeUnknownValueError
    mock_rec = MagicMock()
    mock_rec.recognize_whisper.side_effect = ImportError("No module named 'whisper'")
    mock_rec.recognize_google.side_effect = FakeUnknownValueError("silence")
    fake_sr.Recognizer.return_value = mock_rec

    with patch.dict("sys.modules", {"speech_recognition": fake_sr}):
        adapter = stt.STT(engine="whisper_local")
        res = adapter.transcribe(b"\x00" * 3200)
        assert res == ""
