"""Voice processing package for S.P.I.D.E.Y.

Provides OmniVoice TTS synthesis, speech-to-text integration, audio streaming,
WebSocket live conversations, and Discord voice support.
"""

from . import config, text_clean, sentences, session, tts

__all__ = ["config", "text_clean", "sentences", "session", "tts"]

