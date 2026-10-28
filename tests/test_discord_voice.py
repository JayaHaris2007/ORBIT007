"""Tests for Discord voice bridge, PCM conversion, QueuePCMSource, and call controls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from voice import config
from voice.discord_voice import (
    FRAME_BYTES,
    DiscordVoiceManager,
    QueuePCMSource,
    to_discord_pcm,
)


def test_to_discord_pcm_conversion():
    # 1 second of 24 kHz sine wave
    sample_rate = 24000
    t = np.linspace(0, 1, sample_rate, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    pcm_bytes = to_discord_pcm(sine)

    # 48 kHz stereo 16-bit = 48000 * 2 channels * 2 bytes = 192,000 bytes
    assert len(pcm_bytes) == 48000 * 2 * 2

    # Check for valid 16-bit PCM samples (no out of bounds)
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    assert len(samples) == 96000
    assert samples.max() <= 32767
    assert samples.min() >= -32768


def test_queue_pcm_source():
    source = QueuePCMSource()

    # Empty queue returns silence (3840 bytes of zeros)
    silence = source.read()
    assert len(silence) == FRAME_BYTES
    assert silence == b"\x00" * FRAME_BYTES

    # Feed exactly 2 frames
    payload = b"\x01\x02" * (FRAME_BYTES // 2) * 2
    source.feed(payload)

    frame1 = source.read()
    assert len(frame1) == FRAME_BYTES
    assert frame1 == payload[:FRAME_BYTES]

    frame2 = source.read()
    assert len(frame2) == FRAME_BYTES
    assert frame2 == payload[FRAME_BYTES:]

    # Now empty again -> returns silence
    assert source.read() == b"\x00" * FRAME_BYTES

    # Test clear
    source.feed(payload)
    source.clear()
    assert source.read() == b"\x00" * FRAME_BYTES


def test_discord_voice_allowed_user_filter(monkeypatch):
    mgr = DiscordVoiceManager()
    monkeypatch.setattr(config, "DISCORD_VOICE_ALLOWED_USER_IDS", {12345, 67890})

    assert mgr.is_user_allowed(12345) is True
    assert mgr.is_user_allowed(67890) is True
    assert mgr.is_user_allowed(99999) is False

    # Empty allowlist allows all users
    monkeypatch.setattr(config, "DISCORD_VOICE_ALLOWED_USER_IDS", set())
    assert mgr.is_user_allowed(99999) is True


def test_discord_voice_manager_lifecycle():
    async def _test():
        mgr = DiscordVoiceManager()

        # Create mock message with mock voice channel
        mock_channel = MagicMock()
        mock_vc = MagicMock()
        mock_vc.is_connected = MagicMock(return_value=True)
        mock_vc.disconnect = AsyncMock()
        mock_vc.play = MagicMock()
        mock_channel.connect = AsyncMock(return_value=mock_vc)
        mock_channel.name = "General Voice"
        mock_channel.id = 101

        mock_author = MagicMock()
        mock_author.voice.channel = mock_channel

        mock_msg = MagicMock()
        mock_msg.guild.id = 999
        mock_msg.author = mock_author
        mock_msg.channel.id = 202

        # 1. Join call
        success, msg = await mgr.join_call(mock_msg)
        assert success is True
        assert "General Voice" in msg
        assert mgr.is_call_active(999) is True
        mock_channel.connect.assert_called_once()
        mock_vc.play.assert_called_once()

        # 2. Stop speaking
        stop_msg = mgr.stop_speaking(999)
        assert stop_msg == "Stopped speaking."

        # 3. Voice toggle
        off_msg = mgr.set_voice_enabled(999, False)
        assert "disabled" in off_msg
        call = mgr.get_call(999)
        assert call.voice_enabled is False

        on_msg = mgr.set_voice_enabled(999, True)
        assert "enabled" in on_msg
        assert call.voice_enabled is True

        # 4. Leave call
        leave_msg = await mgr.leave_call(999)
        assert leave_msg == "Left voice channel."
        assert mgr.is_call_active(999) is False
        mock_vc.disconnect.assert_called_once()

    asyncio.run(_test())


def test_discord_input_command_and_normal_message_in_call():
    async def _test():
        import discord_bot

        # 1. Test input_command (!input <prompt>)
        mock_ctx = MagicMock()
        mock_ctx.message = MagicMock()
        with patch.object(
            discord_bot.bot, "respond_to_prompt", new_callable=AsyncMock
        ) as mock_respond:
            await discord_bot.input_command(mock_ctx, prompt="Hello world")
            mock_respond.assert_called_once_with(mock_ctx.message, "Hello world")

        # 2. Test normal message in call (no prefix)
        mock_msg = MagicMock()
        mock_msg.author.bot = False
        mock_msg.content = "Normal conversation message"
        mock_msg.guild.id = 777
        mock_msg.channel.id = 555
        mock_msg.mentions = []

        mock_mgr = MagicMock()
        mock_call = MagicMock()
        mock_call.text_channel_id = 555
        mock_mgr.get_call.return_value = mock_call
        mock_mgr.is_call_active.return_value = True

        with patch.object(
            discord_bot, "get_discord_voice_manager", return_value=mock_mgr
        ):
            with patch.object(
                discord_bot.bot, "respond_to_prompt", new_callable=AsyncMock
            ) as mock_respond:
                await discord_bot.on_message(mock_msg)
                mock_respond.assert_called_once_with(
                    mock_msg, "Normal conversation message"
                )

    asyncio.run(_test())
