"""Tests for Discord voice bridge, PCM conversion, QueuePCMSource, and call controls."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from voice import config
from voice.discord_voice import (
    FRAME_BYTES,
    DiscordVoiceManager,
    QueuePCMSource,
    to_discord_pcm,
    from_discord_pcm,
    _calculate_discord_pcm_energy,
    DiscordVoiceAudioSink,
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


def test_from_discord_pcm_conversion():
    # Empty bytes test
    assert from_discord_pcm(b"") == b""

    # 1 second of 48 kHz stereo sine wave: 48000 samples * 2 channels * 2 bytes = 192000 bytes
    t = np.linspace(0, 1, 48000, dtype=np.float32)
    sine = (np.sin(2 * np.pi * 440 * t) * 30000).astype("<i2")
    stereo = np.column_stack([sine, sine]).tobytes()
    assert len(stereo) == 192000

    pcm16k = from_discord_pcm(stereo)
    # 16000 samples * 1 channel * 2 bytes = 32000 bytes
    assert len(pcm16k) == 32000
    samples = np.frombuffer(pcm16k, dtype="<i2")
    assert samples.max() <= 32767
    assert samples.min() >= -32768


def test_calculate_discord_pcm_energy():
    assert _calculate_discord_pcm_energy(b"") == 0.0
    silence = b"\x00" * 3840
    assert _calculate_discord_pcm_energy(silence) == 0.0

    # Loud sine wave
    t = np.linspace(0, 0.02, 960, dtype=np.float32)
    sine = (np.sin(2 * np.pi * 440 * t) * 10000).astype("<i2")
    stereo = np.column_stack([sine, sine]).tobytes()
    energy = _calculate_discord_pcm_energy(stereo)
    assert energy > 5000.0


def test_discord_voice_audio_sink_vad_and_barge_in():
    from voice.discord_voice import DiscordVoiceCall, DiscordVoiceAudioSink, _VOICE_RECV_AVAILABLE
    if not _VOICE_RECV_AVAILABLE:
        pytest.skip("voice_recv not available")

    call = MagicMock(spec=DiscordVoiceCall)
    call.is_speaking = True
    cancel_mock = MagicMock()
    call.cancel = cancel_mock
    call.source = MagicMock()
    bot = MagicMock()
    mgr = MagicMock()
    mgr.is_user_allowed.return_value = True

    sink = DiscordVoiceAudioSink(call, bot, asyncio.new_event_loop(), mgr)

    # 1. Ignored if user is a bot
    bot_user = MagicMock()
    bot_user.bot = True
    sink.write(bot_user, MagicMock(pcm=b"\x10\x10" * 960))
    assert len(sink.buffers) == 0

    # 2. Ignored if user not allowed
    mgr.is_user_allowed.return_value = False
    human = MagicMock()
    human.bot = False
    human.id = 1234
    human.display_name = "Alice"
    sink.write(human, MagicMock(pcm=b"\x10\x10" * 960))
    assert len(sink.buffers) == 0

    # 3. Allowed human with loud audio triggers barge-in
    mgr.is_user_allowed.return_value = True
    t = np.linspace(0, 0.02, 960, dtype=np.float32)
    loud = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    loud_pcm = np.column_stack([loud, loud]).tobytes()

    sink.write(human, MagicMock(pcm=loud_pcm))
    cancel_mock.set.assert_called_once()
    call.source.clear.assert_called_once()
    assert 1234 in sink.buffers
    assert sink.buffers[1234]["speech_detected"] is True


def test_respond_to_voice_utterance():
    async def _test():
        import discord
        import discord_bot
        mock_user = MagicMock(spec=discord.Member)
        mock_user.id = 42
        mock_user.display_name = "Bob"

        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()

        with patch.object(discord_bot.bot, "get_channel", return_value=mock_channel):
            with patch("app._handle_local_command", return_value=("The time is 5:00 PM.", "local_clock")):
                with patch.object(discord_bot, "get_discord_voice_manager") as mock_get_mgr:
                    mock_mgr = MagicMock()
                    mock_mgr.is_call_active.return_value = True
                    mock_mgr.speak_text = AsyncMock()
                    mock_get_mgr.return_value = mock_mgr

                    await discord_bot.bot.respond_to_voice_utterance(
                        guild_id=101,
                        user=mock_user,
                        text_channel_id=202,
                        prompt="what time is it",
                    )

                    # Should send transcript line
                    mock_channel.send.assert_any_call("🎙️ **Bob**: what time is it")
                    # Should send response line
                    mock_channel.send.assert_any_call("The time is 5:00 PM.")

    asyncio.run(_test())



def test_reminder_delivery_target_storage_in_memory_store(tmp_path):
    from pathlib import Path
    from memory import MemoryStore
    db_file = tmp_path / "test_reminders.db"
    store = MemoryStore(db_file)

    # Test initial retrieval
    assert store.get_reminder_delivery_target("rem-1") is None

    # Test save
    store.save_reminder_delivery_target("rem-1", channel_id=123, guild_id=456, user_id=789)
    target = store.get_reminder_delivery_target("rem-1")
    assert target is not None
    assert target["channel_id"] == 123
    assert target["guild_id"] == 456
    assert target["user_id"] == 789

    # Test upsert
    store.save_reminder_delivery_target("rem-1", channel_id=999, guild_id=456, user_id=789)
    target2 = store.get_reminder_delivery_target("rem-1")
    assert target2["channel_id"] == 999


def test_server_channel_reminder_delivery_routes_to_channel():
    async def _test():
        import discord
        import discord_bot

        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()

        bot = discord_bot.bot
        bot.send_dm = AsyncMock()

        # Mock notification
        mock_notification = {
            "id": 10,
            "chat_id": "test-chat-1",
            "reminder_id": "rem-server-1",
            "content": "Just a nudge: sleep. It's 10:08 PM right now.",
        }

        # Mock target pointing to guild channel
        mock_target = {
            "reminder_id": "rem-server-1",
            "channel_id": 555,
            "guild_id": 777,
            "user_id": 9999,
        }

        with patch.object(bot, "get_channel", return_value=mock_channel):
            with patch("app.memory_store.get_reminder_delivery_target", return_value=mock_target):
                with patch("app.memory_store.list_notifications", return_value=[mock_notification]):
                    with patch("app.memory_store.add_message"):
                        saved_chats = dict(bot.discord_chats)
                        saved_ids = dict(bot.last_notification_id)
                        try:
                            bot.discord_chats.clear()
                            bot.last_notification_id.clear()
                            bot.discord_chats["test-chat-1"] = "9999"
                            bot.last_notification_id["test-chat-1"] = 0

                            # Run deliver_notifications for one iteration
                            with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
                                with patch.object(bot, "wait_until_ready", new_callable=AsyncMock):
                                    try:
                                        await bot.deliver_notifications()
                                    except asyncio.CancelledError:
                                        pass
                        finally:
                            bot.discord_chats = saved_chats
                            bot.last_notification_id = saved_ids

                        # Server channel MUST receive the message mentioning the user
                        mock_channel.send.assert_called_once_with("<@9999> Just a nudge: sleep. It's 10:08 PM right now.")
                        # send_dm MUST NOT have been called
                        bot.send_dm.assert_not_called()

    asyncio.run(_test())


def test_dm_reminder_delivery_routes_to_dm():
    async def _test():
        import discord_bot

        bot = discord_bot.bot
        bot.send_dm = AsyncMock()

        mock_notification = {
            "id": 11,
            "chat_id": "test-chat-2",
            "reminder_id": "rem-dm-1",
            "content": "Just a nudge: dinner. It's 9:36 PM right now.",
        }

        # Target has no guild/channel (DM reminder)
        mock_target = {
            "reminder_id": "rem-dm-1",
            "channel_id": None,
            "guild_id": None,
            "user_id": 8888,
        }

        with patch("app.memory_store.get_reminder_delivery_target", return_value=mock_target):
            with patch("app.memory_store.list_notifications", return_value=[mock_notification]):
                with patch("app.memory_store.add_message"):
                    saved_chats = dict(bot.discord_chats)
                    saved_ids = dict(bot.last_notification_id)
                    try:
                        bot.discord_chats.clear()
                        bot.last_notification_id.clear()
                        bot.discord_chats["test-chat-2"] = "8888"
                        bot.last_notification_id["test-chat-2"] = 0

                        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
                            with patch.object(bot, "wait_until_ready", new_callable=AsyncMock):
                                try:
                                    await bot.deliver_notifications()
                                except asyncio.CancelledError:
                                    pass
                    finally:
                        bot.discord_chats = saved_chats
                        bot.last_notification_id = saved_ids

                    # send_dm MUST be called for DM reminder
                    bot.send_dm.assert_called_once_with(8888, "Just a nudge: dinner. It's 9:36 PM right now.")

    asyncio.run(_test())


def test_server_reminder_auto_joins_voice_when_user_on_call():
    async def _test():
        import discord
        import discord_bot
        from voice.discord_voice import get_discord_voice_manager

        bot = discord_bot.bot
        mock_text_channel = AsyncMock(spec=discord.TextChannel)
        mock_text_channel.send = AsyncMock()

        mock_user_vc = MagicMock(spec=discord.VoiceChannel)
        mock_user_vc.id = 888
        mock_user_vc.name = "Study Lounge"

        mock_member = MagicMock(spec=discord.Member)
        mock_member.id = 9999
        mock_member.voice = MagicMock(channel=mock_user_vc)

        mock_guild = MagicMock(spec=discord.Guild)
        mock_guild.id = 777
        mock_guild.get_member.return_value = mock_member

        mock_notification = {
            "id": 20,
            "chat_id": "test-chat-voice-1",
            "reminder_id": "rem-vc-1",
            "content": "Just a nudge: stretch. It's 11:00 AM right now.",
        }

        mock_target = {
            "reminder_id": "rem-vc-1",
            "channel_id": 555,
            "guild_id": 777,
            "user_id": 9999,
        }

        voice_mgr = get_discord_voice_manager()
        with patch.object(bot, "get_channel", return_value=mock_text_channel):
            with patch.object(bot, "get_guild", return_value=mock_guild):
                with patch("app.memory_store.get_reminder_delivery_target", return_value=mock_target):
                    with patch("app.memory_store.list_notifications", return_value=[mock_notification]):
                        with patch("app.memory_store.add_message"):
                            with patch.object(voice_mgr, "join_voice_channel", new_callable=AsyncMock) as mock_join:
                                with patch.object(voice_mgr, "speak_text", new_callable=AsyncMock) as mock_speak:
                                    saved_chats = dict(bot.discord_chats)
                                    saved_ids = dict(bot.last_notification_id)
                                    try:
                                        bot.discord_chats.clear()
                                        bot.last_notification_id.clear()
                                        bot.discord_chats["test-chat-voice-1"] = "9999"
                                        bot.last_notification_id["test-chat-voice-1"] = 0

                                        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
                                            with patch.object(bot, "wait_until_ready", new_callable=AsyncMock):
                                                try:
                                                    await bot.deliver_notifications()
                                                except asyncio.CancelledError:
                                                    pass
                                    finally:
                                        bot.discord_chats = saved_chats
                                        bot.last_notification_id = saved_ids

                                    # S.P.I.D.E.Y should auto-join the user's voice channel
                                    mock_join.assert_called_once_with(mock_user_vc, 555, guild_id=777)
                                    # S.P.I.D.E.Y should speak the reminder in voice
                                    mock_speak.assert_called_once_with(777, 9999, "Just a nudge: stretch. It's 11:00 AM right now.")
                                    # Text channel gets notification noting voice delivery
                                    mock_text_channel.send.assert_called_once_with(
                                        "<@9999> Just a nudge: stretch. It's 11:00 AM right now. *(Spoken in 🔊 Study Lounge)*"
                                    )

    asyncio.run(_test())


def test_server_reminder_calls_user_when_not_on_call():
    async def _test():
        import discord
        import discord_bot
        from voice.discord_voice import get_discord_voice_manager

        bot = discord_bot.bot
        bot.send_dm = AsyncMock()

        mock_text_channel = AsyncMock(spec=discord.TextChannel)
        mock_text_channel.send = AsyncMock()

        mock_vc = MagicMock(spec=discord.VoiceChannel)
        mock_vc.id = 444
        mock_vc.name = "General"

        mock_member = MagicMock(spec=discord.Member)
        mock_member.id = 9999
        mock_member.voice = None  # User is NOT on call

        mock_guild = MagicMock(spec=discord.Guild)
        mock_guild.id = 777
        mock_guild.name = "Test Guild"
        mock_guild.voice_channels = [mock_vc]
        mock_guild.get_member.return_value = mock_member

        mock_notification = {
            "id": 21,
            "chat_id": "test-chat-voice-2",
            "reminder_id": "rem-vc-2",
            "content": "Just a nudge: sleep. It's 10:08 PM right now.",
        }

        mock_target = {
            "reminder_id": "rem-vc-2",
            "channel_id": 555,
            "guild_id": 777,
            "user_id": 9999,
        }

        voice_mgr = get_discord_voice_manager()
        with patch.object(bot, "get_channel", return_value=mock_text_channel):
            with patch.object(bot, "get_guild", return_value=mock_guild):
                with patch("app.memory_store.get_reminder_delivery_target", return_value=mock_target):
                    with patch("app.memory_store.list_notifications", return_value=[mock_notification]):
                        with patch("app.memory_store.add_message"):
                            with patch.object(voice_mgr, "join_voice_channel", new_callable=AsyncMock) as mock_join:
                                with patch.object(voice_mgr, "speak_text", new_callable=AsyncMock) as mock_speak:
                                    saved_chats = dict(bot.discord_chats)
                                    saved_ids = dict(bot.last_notification_id)
                                    try:
                                        bot.discord_chats.clear()
                                        bot.last_notification_id.clear()
                                        bot.discord_chats["test-chat-voice-2"] = "9999"
                                        bot.last_notification_id["test-chat-voice-2"] = 0

                                        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
                                            with patch.object(bot, "wait_until_ready", new_callable=AsyncMock):
                                                try:
                                                    await bot.deliver_notifications()
                                                except asyncio.CancelledError:
                                                    pass
                                    finally:
                                        bot.discord_chats = saved_chats
                                        bot.last_notification_id = saved_ids

                                    # S.P.I.D.E.Y should join the server's General voice channel to call user
                                    mock_join.assert_called_once_with(mock_vc, 555, guild_id=777)
                                    # S.P.I.D.E.Y should speak reminder in voice channel
                                    mock_speak.assert_called_once_with(777, 9999, "Just a nudge: sleep. It's 10:08 PM right now.")
                                    # Text channel gets call alert with join link
                                    assert mock_text_channel.send.call_count == 1
                                    call_msg = mock_text_channel.send.call_args[0][0]
                                    assert "📞 <@9999> **Incoming Call — Reminder!**" in call_msg
                                    assert "https://discord.com/channels/777/444" in call_msg
                                    # DM ring alert was sent
                                    bot.send_dm.assert_called_once()
                                    # Pending reminder was registered for user in that voice channel
                                    assert (777, 9999) in bot.pending_call_reminders

    asyncio.run(_test())


def test_on_voice_state_update_speaks_pending_reminder():
    async def _test():
        import discord
        import discord_bot
        from voice.discord_voice import get_discord_voice_manager

        bot = discord_bot.bot
        voice_mgr = get_discord_voice_manager()

        mock_vc = MagicMock(spec=discord.VoiceChannel)
        mock_vc.id = 444
        mock_vc.name = "General"

        mock_guild = MagicMock(spec=discord.Guild)
        mock_guild.id = 777

        mock_member = MagicMock(spec=discord.Member)
        mock_member.id = 9999
        mock_member.display_name = "Jaya Haris"
        mock_member.bot = False
        mock_member.guild = mock_guild

        # Setup pending reminder
        bot.pending_call_reminders[(777, 9999)] = {
            "content": "Just a nudge: sleep. It's 10:08 PM right now.",
            "voice_channel_id": 444,
            "created_at": time.perf_counter(),
        }

        before_state = MagicMock(spec=discord.VoiceState, channel=None)
        after_state = MagicMock(spec=discord.VoiceState, channel=mock_vc)

        with patch.object(voice_mgr, "is_call_active", return_value=True):
            with patch.object(voice_mgr, "speak_text", new_callable=AsyncMock) as mock_speak:
                await bot.on_voice_state_update(mock_member, before_state, after_state)
                # Pending reminder should be consumed
                assert (777, 9999) not in bot.pending_call_reminders
                # S.P.I.D.E.Y speaks greeting and reminder
                mock_speak.assert_called_once_with(
                    777, 9999, "Hey Jaya Haris, you made it! Just a nudge: sleep. It's 10:08 PM right now."
                )

    asyncio.run(_test())


def test_discord_server_text_message_works_without_call_join():
    """Verify that messages in server text channels work normally without needing !call join."""
    async def _test():
        import discord_bot

        mock_msg = MagicMock()
        mock_msg.author.bot = False
        mock_msg.content = "set reminder at 22:08 for sleep"
        mock_msg.guild.id = 777
        mock_msg.channel.id = 555
        mock_msg.mentions = []

        mock_mgr = MagicMock()
        mock_mgr.get_call.return_value = None  # No call active!
        mock_mgr.is_call_active.return_value = False

        with patch.object(discord_bot, "get_discord_voice_manager", return_value=mock_mgr):
            with patch.object(discord_bot.bot, "respond_to_prompt", new_callable=AsyncMock) as mock_respond:
                await discord_bot.on_message(mock_msg)
                mock_respond.assert_called_once_with(mock_msg, "set reminder at 22:08 for sleep")

    asyncio.run(_test())
