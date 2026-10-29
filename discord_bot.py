"""Discord gateway for the local assistant.

The bot shares the same task, habit, and reminder commands as the web chat,
falling back to conversation generation for everything else. Tokens and access
controls are read from the process environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import time
from typing import Any

import discord
from discord.ext import commands

import app
from voice.discord_voice import get_discord_voice_manager


LOGGER = logging.getLogger("spidey.discord")
DISCORD_PROVIDER = "discord"
DISCORD_MAX_MESSAGE_LENGTH = 2_000
DISCORD_NOTIFICATION_POLL_SECONDS = 5
DISCORD_LOCK_PORT = 47653


def _acquire_single_instance() -> socket.socket:
    """Refuse to start when another bot process already owns the lock port."""
    guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        guard.bind(("127.0.0.1", DISCORD_LOCK_PORT))
        guard.listen(1)
    except OSError as error:
        guard.close()
        raise SystemExit(
            "Another discord_bot.py instance is already running. Stop it before "
            "starting a new one."
        ) from error
    return guard


def _parse_id_allowlist(name: str) -> set[int]:
    raw_value = os.environ.get(name, "")
    values: set[int] = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError as error:
            raise RuntimeError(f"{name} contains an invalid Discord ID.") from error
    return values


def _split_response(response: str) -> list[str]:
    return [
        response[index : index + DISCORD_MAX_MESSAGE_LENGTH]
        for index in range(0, len(response), DISCORD_MAX_MESSAGE_LENGTH)
    ] or ["I generated an empty response."]


# The local model sometimes writes keycap numbers with stray combining marks,
# dingbats or braille (1️⃣, 2️⃝, 4️⠁), which Discord shows as broken glyphs.
# The web panel cleans these at render time; Discord has no such layer.
_BROKEN_LIST_MARKER = re.compile(
    r"(?m)^[ \t]*(\d{1,2})"
    r"[\ufe0f\u20d0-\u20ff\u2460-\u24ff\u2700-\u27bf\u2800-\u28ff]+"
    r"[ \t]*(?=\S)"
)
_INVISIBLE_MARKS = re.compile(
    r"[\ufe00-\ufe0f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufffd]"
)


def _clean_reply_text(text: str) -> str:
    text = _BROKEN_LIST_MARKER.sub(r"\1. ", text)
    text = _INVISIBLE_MARKS.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class SpideyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(
            command_prefix=os.environ.get("AI_DISCORD_PREFIX", "!"),
            intents=intents,
            help_command=None,
        )
        self.allowed_user_ids = _parse_id_allowlist("AI_DISCORD_ALLOWED_USER_IDS")
        self.allowed_guild_ids = _parse_id_allowlist("AI_DISCORD_ALLOWED_GUILD_IDS")
        self.discord_chats: dict[str, str] = {}
        self.last_notification_id: dict[str, int] = {}
        self.pending_call_reminders: dict[tuple[int, int], dict[str, Any]] = {}

    async def setup_hook(self) -> None:
        for chat in app.memory_store.list_external_chats(DISCORD_PROVIDER):
            self.track_chat(chat["id"], str(chat["external_user_id"]))
        app._restore_reminders(provider=DISCORD_PROVIDER)
        ready = app.init_learning_module(DISCORD_PROVIDER)
        LOGGER.info(
            "Learning module %s; Discord weekly nudge %s.",
            "initialized" if ready else "NOT available",
            "scheduled" if ready else "not scheduled",
        )
        asyncio.create_task(self.deliver_notifications())
        LOGGER.info(
            "Discord command registry initialized; tracking %d chat(s).",
            len(self.discord_chats),
        )

    def track_chat(self, chat_id: str, external_user_id: str) -> None:
        if chat_id in self.discord_chats:
            return
        self.discord_chats[chat_id] = external_user_id
        self.last_notification_id[chat_id] = app.memory_store.latest_notification_id(
            chat_id
        )

    async def deliver_notifications(self) -> None:
        """Poll the shared database and deliver reminders to server channels or DMs."""
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(DISCORD_NOTIFICATION_POLL_SECONDS)
            for chat_id, external_user_id in list(self.discord_chats.items()):
                last_id = self.last_notification_id.get(chat_id, 0)
                try:
                    notifications = await asyncio.to_thread(
                        app.memory_store.list_notifications, chat_id, last_id
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not read notifications for chat %s.", chat_id
                    )
                    continue
                for notification in notifications:
                    content = notification["content"].strip()
                    if not content:
                        self.last_notification_id[chat_id] = notification["id"]
                        continue

                    reminder_id = notification.get("reminder_id")
                    target = None
                    if reminder_id:
                        try:
                            target = await asyncio.to_thread(
                                app.memory_store.get_reminder_delivery_target,
                                reminder_id,
                            )
                        except Exception as e:
                            LOGGER.warning("Could not fetch reminder target for %s: %s", reminder_id, e)

                    delivered = False
                    if target and target.get("channel_id") and target.get("guild_id"):
                        guild_id = target["guild_id"]
                        channel_id = target["channel_id"]
                        user_id = target["user_id"]

                        channel = self.get_channel(channel_id)
                        if channel is None:
                            try:
                                channel = await self.fetch_channel(channel_id)
                            except Exception:
                                channel = None

                        guild = self.get_guild(guild_id)
                        if guild is None:
                            try:
                                guild = await self.fetch_guild(guild_id)
                            except Exception:
                                guild = None

                        member = None
                        if guild is not None:
                            member = guild.get_member(user_id)
                            if member is None:
                                try:
                                    member = await guild.fetch_member(user_id)
                                except Exception:
                                    member = None

                        if channel is not None and isinstance(
                            channel, (discord.TextChannel, discord.Thread)
                        ):
                            try:
                                voice_mgr = get_discord_voice_manager()
                                user_vc = getattr(getattr(member, "voice", None), "channel", None) if member else None

                                if user_vc is not None:
                                    # Case A: User is currently on call in user_vc
                                    call = voice_mgr.get_call(guild_id)
                                    if not call or getattr(getattr(call, "voice_client", None), "channel", None) != user_vc:
                                        await voice_mgr.join_voice_channel(user_vc, channel_id, guild_id=guild_id)

                                    asyncio.create_task(
                                        voice_mgr.speak_text(guild_id, user_id, content)
                                    )

                                    user_mention = f"<@{user_id}>"
                                    vc_name = getattr(user_vc, "name", "voice")
                                    server_msg = f"{user_mention} {content} *(Spoken in 🔊 {vc_name})*"
                                    for chunk in _split_response(server_msg):
                                        await channel.send(chunk)
                                else:
                                    # Case B: User is NOT currently on call ("if i am not on call")
                                    target_vc = None
                                    if guild and getattr(guild, "voice_channels", None):
                                        for vc in guild.voice_channels:
                                            if "general" in vc.name.lower():
                                                target_vc = vc
                                                break
                                        if target_vc is None and guild.voice_channels:
                                            target_vc = guild.voice_channels[0]

                                    if target_vc is not None:
                                        # Join target voice channel to call the user
                                        await voice_mgr.join_voice_channel(target_vc, channel_id, guild_id=guild_id)

                                        # Register pending reminder so when user hops in, Spidey speaks
                                        self.pending_call_reminders[(guild_id, user_id)] = {
                                            "content": content,
                                            "voice_channel_id": target_vc.id,
                                            "text_channel_id": channel_id,
                                            "created_at": time.perf_counter(),
                                        }

                                        # Speak reminder in voice channel
                                        asyncio.create_task(
                                            voice_mgr.speak_text(guild_id, user_id, content)
                                        )

                                        # Send incoming call alert in text channel with join link
                                        user_mention = f"<@{user_id}>"
                                        call_link = f"https://discord.com/channels/{guild_id}/{target_vc.id}"
                                        server_msg = (
                                            f"📞 {user_mention} **Incoming Call — Reminder!**\n"
                                            f"{content}\n"
                                            f"👉 I joined **{target_vc.name}** to call you — [Click here to join the call]({call_link})!"
                                        )
                                        for chunk in _split_response(server_msg):
                                            await channel.send(chunk)

                                        # Also send a quick DM ring alert in case user has Discord minimized
                                        guild_name = getattr(guild, "name", "Server")
                                        dm_msg = (
                                            f"📞 **S.P.I.D.E.Y is calling you in {guild_name} — Reminder!**\n"
                                            f"{content}\n"
                                            f"👉 Join the call in **{target_vc.name}**: {call_link}"
                                        )
                                        try:
                                            await self.send_dm(int(user_id), dm_msg)
                                        except Exception:
                                            pass

                                        # Schedule auto-leave after 2 minutes if user doesn't join
                                        async def _auto_leave_if_unjoined(gid: int, uid: int, vcid: int) -> None:
                                            await asyncio.sleep(120)
                                            pending_check = self.pending_call_reminders.get((gid, uid))
                                            if pending_check and pending_check.get("voice_channel_id") == vcid:
                                                self.pending_call_reminders.pop((gid, uid), None)
                                                c = voice_mgr.get_call(gid)
                                                if c and getattr(getattr(c, "voice_client", None), "channel", None) and c.voice_client.channel.id == vcid:
                                                    members = getattr(c.voice_client.channel, "members", [])
                                                    human_members = [m for m in members if not getattr(m, "bot", False)]
                                                    if not human_members:
                                                        await voice_mgr.leave_call(gid)
                                                        LOGGER.info("Auto-left voice channel %s after 2 min unanswered reminder call.", vcid)

                                        asyncio.create_task(_auto_leave_if_unjoined(guild_id, user_id, target_vc.id))
                                    else:
                                        user_mention = f"<@{user_id}>"
                                        server_msg = f"{user_mention} {content}"
                                        for chunk in _split_response(server_msg):
                                            await channel.send(chunk)

                                self.last_notification_id[chat_id] = notification["id"]
                                LOGGER.info(
                                    "Delivered notification %s to server channel %s for user %s (voice handled).",
                                    notification["id"],
                                    channel_id,
                                    user_id,
                                )
                                delivered = True
                                try:
                                    await asyncio.to_thread(
                                        app.memory_store.add_message,
                                        chat_id,
                                        "assistant",
                                        content,
                                    )
                                except Exception:
                                    pass
                            except discord.Forbidden:
                                LOGGER.warning(
                                    "Cannot send notification %s to channel %s (forbidden); falling back to DM.",
                                    notification["id"],
                                    channel_id,
                                )
                            except Exception:
                                LOGGER.exception(
                                    "Could not deliver notification %s to channel %s; falling back to DM.",
                                    notification["id"],
                                    channel_id,
                                )

                    if not delivered:
                        try:
                            await self.send_dm(int(external_user_id), content)
                            self.last_notification_id[chat_id] = notification["id"]
                            LOGGER.info(
                                "Delivered notification %s to user %s via DM.",
                                notification["id"],
                                external_user_id,
                            )
                            try:
                                await asyncio.to_thread(
                                    app.memory_store.add_message,
                                    chat_id,
                                    "assistant",
                                    content,
                                )
                            except Exception:
                                pass
                        except discord.Forbidden:
                            self.last_notification_id[chat_id] = notification["id"]
                            LOGGER.warning(
                                "Cannot DM notification %s to user %s; DMs are closed.",
                                notification["id"],
                                external_user_id,
                            )
                        except Exception:
                            LOGGER.exception(
                                "Could not deliver notification %s.",
                                notification["id"],
                            )

    async def send_dm(self, user_id: int, content: str) -> None:
        user = self.get_user(user_id)
        if user is None:
            user = await self.fetch_user(user_id)
        for chunk in _split_response(content):
            await user.send(chunk)

    async def on_ready(self) -> None:
        if self.user is not None:
            LOGGER.info("Discord bot connected as %s (%s).", self.user, self.user.id)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """When a user joins a voice channel where Spidey is waiting for a reminder, speak it."""
        if getattr(member, "bot", False) or not getattr(after, "channel", None):
            return

        key = (member.guild.id, member.id)
        pending = self.pending_call_reminders.get(key)
        if pending and pending.get("voice_channel_id") == after.channel.id:
            # Check within 5 minutes
            if time.perf_counter() - pending.get("created_at", 0) < 300:
                self.pending_call_reminders.pop(key, None)
                voice_mgr = get_discord_voice_manager()
                if voice_mgr.is_call_active(member.guild.id):
                    content = pending["content"]
                    LOGGER.info(
                        "User %s joined voice channel %s for pending reminder.",
                        member.display_name,
                        after.channel.name,
                    )
                    asyncio.create_task(
                        voice_mgr.speak_text(
                            member.guild.id,
                            member.id,
                            f"Hey {member.display_name}, you made it! {content}",
                        )
                    )

    def is_allowed(self, message: discord.Message) -> bool:
        if self.allowed_user_ids and message.author.id not in self.allowed_user_ids:
            return False
        if (
            self.allowed_guild_ids
            and message.guild is not None
            and message.guild.id not in self.allowed_guild_ids
        ):
            return False
        return True

    async def respond_to_prompt(
        self, message: discord.Message, prompt: str
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            await message.reply("Use `!ask <message>` or `!input <message>` or mention me with a question.")
            return
        if len(prompt) > app.MAX_PROMPT_CHARS:
            await message.reply(
                f"That message is too long. Keep it under {app.MAX_PROMPT_CHARS} characters."
            )
            return
        if not self.is_allowed(message):
            LOGGER.warning(
                "Rejected Discord request from user=%s guild=%s.",
                message.author.id,
                message.guild.id if message.guild else "dm",
            )
            await message.reply("You are not authorized to use this assistant.")
            return

        LOGGER.info(
            "Discord prompt from user=%s guild=%s chars=%d",
            message.author.id,
            message.guild.id if message.guild else "dm",
            len(prompt),
        )

        # Handle voice call intents ("join call", "!join call", "leave call", "voice on/off")
        if message.guild is not None:
            norm_call = re.sub(r"\s+", " ", prompt.strip().lower()).rstrip(".!?")
            voice_mgr = get_discord_voice_manager()
            if re.fullmatch(
                r"(?:(?:call\s+)?join(?:\s+(?:the\s+|my\s+)?(?:call|voice|vc|voice\s+channel))?|"
                r"hop\s+(?:in|into)\s+(?:call|voice|vc)|connect\s+to\s+(?:voice|call|vc))",
                norm_call,
            ):
                _, msg = await voice_mgr.join_call(message)
                await message.reply(msg)
                return
            if re.fullmatch(
                r"(?:(?:call\s+)?(?:leave|exit)(?:\s+(?:the\s+|my\s+)?(?:call|voice|vc|voice\s+channel))?|"
                r"disconnect(?:\s+from\s+(?:call|voice|vc))?|hang\s+up)",
                norm_call,
            ):
                msg = await voice_mgr.leave_call(message.guild.id)
                await message.reply(msg)
                return
            if re.fullmatch(r"(?:call\s+)?stop(?:\s+(?:speaking|talking|voice))?", norm_call):
                msg = voice_mgr.stop_speaking(message.guild.id)
                await message.reply(msg)
                return
            voice_toggle_match = re.fullmatch(
                r"(?:call\s+)?voice\s+(on|off|enable|disable|1|0)", norm_call
            )
            if voice_toggle_match:
                enable = voice_toggle_match.group(1) in {"on", "enable", "1"}
                msg = voice_mgr.set_voice_enabled(message.guild.id, enable)
                await message.reply(msg)
                return

        chat = app.memory_store.get_or_create_external_chat(
            DISCORD_PROVIDER, str(message.author.id)
        )
        chat_id = chat["id"]
        self.track_chat(chat_id, str(message.author.id))

        existing_reminders = {
            r["id"] for r in app.memory_store.list_reminders(active_only=True)
        }

        try:
            command = await asyncio.to_thread(
                app._handle_local_command, prompt, chat_id
            )
        except ValueError as error:
            await message.reply(str(error))
            return
        except Exception:
            LOGGER.exception("Discord command handling failed.")
            await message.reply("Something went wrong handling that command.")
            return

        if command is not None:
            response, provider = command
            LOGGER.info("Answered locally with %s.", provider)
            new_reminders = [
                r for r in app.memory_store.list_reminders(active_only=True)
                if r["id"] not in existing_reminders and r.get("chat_id") == chat_id
            ]
            for r in new_reminders:
                channel_id = message.channel.id if message.guild is not None else None
                guild_id = message.guild.id if message.guild is not None else None
                await asyncio.to_thread(
                    app.memory_store.save_reminder_delivery_target,
                    r["id"],
                    channel_id,
                    guild_id,
                    message.author.id,
                )
            app.memory_store.add_message(chat_id, "user", prompt)
            app.memory_store.add_message(chat_id, "assistant", response)
            for chunk in _split_response(response):
                await message.reply(chunk)
            if message.guild is not None:
                voice_mgr = get_discord_voice_manager()
                if voice_mgr.is_call_active(message.guild.id):
                    asyncio.create_task(
                        voice_mgr.speak_text(message.guild.id, message.author.id, response)
                    )
            return

        app.memory_store.add_message(chat_id, "user", prompt)
        try:
            async with message.channel.typing():
                await asyncio.to_thread(app.ensure_model_loaded)
                context = app.memory_store.build_context(
                    chat_id, app.tokenizer, prompt
                )
                memories = app._memories_for_prompt(prompt)
                response = await asyncio.to_thread(
                    app.generate_response,
                    prompt,
                    app._token_budget(prompt),
                    context,
                    memories,
                )
        except ValueError as error:
            await message.reply(str(error))
            return
        except Exception:
            LOGGER.exception("Discord generation failed.")
            await message.reply("I could not generate a response right now.")
            return

        response = _clean_reply_text(response)
        app.memory_store.add_message(chat_id, "assistant", response)
        for chunk in _split_response(response):
            await message.reply(chunk)
        if message.guild is not None:
            voice_mgr = get_discord_voice_manager()
            if voice_mgr.is_call_active(message.guild.id):
                asyncio.create_task(
                    voice_mgr.speak_text(message.guild.id, message.author.id, response)
                )

    async def respond_to_voice_utterance(
        self,
        guild_id: int,
        user: discord.User | discord.Member,
        text_channel_id: int,
        prompt: str,
    ) -> None:
        """Process an utterance heard in Discord voice, replying in text and voice."""
        prompt = prompt.strip()
        if not prompt:
            return

        text_channel = self.get_channel(text_channel_id)
        if text_channel is not None and isinstance(text_channel, (discord.TextChannel, discord.Thread)):
            try:
                await text_channel.send(f"🎙️ **{user.display_name}**: {prompt}")
            except Exception as e:
                LOGGER.warning("Could not post user voice transcript to text channel: %s", e)

        chat = app.memory_store.get_or_create_external_chat(
            DISCORD_PROVIDER, str(user.id)
        )
        chat_id = chat["id"]
        self.track_chat(chat_id, str(user.id))

        existing_reminders = {
            r["id"] for r in app.memory_store.list_reminders(active_only=True)
        }

        try:
            command = await asyncio.to_thread(
                app._handle_local_command, prompt, chat_id
            )
        except ValueError as error:
            if text_channel is not None:
                await text_channel.send(str(error))
            return
        except Exception:
            LOGGER.exception("Discord voice command handling failed.")
            if text_channel is not None:
                await text_channel.send("Something went wrong handling that command.")
            return

        if command is not None:
            response, provider = command
            LOGGER.info("Voice answered locally with %s.", provider)
            new_reminders = [
                r for r in app.memory_store.list_reminders(active_only=True)
                if r["id"] not in existing_reminders and r.get("chat_id") == chat_id
            ]
            for r in new_reminders:
                await asyncio.to_thread(
                    app.memory_store.save_reminder_delivery_target,
                    r["id"],
                    text_channel_id,
                    guild_id,
                    user.id,
                )
            app.memory_store.add_message(chat_id, "user", prompt)
            app.memory_store.add_message(chat_id, "assistant", response)
            if text_channel is not None:
                for chunk in _split_response(response):
                    await text_channel.send(chunk)
            voice_mgr = get_discord_voice_manager()
            if voice_mgr.is_call_active(guild_id):
                asyncio.create_task(
                    voice_mgr.speak_text(guild_id, user.id, response)
                )
            return

        app.memory_store.add_message(chat_id, "user", prompt)
        try:
            from contextlib import nullcontext
            typing_ctx = text_channel.typing() if text_channel is not None else nullcontext()
            async with typing_ctx:
                await asyncio.to_thread(app.ensure_model_loaded)
                context = app.memory_store.build_context(
                    chat_id, app.tokenizer, prompt
                )
                memories = app._memories_for_prompt(prompt)
                response = await asyncio.to_thread(
                    app.generate_response,
                    prompt,
                    app._token_budget(prompt),
                    context,
                    memories,
                )
        except ValueError as error:
            if text_channel is not None:
                await text_channel.send(str(error))
            return
        except Exception:
            LOGGER.exception("Discord voice generation failed.")
            if text_channel is not None:
                await text_channel.send("I could not generate a response right now.")
            return

        response = _clean_reply_text(response)
        app.memory_store.add_message(chat_id, "assistant", response)
        if text_channel is not None:
            for chunk in _split_response(response):
                await text_channel.send(chunk)
        voice_mgr = get_discord_voice_manager()
        if voice_mgr.is_call_active(guild_id):
            asyncio.create_task(
                voice_mgr.speak_text(guild_id, user.id, response)
            )


bot = SpideyBot()


@bot.command(name="ask")
async def ask_command(ctx: commands.Context[Any], *, prompt: str = "") -> None:
    await bot.respond_to_prompt(ctx.message, prompt)


@bot.command(name="input")
async def input_command(ctx: commands.Context[Any], *, prompt: str = "") -> None:
    await bot.respond_to_prompt(ctx.message, prompt)


@bot.command(name="ping")
async def ping_command(ctx: commands.Context[Any]) -> None:
    """Instant health check that never loads the model."""
    latency = bot.latency * 1000
    detail = f"{round(latency)} ms" if latency > 0 else "unavailable"
    await ctx.reply(f"Online. Gateway latency {detail}.")


@bot.group(name="call", invoke_without_command=True)
async def call_group(ctx: commands.Context[Any]) -> None:
    """Voice call controls: !call join, !call leave, !call stop, !call voice on|off."""
    await ctx.reply("Usage: `!call join`, `!call leave`, `!call stop`, `!call voice on|off`")


@call_group.command(name="join")
async def call_join(ctx: commands.Context[Any]) -> None:
    voice_mgr = get_discord_voice_manager()
    success, msg = await voice_mgr.join_call(ctx.message)
    await ctx.reply(msg)


@call_group.command(name="leave")
async def call_leave(ctx: commands.Context[Any]) -> None:
    if ctx.guild is None:
        await ctx.reply("Calls are only available in a server.")
        return
    voice_mgr = get_discord_voice_manager()
    msg = await voice_mgr.leave_call(ctx.guild.id)
    await ctx.reply(msg)


@call_group.command(name="exit")
async def call_exit(ctx: commands.Context[Any]) -> None:
    """Alias for !call leave."""
    await call_leave(ctx)


@call_group.command(name="stop")
async def call_stop(ctx: commands.Context[Any]) -> None:
    if ctx.guild is None:
        return
    voice_mgr = get_discord_voice_manager()
    msg = voice_mgr.stop_speaking(ctx.guild.id)
    await ctx.reply(msg)


@call_group.command(name="voice")
async def call_voice_toggle(ctx: commands.Context[Any], mode: str = "") -> None:
    if ctx.guild is None:
        return
    voice_mgr = get_discord_voice_manager()
    mode_clean = mode.strip().lower()
    if mode_clean in {"on", "enable", "1"}:
        msg = voice_mgr.set_voice_enabled(ctx.guild.id, True)
    elif mode_clean in {"off", "disable", "0"}:
        msg = voice_mgr.set_voice_enabled(ctx.guild.id, False)
    else:
        msg = "Usage: `!call voice on` or `!call voice off`"
    await ctx.reply(msg)


@bot.command(name="join")
async def top_join_command(ctx: commands.Context[Any], *, _rest: str = "") -> None:
    """Alias for !call join and !join call."""
    voice_mgr = get_discord_voice_manager()
    _, msg = await voice_mgr.join_call(ctx.message)
    await ctx.reply(msg)


@bot.command(name="leave")
async def top_leave_command(ctx: commands.Context[Any], *, _rest: str = "") -> None:
    """Alias for !call leave and !leave call."""
    if ctx.guild is None:
        await ctx.reply("Calls are only available in a server.")
        return
    voice_mgr = get_discord_voice_manager()
    msg = await voice_mgr.leave_call(ctx.guild.id)
    await ctx.reply(msg)


@bot.command(name="stop")
async def top_stop_command(ctx: commands.Context[Any], *, _rest: str = "") -> None:
    """Alias for !call stop."""
    if ctx.guild is None:
        return
    voice_mgr = get_discord_voice_manager()
    msg = voice_mgr.stop_speaking(ctx.guild.id)
    await ctx.reply(msg)


@bot.command(name="voice")
async def top_voice_command(ctx: commands.Context[Any], mode: str = "") -> None:
    """Alias for !call voice on|off."""
    await call_voice_toggle(ctx, mode=mode)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if message.guild is not None:
        clean_text = message.content.strip().lower()
        if clean_text in {"stop", "stop talking"}:
            voice_mgr = get_discord_voice_manager()
            if voice_mgr.is_call_active(message.guild.id):
                reply_msg = voice_mgr.stop_speaking(message.guild.id)
                await message.reply(reply_msg)
                return
    if bot.user is not None and bot.user in message.mentions:
        prompt = message.content.replace(f"<@{bot.user.id}>", "")
        prompt = prompt.replace(f"<@!{bot.user.id}>", "").strip()
        await bot.respond_to_prompt(message, prompt)
        return
    if not message.content.startswith(bot.command_prefix):
        await bot.respond_to_prompt(message, message.content)
        return
    await bot.process_commands(message)


@bot.event
async def on_command_error(
    ctx: commands.Context[Any], error: commands.CommandError
) -> None:
    """Treat an unregistered `!something` as a prompt, like the web chat."""
    if isinstance(error, commands.CommandNotFound):
        prefix = ctx.prefix or ""
        await bot.respond_to_prompt(ctx.message, ctx.message.content[len(prefix) :])
        return
    LOGGER.error("Discord command failed: %s", error)


_discord_task: asyncio.Task | None = None


async def start_discord_bot_in_process(token: str) -> None:
    """Start the Discord bot inside the existing FastAPI server event loop."""
    global _discord_task
    if _discord_task is not None and not _discord_task.done():
        LOGGER.info("Discord bot is already running in-process.")
        return
    LOGGER.info("Starting Discord bot in-process with FastAPI server...")
    _discord_task = asyncio.create_task(bot.start(token))


async def stop_discord_bot_in_process() -> None:
    """Stop the in-process Discord bot gracefully."""
    global _discord_task
    if bot.is_ready():
        LOGGER.info("Closing in-process Discord bot connection...")
        try:
            await bot.close()
        except Exception as e:
            LOGGER.warning("Error closing Discord bot: %s", e)
    if _discord_task is not None:
        try:
            await _discord_task
        except (asyncio.CancelledError, Exception):
            pass
        _discord_task = None


def main() -> None:
    token = os.environ.get("AI_DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "AI_DISCORD_TOKEN is required. Add it to the .env file in the project "
            "root (Discord Developer Portal -> Bot -> Reset Token); never put it "
            "in the frontend."
        )
    logging.basicConfig(
        level=os.environ.get("AI_DISCORD_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Check if the S.P.I.D.E.Y server is already running with the in-process bot
    try:
        import urllib.request
        req = urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1.5)
        if req.status == 200:
            print("\n[INFO] S.P.I.D.E.Y server is already running on port 8000 with the in-process Discord bot.")
            print("To avoid duplicate model loading in VRAM, the bot runs inside the server process.")
            print("No separate discord_bot.py process is needed. Exiting.\n")
            return
    except Exception:
        pass

    from runtime.guard import acquire_model_process_lock
    _model_guard = acquire_model_process_lock()  # noqa: F841
    guard = _acquire_single_instance()  # noqa: F841 - held for the process lifetime
    bot.run(token)


if __name__ == "__main__":
    main()
