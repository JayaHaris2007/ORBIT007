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
from typing import Any

import discord
from discord.ext import commands

import app


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
        """Poll the shared database and DM reminders to their Discord owners."""
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
                    try:
                        await self.send_dm(int(external_user_id), content)
                        self.last_notification_id[chat_id] = notification["id"]
                        LOGGER.info(
                            "Delivered notification %s to user %s.",
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
            await message.reply("Use `!ask <message>` or mention me with a question.")
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

        chat = app.memory_store.get_or_create_external_chat(
            DISCORD_PROVIDER, str(message.author.id)
        )
        chat_id = chat["id"]
        self.track_chat(chat_id, str(message.author.id))

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
            app.memory_store.add_message(chat_id, "user", prompt)
            app.memory_store.add_message(chat_id, "assistant", response)
            for chunk in _split_response(response):
                await message.reply(chunk)
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


bot = SpideyBot()


@bot.command(name="ask")
async def ask_command(ctx: commands.Context[Any], *, prompt: str = "") -> None:
    await bot.respond_to_prompt(ctx.message, prompt)


@bot.command(name="ping")
async def ping_command(ctx: commands.Context[Any]) -> None:
    """Instant health check that never loads the model."""
    latency = bot.latency * 1000
    detail = f"{round(latency)} ms" if latency > 0 else "unavailable"
    await ctx.reply(f"Online. Gateway latency {detail}.")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if bot.user is not None and bot.user in message.mentions:
        prompt = message.content.replace(f"<@{bot.user.id}>", "")
        prompt = prompt.replace(f"<@!{bot.user.id}>", "").strip()
        await bot.respond_to_prompt(message, prompt)
        return
    if message.guild is None and not message.content.startswith(bot.command_prefix):
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
    guard = _acquire_single_instance()  # noqa: F841 - held for the process lifetime
    bot.run(token)


if __name__ == "__main__":
    main()
