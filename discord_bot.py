"""Discord gateway for the local assistant.

The bot intentionally exposes only conversation generation in this first
integration. Tokens and access controls are read from the process environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import discord
from discord.ext import commands

import app


LOGGER = logging.getLogger("jarvis.discord")
DISCORD_PROVIDER = "discord"
DISCORD_MAX_MESSAGE_LENGTH = 2_000


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


class JarvisBot(commands.Bot):
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

    async def setup_hook(self) -> None:
        LOGGER.info("Discord command registry initialized.")

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

        chat = app.memory_store.get_or_create_external_chat(
            DISCORD_PROVIDER, str(message.author.id)
        )
        chat_id = chat["id"]
        await asyncio.to_thread(app.ensure_model_loaded)
        app.memory_store.add_message(chat_id, "user", prompt)
        context = app.memory_store.build_context(chat_id, app.tokenizer, prompt)
        memories = app.memory_store.search_memories(prompt)
        try:
            async with message.channel.typing():
                response = await asyncio.to_thread(
                    app.generate_response,
                    prompt,
                    app.MAX_NEW_TOKENS,
                    context,
                    memories,
                )
        except ValueError as error:
            await message.reply(str(error))
            return
        except (RuntimeError, OSError):
            LOGGER.exception("Discord generation failed.")
            await message.reply("I could not generate a response right now.")
            return

        app.memory_store.add_message(chat_id, "assistant", response)
        for chunk in _split_response(response):
            await message.reply(chunk)


bot = JarvisBot()


@bot.command(name="ask")
async def ask_command(ctx: commands.Context[Any], *, prompt: str = "") -> None:
    await bot.respond_to_prompt(ctx.message, prompt)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if bot.user is not None and bot.user in message.mentions:
        prompt = message.content.replace(f"<@{bot.user.id}>", "")
        prompt = prompt.replace(f"<@!{bot.user.id}>", "").strip()
        await bot.respond_to_prompt(message, prompt)
        return
    await bot.process_commands(message)


def main() -> None:
    token = os.environ.get("AI_DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "AI_DISCORD_TOKEN is required. Create a Discord bot token and set it "
            "in the backend environment; never put it in the frontend."
        )
    logging.basicConfig(
        level=os.environ.get("AI_DISCORD_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot.run(token)


if __name__ == "__main__":
    main()
