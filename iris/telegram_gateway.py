"""Telegram gateway: chat with Iris from your phone.

Security model:
- Only Telegram user ids listed in TELEGRAM_ALLOWED_USERS may talk to the bot.
  Anyone else (including you, before configuring it) is told their id so you
  can add it to .env.
- There is no interactive approval prompt over Telegram, so shell commands are
  refused unless IRIS_YOLO=1 is set.
"""

import asyncio
import logging
import sys

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import config
from .agent import Agent

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

_agents: dict[int, Agent] = {}


def _agent_for(chat_id: int) -> Agent:
    if chat_id not in _agents:
        _agents[chat_id] = Agent()
    return _agents[chat_id]


def _authorized(update: Update) -> bool:
    user = update.effective_user
    return bool(
        user
        and config.TELEGRAM_ALLOWED_USERS
        and user.id in config.TELEGRAM_ALLOWED_USERS
    )


async def _deny(update: Update):
    uid = update.effective_user.id if update.effective_user else "?"
    if config.TELEGRAM_ALLOWED_USERS:
        await update.message.reply_text("Sorry, you are not on Iris's allowlist.")
    else:
        await update.message.reply_text(
            "Iris is locked down: no allowlist configured yet.\n"
            f"Your Telegram user id is {uid}.\n"
            f"Add TELEGRAM_ALLOWED_USERS={uid} to the .env file and restart the bot."
        )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        await _deny(update)
        return
    agent = _agent_for(update.effective_chat.id)
    await update.message.reply_text(
        f"Hi! I'm Iris ({agent.description}). Send me anything.\n"
        "/new resets the conversation."
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        await _deny(update)
        return
    _agent_for(update.effective_chat.id).reset()
    await update.message.reply_text("Fresh conversation started.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if not _authorized(update):
        await _deny(update)
        return

    chat_id = update.effective_chat.id
    agent = _agent_for(chat_id)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(agent.chat, update.message.text)
    except Exception as e:
        reply = f"⚠️ error: {e}"
    if not reply.strip():
        reply = "(empty response)"
    # Telegram caps messages at 4096 chars — send in chunks.
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i : i + 4000])


def main():
    if not config.TELEGRAM_BOT_TOKEN:
        print(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "1. Open Telegram, talk to @BotFather, send /newbot, follow the steps.\n"
            "2. Put the token in C:\\Users\\<you>\\iris\\.env as TELEGRAM_BOT_TOKEN=...\n"
            "3. Run this again, message your bot once, and it will tell you your\n"
            "   user id to put in TELEGRAM_ALLOWED_USERS."
        )
        sys.exit(1)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Iris Telegram gateway running (Ctrl+C to stop)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
