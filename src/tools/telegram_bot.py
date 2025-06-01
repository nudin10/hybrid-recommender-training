import os
from telegram.ext import Application
import asyncio
from src.tools.logger import Logger
import logging

class TelegramBot:
    def __init__(self, debug=False):
        self.logger = Logger(name="TelegramBot", level=logging.DEBUG if debug else logging.INFO)
        self.token: str = os.getenv("TELEGRAM_BOT_TOKEN") #type:ignore
        if not self.token:
            self.token = os.getenv("RUNPOD_SECRET_TELEGRAM_BOT_TOKEN") #type:ignore

        self.chat_id: str = os.getenv("TELEGRAM_CHAT_ID") #type:ignore
        if not self.chat_id:
            self.chat_id = os.getenv("RUNPOD_SECRET_TELEGRAM_CHAT_ID") #type:ignore

        self.app = Application.builder().token(self.token).build()

    async def send_message(self, message: str):
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=message)
            await asyncio.sleep(1)
        except Exception as e:
            raise
    
    async def send_error(self, message: str) -> None:
        try:
            await self.send_message("[ERROR]: "+message)
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            raise

    async def send_warning(self, message: str) -> None:
        try:
            await self.send_message("[WARNING]: "+message)
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            raise
    
    async def send_messages(self, messages: list[str]) -> None:
        for message in messages:
            try:
                await self.app.bot.send_message(chat_id=self.chat_id, text=message)
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
                raise

telegram_bot: TelegramBot = None #type:ignore

def init_telegram_bot(debug=False):
    global telegram_bot
    if telegram_bot is None:
        telegram_bot = TelegramBot(debug)
    return telegram_bot

def get_telegram_bot() -> TelegramBot:
    if telegram_bot is None:
        raise RuntimeError("Telegram telegram_bot not initialised. Call init_telegram_bot() to initialise.")
    return telegram_bot
