from dotenv import load_dotenv
load_dotenv()

from asyncio import run

from src.cache.redis_client import init_redis
from src.tools.telegram_bot import init_telegram_bot
from src.tools.logger import init_global_logger
from data_loader.data_loader import load_data_into_redis

async def main():
    r = init_redis()
    init_telegram_bot()
    init_global_logger()

    DATA_PATH="data"
    await load_data_into_redis(r, DATA_PATH)

if __name__ == "__main__":
    run(main())
