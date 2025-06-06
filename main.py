from dotenv import load_dotenv
load_dotenv()

from asyncio import run

from src.tools.telegram_bot import init_telegram_bot, get_telegram_bot
from src.tools.logger import init_global_logger, get_global_logger
from src.model.model_data_loader import load_data
from src.executor.all import run_all

import traceback
import sys

async def main():
    try:
        print("Initialising logger")
        init_global_logger()
        print("Successfully initiated logger")

        print("Initialising Telegram bot")
        init_telegram_bot()
        print("Successfully initiated Telegram bot")
    
    except Exception as e:
        raise RuntimeError(f"Error setting up dependencies: {e}")
    
    try:
        load_data()
    except Exception as e:
        raise RuntimeError(f"Error loading RecBole data: {e}")
    
    try:
        gl = get_global_logger()
        b = get_telegram_bot()

        m = "Starting recommender model program"
        gl.info(m)
        await b.send_message(m)

        await run_all()

        m = "Finished running recommender model program"
        gl.info(m)
        await b.send_message(m)

    except:
        raise

if __name__ == "__main__":
    try:
        run(main())
    except Exception as e:
        print(f"Application terminated due to error -> {type(e).__name__}: {e}")

        print("\n--- Full Traceback ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # By default it prints to sys.stderr
        print("----------------------\n", file=sys.stderr)

        sys.exit(1)
