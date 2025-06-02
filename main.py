from dotenv import load_dotenv
load_dotenv()

from asyncio import run

from src.tools.telegram_bot import init_telegram_bot, get_telegram_bot
from src.tools.logger import init_global_logger, get_global_logger
from src.model.model_data_loader import load_data
from src.executor.all import run_all

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

        m = "Running all"
        gl.info(m)
        await b.send_message(m)

        run_all()

        m = "Finished running all"
        gl.info(m)
        await b.send_message(m)

    except Exception as e:
        raise RuntimeError(f"Error loading training Model: {e}")

if __name__ == "__main__":
    try:
        run(main())
    except Exception as e:
        print(f"Application terminated due to unhandled error: {e}")
        exit(1)
