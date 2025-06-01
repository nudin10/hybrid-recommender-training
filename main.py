from dotenv import load_dotenv
load_dotenv()

from asyncio import run

from src.cache.redis_client import init_redis
from src.tools.telegram_bot import init_telegram_bot
from src.tools.logger import init_global_logger, get_global_logger
from src.data_loader.data_loader import load_data_into_redis
from src.model.model_data_loader import load_data
from src.model.definition.recommendation_model import RecommenderSystemModel
from src.model.hybrid_model import HybridRecommenderModel
from src.model.bert4Rec import Bert4RecModel
from src.analysis.results import collect, analyse, visualise

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

        models = [
            Bert4RecModel,
            HybridRecommenderModel
        ]

        for _model in models:
            model: Bert4RecModel | HybridRecommenderModel = _model()
            gl.info(f"Training model: {_model.model_name}")

            model.train()
            model.evaluate()
        
            collect()
            analyse()
            visualise()

    except Exception as e:
        raise RuntimeError(f"Error loading training Model: {e}")

if __name__ == "__main__":
    try:
        run(main())
    except Exception as e:
        print(f"Application terminated due to unhandled error: {e}")
        exit(1)
