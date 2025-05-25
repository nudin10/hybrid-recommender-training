from dotenv import load_dotenv
load_dotenv()

from asyncio import run

from src.cache.redis_client import init_redis
from src.tools.telegram_bot import init_telegram_bot
from src.tools.logger import init_global_logger
from data_loader.data_loader import load_data_into_redis
from model.model_data_loader import load_data
from model.definition.recommendation_model import RecommenderSystemModel
from model.hybrid_model import HybridRecommenderModel
from model.bert4Rec import Bert4RecModel
from src.analysis.results import collect, analyse, visualise

async def main():
    try:
        r = init_redis()
        init_telegram_bot()
        init_global_logger()
    except Exception as e:
        raise RuntimeError(f"Error setting up dependencies: {e}")

    try:
        DATA_PATH="data"
        await load_data_into_redis(r, DATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading data into Redis: {e}")
    
    try:
        load_data()
    except Exception as e:
        raise RuntimeError(f"Error loading RecBole data: {e}")
    
    try:
        models = [
            Bert4RecModel,
            HybridRecommenderModel
        ]

        for _model in models:
            model: RecommenderSystemModel = _model()
            model.train()
            model.evaluate()
        
            collect()
            analyse()
            visualise()

    except Exception as e:
        raise RuntimeError(f"Error loading training custom Model: {e}")

if __name__ == "__main__":
    try:
        run(main())
    except Exception as e:
        print(f"Application terminated due to unhandled error: {e}")
        exit(1)
