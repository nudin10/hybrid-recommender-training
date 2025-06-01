import polars as pl
from pathlib import Path
from src.cache.redis_client import Redis_Client
from src.data_loader.batch_operations import load_into_redis_by_batch
from src.tools.telegram_bot import get_telegram_bot
from src.tools.logger import get_global_logger

async def load_data_into_redis(redis_client: Redis_Client, data_file_path: str):
    '''
    Script to consolidate different reviews per item based on item id. Consolidation here is "combining" the different embeddings.
    Produce one single file that can be easily loaded into Redis cache as HKeys
    '''

    model_keys = [
        "phi",
        "qwen"
    ]

    for model_key in model_keys:
        message=f"Loading data for model: {model_key}"
        b = get_telegram_bot()
        await b.send_message(message)
        gl = get_global_logger()

        gl.info(message)

        file_path = data_file_path + f"/load_redis_{model_key}.ndjson"
        gl.info(f"File path: {Path(file_path).absolute().resolve()}")
        try:
            load_into_redis_by_batch(redis_client, model_key=model_key, file_path=file_path)
        except Exception as e:
            raise
