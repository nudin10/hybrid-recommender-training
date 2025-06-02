from src.tools.logger import get_global_logger
from src.tools.telegram_bot import get_telegram_bot
from src.analysis.results import Result
from src.model.hybrid_model import HybridRecommender
from src.model.bert4Rec import Bert4Rec
from datetime import datetime

import gc

async def run_all() -> None:
    gl = get_global_logger()
    b = get_telegram_bot()

    models = [
        {
            "name": "Bert4Rec",
            "dataset_dir_name" : "base",
            "model": Bert4Rec
        },
        {
            "name": "HybridPhi",
            "dataset_dir_name" : "phi",
            "model": HybridRecommender
        },
        # {
        #     "name": "HybridQwen",
        #     "dataset_dir_name" : "qwen",
        #     "model": HybridRecommender
        # }
    ]

    for model_config in models:
        m = f"Training {model_config['name']}"
        gl.info(m)
        await b.send_message(m)

        unique_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        result = Result(name=model_config["name"]+"_"+unique_suffix)

        model: Bert4Rec | HybridRecommender  = model_config["model"](dataset_dir_name = model_config["dataset_dir_name"])

        try:
            best_valid_score, best_valid_result = model.train()
            test_result = model.evaluate()
        except:
            raise

        result.collect(
            {
                "best_valid_score": best_valid_score,
                "best_valid_result": best_valid_result,
                "test_result": test_result,
            }
        )
        stored_path = result.store()
        
        m = f"Successfully stored result in: {str(stored_path.resolve())}"
        gl.info(m)
        await b.send_message(m)

        model.cleanup()
        
        gc.collect()

        gl.info(f"Finished training {model_config['name']}")
