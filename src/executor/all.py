from src.tools.logger import get_global_logger
from src.analysis.results import Result
from src.model.hybrid_model import HybridRecommender
from src.model.bert4Rec import Bert4Rec

import gc

def run_all():
    gl = get_global_logger()

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
        gl.info(f"Training base {model_config['name']}")
        result = Result(name=model_config["name"])

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
        gl.info(f"Successfully stored result in: {str(stored_path.resolve())}")

        model.cleanup()
        
        gc.collect()

        gl.info(f"Finished training base {model_config['name']}")
