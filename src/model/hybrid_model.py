from src.model.definition.recommendation_model import RecommenderSystemModel
from src.model.definition.custom_model import HybridRecommenderModel
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from src.errors.model import TrainingException, EvaluationException
import torch

class HybridRecommender(RecommenderSystemModel):
    def __init__(self, dataset_dir_name: str) -> None:
        self.dataset_dir_name = dataset_dir_name
        self.config_files = ["configs/base.yaml", "configs/hybrid.yaml"]

        super().__init__(model=HybridRecommenderModel, dataset=self.dataset_dir_name, config_files=self.config_files)

        self.dataset: SequentialDataset = create_dataset(self.config)
        self.contextual_embeddings = self.dataset.get_preload_weight("iid")

        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)

        self.model = HybridRecommenderModel(
            config=self.config,
            dataset=self.train_data.dataset #type: ignore
        ).to(self.config["device"])

        self.trainer: Trainer = None #type: ignore
        
    def train(self):
        if self.trainer is None:
            self.trainer = Trainer(self.config, self.model)
        
        try:
            best_valid_score, best_valid_result = self.trainer.fit(self.train_data, self.valid_data, saved=True, show_progress=True)
        except Exception as e:
            raise TrainingException(e)

        return best_valid_score, best_valid_result

    def evaluate(self):
        if self.trainer is None:
            self.trainer = Trainer(self.config, self.model)

        try:
            test_result = self.trainer.evaluate(self.test_data, show_progress=True)
        except Exception as e:
            raise EvaluationException(e)

        return test_result
    
    def cleanup(self):
        self.model = None
        if self.trainer is not None:
            if hasattr(self.trainer, 'model'):
                self.trainer.model = None
            
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                self.trainer.optimizer = None #type: ignore

            self.trainer = None #type: ignore

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.dataset = None #type: ignore
        self.contextual_embeddings = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
