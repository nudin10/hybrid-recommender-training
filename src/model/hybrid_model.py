from src.model.definition.recommendation_model import RecommenderSystemModel
from src.model.definition.custom_model import HybridRecommenderModel
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

class HybridRecommender(RecommenderSystemModel):
    def __init__(self, dataset_dir_name: str) -> None:
        self.model_name="HybridRecommender"
        self.dataset_dir_name = dataset_dir_name
        self.config_files = ["configs/base.yaml", "configs/hybrid.yaml"]

        super().__init__(model=self.model_name, dataset=self.dataset_dir_name, config_files=self.config_files)

        self.dataset: SequentialDataset = create_dataset(super().config)
        self.contextual_embeddings = self.dataset.get_preload_weight("iid")

        self.train_data, self.valid_data, self.test_data = data_preparation(super().config, self.dataset)

        self.model = HybridRecommenderModel(
            config=super().config,
            dataset=self.train_data.dataset #type: ignore
        ).to(super().config["device"])

        self.trainer: Trainer = None #type: ignore
        

    def train(self):
        if self.trainer is None:
            self.trainer = Trainer(super().config, self.model)
        
        best_valid_score, best_valid_result = self.trainer.fit(self.train_data, self.valid_data, saved=True)

        return best_valid_score, best_valid_result


    def evaluate(self):
        if self.trainer is None:
            self.trainer = Trainer(super().config, self.model)

        test_result = self.trainer.evaluate(self.test_data)

        return test_result
