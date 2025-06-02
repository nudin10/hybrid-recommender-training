from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from src.model.definition.recommendation_model import RecommenderSystemModel
from recbole.config import Config
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from src.errors.model import TrainingException, EvaluationException

class Bert4RecModel(RecommenderSystemModel):
    def __init__(self, dataset_dir_name: str = "base") -> None:
        self.model_name="Bert4Rec"
        self.dataset_dir_name = dataset_dir_name
        self.config_files = ["configs/base.yaml", "configs/bert4Rec.yaml"]

        super().__init__(model=self.model_name, dataset=self.dataset_dir_name, config_files=self.config_files)

        self.dataset: SequentialDataset = create_dataset(super().config)
        self.contextual_embeddings = self.dataset.get_preload_weight("iid")

        self.train_data, self.valid_data, self.test_data = data_preparation(super().config, self.dataset)

        self.model = BERT4Rec(
            config=super().config,
            dataset=self.train_data.dataset #type: ignore
        ).to(super().config["device"])

        self.trainer: Trainer = None #type: ignore

    def train(self):
        if self.trainer is None:
            self.trainer = Trainer(super().config, self.model)
        
        try:
            best_valid_score, best_valid_result = self.trainer.fit(self.train_data, self.valid_data, saved=True)
        except Exception as e:
            raise TrainingException(e)

        return best_valid_score, best_valid_result

    def evaluate(self):
        if self.trainer is None:
            self.trainer = Trainer(super().config, self.model)

        try:
            test_result = self.trainer.evaluate(self.test_data)
        except Exception as e:
            raise EvaluationException(e)

        return test_result
