from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from model.definition.recommendation_model import RecommenderSystemModel
from recbole.config import Config
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

class Bert4RecModel(RecommenderSystemModel):
    def __init__(self, dataset_dir_name: str) -> None:
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
        

    def train(self):
        trainer = Trainer(super().config, self.model)
        best_valid_score, best_valid_result = trainer.fit(self.train_data)

        return best_valid_score, best_valid_result


    def evaluate(self):
        pass
