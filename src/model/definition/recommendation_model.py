from abc import ABC, abstractmethod
from recbole.config import Config
from recbole.utils import init_logger, init_seed
import logging

class RecommenderSystemModel(ABC):
    def __init__(self, model: str, dataset: str, config_files: list[str]) -> None:
        self.config = Config(model=model, dataset=dataset, config_file_list=config_files)
        
        init_seed(self.config["seed"], self.config["reproducibility"])
        
        init_logger(self.config)
        self.logger = logging.getLogger()
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(c_handler)
        self.logger.info(self.config)

        self.model_name=""

        super().__init__()

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def cleanup(self):
        ...
