from abc import ABC, abstractmethod
from recbole.config import Config
from recbole.utils import init_logger, init_seed
from src.tools.common import get_project_root
from pathlib import Path
import logging

class RecommenderSystemModel(ABC):
    def __init__(self, model, dataset: str, config_files: list[str]) -> None:
        absolute_config_files = []
        project_root = get_project_root()
        
        for config in config_files:
            config_path = Path(config)
            if not config_path.is_absolute():
                absolute_path = (project_root / "src" / "model" / config_path).resolve()
            else:
                absolute_path = config_path.resolve()
            
            if not absolute_path.exists():
                raise FileNotFoundError(f"Config file not found: {str(absolute_path)}")
            
            absolute_config_files.append(str(absolute_path))

        self.config = Config(model=model, dataset=dataset, config_file_list=absolute_config_files)
        
        data_parent_dir = (project_root / self.config["dataset_dir"] / dataset).resolve() # type: ignore
        self.config["data_path"] = str(data_parent_dir)
        
        init_seed(self.config["seed"], self.config["reproducibility"])

        self.model_name = type(self).__name__ 
        
        init_logger(self.config)
        self.logger = logging.getLogger(name=self.model_name)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(c_handler)
        self.logger.info(self.config)

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
