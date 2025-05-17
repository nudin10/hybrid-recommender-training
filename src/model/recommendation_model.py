from abc import ABC, abstractmethod

class RecommenderSystemModel(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
