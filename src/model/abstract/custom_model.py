from recbole.model.abstract_recommender import SequentialRecommender

class HybridRecommenderModel(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
