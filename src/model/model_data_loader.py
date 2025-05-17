import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction
import numpy as np


class ItemUserDataLoader(AbstractDataLoader):
    """
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.item_id_field = dataset.item_id_field
        self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})
        self.sample_size = len(self.user_list)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        pass

    def collate_fn(self, index):
        pass

def load_data():
    pass
