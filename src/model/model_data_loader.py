import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction
import numpy as np


# Unused
class ItemUserDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        pass

    def collate_fn(self, index):
        pass

def load_data():
    pass
