import torch
import numpy as np
from recbole.data.dataset import SequentialDataset

# Unused
class CustomDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self._load_item_embedding()

    def _load_item_embedding(self):
        item_embedding = np.load('my_item_embedding.npy')  # shape: [num_items, dim]
        self.item_embedding = torch.FloatTensor(item_embedding)
