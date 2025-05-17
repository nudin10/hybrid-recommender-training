# Not supposed to be ran during online training
# Should be done offline
# Consolidates all review text for each items into one
# Consolidation via mean pooling (averaging)
import torch

def mean_pooling_consolidator(embeddings: list[torch.Tensor]) -> torch.Tensor:
    stacked_embeddings = torch.stack(embeddings)
    consolidated = torch.mean(stacked_embeddings, dim=0)
    return consolidated
