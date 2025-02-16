import torch
import numpy as np
from torch.utils.data import BatchSampler


class DomainSampler(BatchSampler):
    def __init__(self,
                 ds_indices: np.array,
                 target_domain: int,
                 batch_size: int
                 ):
        # super().__init__()
        self.ds_indices = ds_indices
        self.batch_size = batch_size
        self.target_domain = target_domain

        # Get indices for each domain
        self.target_indices = np.where(ds_indices == target_domain)[0]
        self.source_indices = np.where(ds_indices != target_domain)[0]

    def __iter__(self):
        fixed_size = self.batch_size // 2 # half batch for target domain, half for source domains
        num_batches = min(len(self.target_indices), len(self.source_indices)) //fixed_size
        target_indices_shuffled = torch.randperm(len(self.target_indices)).tolist()
        source_indices_shuffled = torch.randperm(len(self.source_indices)).tolist()

        for i in range(num_batches):
            batch_target = [self.target_indices[idx] for idx in
                            target_indices_shuffled[i *fixed_size:(i + 1) * fixed_size]]
            batch_other = [self.source_indices[idx] for idx in
                           source_indices_shuffled[i *fixed_size :(i + 1) * fixed_size]]
            yield batch_target + batch_other

    def __len__(self):
        return min(len(self.target_indices), len(self.source_indices)) // (self.batch_size // 2)

