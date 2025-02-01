import os
import torch
import pandas as pd
import lightning as pl
from typing import List, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import MRIDataset


class MRIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 target_domain: Optional[str] = None,
                 ages: List[int] = (0, 100),
                 batch_size: int=32,
                 num_workers: int=4,
                 pin_memory: bool=True,
                 balance_by: Optional[str]=None,
                 ):
        super(MRIDataModule, self).__init__()
        self.data_dir = data_dir
        self.target_domain = target_domain
        self.ages = ages
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.balance_by = balance_by

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        train_metadata_path = os.path.join(self.data_dir, "train_metadata.csv")
        val_data_dir = os.path.join(self.data_dir, "val")
        val_metadata_path = os.path.join(self.data_dir, "val_metadata.csv")
        # test_data_dir = os.path.join(self.data_dir, "test")
        # test_metadata_path = os.path.join(self.data_dir, "test_metadata.csv")

        if stage == 'fit':
            self.train_dataset = MRIDataset(train_data_dir, train_metadata_path,
                                            domains_to_drop=['ICBM', self.target_domain],
                                            ages=self.ages,
                                            data_augmentation_flag=True,
                                            scan_normalization_flag=True,
                                            scan_normalization_type=1)

        self.val_dataset = MRIDataset(val_data_dir, val_metadata_path,
                                      domains_to_drop=['ICBM', self.target_domain],
                                      ages=self.ages,
                                      data_augmentation_flag=False,
                                      scan_normalization_flag=True,
                                      scan_normalization_type=1
                                      )
        if stage == "test":
            self.test_dataset = MRIDataset(train_data_dir, train_metadata_path,
                                           subset_domains=[self.target_domain],
                                            ages=self.ages,
                                            data_augmentation_flag=False,
                                            scan_normalization_flag=True,
                                            scan_normalization_type=1)

    def train_dataloader(self):

        if self.balance_by == 'age':
            print("Balancing by age")
            age_col = self.train_dataset.metadata["Age"]
            age_bins = pd.cut(age_col, bins=range(int(age_col.min()) - 1, int(age_col.max()) + 1, 5), right=True)
            age_counts = age_bins.value_counts().sort_index()
            age_counts_mapped = age_bins.map(age_counts).astype(float)
            weights = 1.0 / age_counts_mapped
            weights = torch.tensor(weights.values)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        elif self.balance_by == 'domain':
            print("Balancing by domain")
            domain_col = self.train_dataset.metadata["ProjTitle"]
            domain_counts = domain_col.value_counts().sort_index()
            domain_counts_mapped = domain_col.map(domain_counts).astype(float)
            weights = 1.0 / domain_counts_mapped
            weights = torch.tensor(weights.values)
            sampler = WeightedRandomSampler(weights, num_samples=10000, replacement=True)

        else:
            sampler = None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=self.pin_memory, drop_last=self.target_domain=='Camcan')

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        pass