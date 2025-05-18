import os
import numpy as np
import lightning as pl
from typing import Optional
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.transforms as tforms

from .dataset import CIFAR10Dataset, CIFAR10CDataset, CORRUPTIONS


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 target_domain: Optional[str] = None,
                 severity : int = 2,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.target_domain = target_domain
        self.severity = severity
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tform = tforms.Compose([tforms.ToTensor(),
                                     tforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def setup(self, stage: str) -> None:
        # load base cifar10
        cifar10_train = CIFAR10Dataset(root=self.data_dir, train=True, transform=self.tform)
        # load all corruptions
        cifar10c_dict = {corruption: CIFAR10CDataset(data_dir=self.data_dir, corruption=corruption,
                                                     severity=self.severity, transform=self.tform)
                         for corruption in CORRUPTIONS}

        # train_ds = ConcatDataset([cifar10_train] + list([ds for corruption, ds in cifar10c_dict.items() if corruption != 'none']))


