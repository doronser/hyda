import os
import numpy as np
import lightning as pl
import torchxrayvision as xrv
import torchvision.transforms as tforms
from typing import Optional
from torch.utils.data import DataLoader
from torchxrayvision.datasets import Merge_Dataset

from .dataset import NIHDataset, CheXpertDataset, VinBDataset, DOMAIN_ENUM
from .sampler import DomainSampler


class CXRDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 unique_patients: bool = False,
                 target_domain: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 use_sampler: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.unique_patients = unique_patients
        self.target_domain = target_domain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_sampler = use_sampler

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def task_weights(self):
        self.setup()
        train_dl = self.train_dataloader()
        train_labels = train_dl.dataset.labels
        weights = np.nansum(train_labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        return weights

    def setup(self, stage=None):
        # define transforms and augmentations
        transform = tforms.Compose([xrv.datasets.XRayCenterCrop(),
                                    xrv.datasets.XRayResizer(224)])
        # augment = tforms.Compose([lambda img: img.transpose(1,2,0), # hack: support np.array to torchvision.Image
        #                           tforms.ToImage(),
        #                           tforms.RandomHorizontalFlip(p=0.5),
        #                           tforms.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1))])
        augment = tforms.Compose([xrv.datasets.ToPILImage(),
                                  # tforms.RandomHorizontalFlip(p=0.5),
                                  tforms.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                                  tforms.ToTensor()
                                  ])


        # load NIH, CheXphoto and VinBD datasets
        nih_train = NIHDataset(imgpath=os.path.join(self.data_dir, 'nih-chest-xrays/images'),
                                csvpath=os.path.join(self.data_dir, 'nih-chest-xrays/train.csv'),
                                transform=transform, data_aug=augment, unique_patients=self.unique_patients)
        nih_val = NIHDataset(imgpath=os.path.join(self.data_dir, 'nih-chest-xrays/images'),
                              csvpath=os.path.join(self.data_dir, 'nih-chest-xrays/val.csv'),
                              transform=transform, data_aug=None, unique_patients=self.unique_patients)

        chex_train = CheXpertDataset(imgpath=os.path.join(self.data_dir, 'chexpert/'),
                                  csvpath=os.path.join(self.data_dir, 'chexpert/train.csv'),
                                  transform=transform, data_aug=augment, unique_patients=self.unique_patients)
        chex_val = CheXpertDataset(imgpath=os.path.join(self.data_dir, 'chexpert/'),
                                csvpath=os.path.join(self.data_dir, 'chexpert/train_val.csv'),
                                transform=transform, data_aug=None, unique_patients=self.unique_patients)

        vind_train = VinBDataset(imgpath=os.path.join(self.data_dir, 'vinbd/png/train/'),
                                  csvpath=os.path.join(self.data_dir, 'vinbd/train.csv'),
                                  transform=transform, data_aug=augment)
        vinb_val = VinBDataset(imgpath=os.path.join(self.data_dir, 'vinbd/png/train/'),
                                csvpath=os.path.join(self.data_dir, 'vinbd/val.csv'),
                                transform=transform, data_aug=None)

        # label alignment
        for ds in [nih_train, chex_train, vind_train, nih_val, chex_val, vinb_val]:
            # xrv.datasets.relabel_dataset(xrv.models.DenseNet.targets, ds, silent=True)
            subset_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Pneumothorax']
            xrv.datasets.relabel_dataset(subset_labels, ds, silent=False)


        # merge based on target domain
        if self.target_domain is None or self.use_sampler:
            train_ds = Merge_Dataset([nih_train, chex_train, vind_train])
            val_ds = Merge_Dataset([nih_val, chex_val, vinb_val])
            test_ds = None
        elif self.target_domain == 'NIH':
            train_ds = Merge_Dataset([chex_train, vind_train])
            val_ds = Merge_Dataset([chex_val, vinb_val])
            test_ds = nih_val
        elif self.target_domain == 'CheXpert':
            train_ds = Merge_Dataset([nih_train, vind_train])
            val_ds = Merge_Dataset([nih_val, vinb_val])
            test_ds = chex_val
        elif self.target_domain == 'VinBrain':
            train_ds = Merge_Dataset([nih_train, chex_train])
            val_ds = Merge_Dataset([nih_val, chex_val])
            test_ds = vinb_val
        else:
            raise ValueError(f"Target domain {self.target_domain} not supported")

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

    def train_dataloader(self):
        if self.use_sampler:
            # guarantee half of each batch is from the target domain
            sampler = DomainSampler(self.train_dataset.which_dataset, DOMAIN_ENUM[self.target_domain], self.batch_size)
            return DataLoader(self.train_dataset,num_workers=self.num_workers, batch_sampler=sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
