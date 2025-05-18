import torch
import lightning as pl
import torchvision.transforms as tforms
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .dataset import MNIST, MNISTM, SVHN, USPS

class DigitsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, target_domain=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_domain = target_domain
        self.transform = tforms.Compose([tforms.Grayscale(),
                                         tforms.Resize((32, 32), interpolation=tforms.InterpolationMode.NEAREST),
                                         tforms.ToTensor(),
                                         tforms.Normalize((0.5,), (0.5,))
                                         ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_digits_dataset(self, dataset_cls, test_size, **kwargs):
        ds = dataset_cls(self.data_dir, transform=self.transform, **kwargs)
        train_idx, val_idx = train_test_split(torch.arange(len(ds)), stratify=ds.targets,
                                              test_size=test_size, random_state=42)
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
        return train_ds, val_ds

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            # load MNIST, MNIST-M, SVHN and USPS datasets
            mnist_train, mnist_val = self.load_digits_dataset(MNIST, 10_000, train=True)
            svhn_train, svhn_val = self.load_digits_dataset(SVHN, 7_000, split='train')
            usps_train, usps_val = self.load_digits_dataset(USPS, 1_000, train=True)
            mnistm_train, mnistm_val = self.load_digits_dataset(MNISTM, 10_000, train=True)

            if self.target_domain is None:
                self.train_dataset = torch.utils.data.ConcatDataset([mnist_train, mnistm_train, svhn_train, usps_train])
                self.val_dataset = torch.utils.data.ConcatDataset([mnist_val, mnistm_train, svhn_val, usps_val])
            elif self.target_domain == 'MNIST':
                self.train_dataset = torch.utils.data.ConcatDataset([mnistm_train, svhn_train, usps_train])
                self.val_dataset = torch.utils.data.ConcatDataset([mnistm_train, svhn_val, usps_val])
                self.test_dataset = mnist_val
            elif self.target_domain == 'MNIST-M':
                self.train_dataset = torch.utils.data.ConcatDataset([mnist_train, svhn_train, usps_train])
                self.val_dataset = torch.utils.data.ConcatDataset([mnist_val, svhn_val, usps_val])
                self.test_dataset = mnistm_val
            elif self.target_domain == 'SVHN':
                self.train_dataset = torch.utils.data.ConcatDataset([mnist_train, mnistm_train, usps_train])
                self.val_dataset = torch.utils.data.ConcatDataset([mnist_val, mnistm_train, usps_val])
                self.test_dataset = svhn_val
            elif self.target_domain == 'USPS':
                self.train_dataset = torch.utils.data.ConcatDataset([mnist_train, mnistm_train, svhn_train])
                self.val_dataset = torch.utils.data.ConcatDataset([mnist_val, mnistm_train, svhn_val])
                self.test_dataset = usps_val
            else:
                raise NotImplementedError('target domain is supported')
        elif stage=='test':
            mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            mnistm_test = MNISTM(self.data_dir, train=False, transform=self.transform)
            usps_test = USPS(self.data_dir, train=False, transform=self.transform)
            svhn_test = SVHN(self.data_dir, split='test', transform=self.transform)
            self.test_dataset = dict(MNIST=mnist_test, MNISTM=mnistm_test, USPS=usps_test, SVHN=svhn_test)

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        if isinstance(self.test_dataset,Dataset):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        elif isinstance(self.test_dataset, dict):
            return {ds_name: DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                    for ds_name, ds in self.test_dataset.items()}