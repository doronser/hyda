import os
import numpy as np
from robustbench.data import load_cifar10c, load_cifar10, load_cifar100, load_cifar100c
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

CORRUPTIONS = ['none',
               'brightness',
               'contrast',
               'defocus_blur',
               'elastic_transform',
               'fog',
               'frost',
               'gaussian_blur',
               'gaussian_noise',
               'glass_blur',
               'impulse_noise',
               'jpeg_compression',
               'labels',
               'motion_blur',
               'pixelate',
               'saturate',
               'shot_noise',
               'snow',
               'spatter',
               'speckle_noise',
               'zoom_blur'
               ]

CORRUPTIONS_ENUM = {corruption: i for i, corruption in enumerate(CORRUPTIONS)}

class CIFAR10Dataset(CIFAR10):
    def __init__(self, root='data',
                 train=True,
                 transform= None):
        super().__init__(root=root, train=train, transform=transform)
        self.corruption = 'none'
        self.curruption_label = CORRUPTIONS_ENUM[self.corruption]

    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        return image, label, np.zeros_like(label)



class CIFAR10CDataset(Dataset):
    """Dataset for loading corrupted versions of CIFAR-10-C test dataset"""
    def __init__(self, data_dir, corruption, severity=-1, transform=None):
        self.data_dir = data_dir
        self.corruption = corruption
        self.corruption_label = CORRUPTIONS_ENUM[corruption]
        self.transform = transform

        data_path = os.path.join(data_dir, corruption + ".npy")
        labels_path = os.path.join(data_dir, "labels.npy")
        self.labels = np.load(labels_path)
        self.data = np.load(data_path)

        cifar_size = 10000 # total number of test images in dataset
        if severity > 0:
            self.data = self.data[(severity - 1) * cifar_size : severity * cifar_size]

        # convert to HWC
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.labels[item]
        corrution_label = np.ones_like(label) * self.corruption_label
        return image, label, corrution_label

