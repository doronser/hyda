import os
from skimage.io import imread
from torchxrayvision.datasets import VinBrain_Dataset, NIH_Dataset, CheX_Dataset, normalize, apply_transforms


DOMAIN_ENUM = {
    'NIH': 0,
    'CheXpert': 1,
    'VinBrain': 2
}
DOMAINS = list(DOMAIN_ENUM.keys())

class VinBDataset(VinBrain_Dataset):
    """Load precomputed pngs instead of dicoms for faster processing"""
    def __getitem__(self, idx):
        sample = {"idx": idx,
                  "lab": self.labels[idx]}

        img_id = self.csv['image_id'].iloc[idx]
        img_path = os.path.join(self.imgpath, img_id + ".png")
        img = imread(img_path)
        sample['img'] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        sample['domain'] = DOMAIN_ENUM['VinBrain']
        return sample

class NIHDataset(NIH_Dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['domain'] = DOMAIN_ENUM['NIH']
        return sample

class CheXpertDataset(CheX_Dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['domain'] = DOMAIN_ENUM['CheXpert']
        return sample