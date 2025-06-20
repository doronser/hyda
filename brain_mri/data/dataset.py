import os
import monai
import torch
import numpy as np
import pandas as pd
from typing import List, Literal
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

ALL_DOMAINS =('ukBioBank', 'AIBL', 'Corr', 'GSP', 'UCLA_LA5c', 'ADHD200', 'ixi', 'FCP',
              'Camcan', 'NKI', 'HBN', 'ABIDE', 'ICBM', 'ADNI', 'SLIM', 'PPMI',
              'ADNIDOD', 'Oasis', 'brainomics_data', 'cs_schizbull08', 'COBRE')

DOMAIN_ENUM = {domain: i for i, domain in enumerate(ALL_DOMAINS)}


class MRIDataset(Dataset):
    def __init__(self,
                 data_dir, metadata_path,
                 subset_domains: List[str]=ALL_DOMAINS,
                 domains_to_drop: List[str]=('ICBM'),
                 ages: List[int]=(0, 100),
                 gender: List[Literal['M', 'F']]=('M', 'F'),
                 hand=None,
                 subjects: List[str]=None,
                 data_augmentation_flag=False,
                 scan_normalization_flag=False,
                 scan_normalization_type=None,
                 classification_age_bin_size=5,
                 domain_enum=DOMAIN_ENUM,
                 merge_oasis=True,
                 ):
        self.data_dir = data_dir
        self.subjects = subjects
        self.gender = gender
        self.min_age = ages[0]
        self.max_age = ages[1]
        self.domains = set(subset_domains) - set(domains_to_drop)
        self.data_augmentation_flag = data_augmentation_flag
        self.transform = self.get_transforms() if data_augmentation_flag else None
        self.scan_normalization_flag = scan_normalization_flag
        self.scan_normalization_type = scan_normalization_type
        self.classification_age_bin_size = classification_age_bin_size
        self.domain_enum = domain_enum
        self.merge_oasis = merge_oasis

        self.metadata = pd.read_csv(metadata_path)

        # merge oasis datasets
        if self.merge_oasis:
            self.metadata["ProjTitle"] = self.metadata["ProjTitle"].replace(["OasisLong", "OasisCross"], "Oasis")

        # basic cleanup
        self.metadata = self.metadata.dropna(subset=["Subject", "Age", "Gender"], how='any').reset_index(drop=True)

        # filter by domain
        self.metadata = self.metadata.query("ProjTitle in @subset_domains and ProjTitle not in @domains_to_drop").reset_index(drop=True)

        # filter by age
        self.metadata = self.metadata.query("Age >= @ages[0] and Age <= @ages[1]").reset_index(drop=True)

        # filter by gender
        self.metadata = self.metadata.query("Gender in @gender").reset_index(drop=True)

        # filter by hand
        if hand is not None:
            self.metadata = self.metadata.loc[self.metadata["Hand"] == hand].reset_index(drop=True)

        # filter by subject name
        if subjects is not None:
            self.metadata = self.metadata.loc[self.metadata["Subject"].isin(subjects)].reset_index(drop=True)

        # drop problematic subjects
        problematic_subjects = ["sub82071", "Sub0450_Ses1", "1042211_20252_2_0", "1216360_20252_2_0", "1144361_20252_2_0"]
        self.metadata = self.metadata.query("Subject not in @problematic_subjects").reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)



    @staticmethod
    def get_mean_std_dict():
        mean = {
            'ukBioBank': 0.326,
            'AIBL': 0.290,
            'Corr': 0.281,
            'GSP': 0.160,
            'UCLA_LA5c': 0.260,
            'ixi': 0.353,
            'FCP': 0.266,
            'Camcan': 0.319,
            'NKI': 0.339,
            'HBN': 0.422,
            'ABIDE': 0.280,
            'ICBM': 0.335,
            'ADNI': 0.273,
            'SLIM': 0.310,
            'PPMI': 0.304,
            'ADNIDOD': 0.300,
            'OasisLong': 0.304,
            'OasisCross': 0.304,
            'brainomics_data': 0.259,
            'cs_schizbull08': 0.367,
            'COBRE': 0.271,
        }
        std = {
            'ukBioBank': 0.456,
            'AIBL': 0.421,
            'Corr': 0.390,
            'GSP': 0.301,
            'UCLA_LA5c': 0.358,
            'ixi': 0.427,
            'FCP': 0.391,
            'Camcan': 0.431,
            'NKI': 0.461,
            'HBN': 0.406,
            'ABIDE': 0.386,
            'ICBM': 0.435,
            'ADNI': 0.402,
            'SLIM': 0.419,
            'PPMI': 0.413,
            'ADNIDOD': 0.473,
            'OasisLong': 0.449,
            'OasisCross': 0.434,
            'brainomics_data': 0.367,
            'cs_schizbull08': 0.435,
            'COBRE': 0.365,
        }
        return mean, std

    @staticmethod
    def get_norm_mean_std_dict():
        mean = {
            'ukBioBank': 0.105,
            'AIBL': 0.095,
            'Corr': 0.102,
            'GSP': 0.089,
            'UCLA_LA5c': 0.122,
            'ixi': 0.144,
            'FCP': 0.084,
            'Camcan': 0.124,
            'NKI': 0.109,
            'HBN': 0.172,
            'ABIDE': 0.081,
            'ICBM': 0.102,
            'ADNI': 0.100,
            'SLIM': 0.107,
            'PPMI': 0.108,
            'ADNIDOD': 0.071,
            'OasisLong': 0.100,
            'OasisCross': 0.102,
            'brainomics_data': 0.096,
            'cs_schizbull08': 0.104,
            'COBRE': 0.148,
        }
        std = {
            'ukBioBank': 0.146,
            'AIBL': 0.138,
            'Corr': 0.145,
            'GSP': 0.172,
            'UCLA_LA5c': 0.171,
            'ixi': 0.175,
            'FCP': 0.132,
            'Camcan': 0.168,
            'NKI': 0.148,
            'HBN': 0.170,
            'ABIDE': 0.117,
            'ICBM': 0.141,
            'ADNI': 0.149,
            'SLIM': 0.146,
            'PPMI': 0.148,
            'ADNIDOD': 0.114,
            'OasisLong': 0.148,
            'OasisCross': 0.147,
            'brainomics_data': 0.139,
            'cs_schizbull08': 0.124,
            'COBRE': 0.206,
        }
        return mean, std

    def normalize_scan(self, scan, site):
        if self.scan_normalization_type == 0:
            '0: min - max w.o range scaling - 1, 1;'
            scan = (scan - scan.min()) / (scan.max() - scan.min())
        elif self.scan_normalization_type == 1:
            '1: min-max w. range scaling -1,1;'
            scan = (scan - scan.min()) / (scan.max() - scan.min())
            scan = 2 * scan - 1
        elif self.scan_normalization_type == 2:
            '2: site-based mean and std norm;'
            mu, std = self.get_mean_std_dict()
            mu = mu[site]
            std = std[site]
            scan = (scan - mu) / std
        elif self.scan_normalization_type == 3:
            '3: min-max w.o scaling and site-based;'
            scan = (scan - scan.min()) / (scan.max() - scan.min())
            mu, std = self.get_norm_mean_std_dict()
            mu = mu[site]
            std = std[site]
            scan = (scan - mu) / std
        elif self.scan_normalization_type == 4:
            '4: scan-based mean and std norm;'
            scan = (scan - scan.mean()) / scan.std()
        else:
            '5: min-max w.o scaling and scan-based;'
            scan = (scan - scan.min()) / (scan.max() - scan.min())
            scan = (scan - scan.mean()) / scan.std()

        return scan

    @staticmethod
    def get_transforms():
        """
        Transforms to be applied to the data based on Gideon's paper:
        1. Random rotation x/y/z~uniform(-10, 10) degrees
        2. Random shift intensity ~uniform(-5, 5)
        3. [?] Random scaling ~N(0, 0.1)
        4. Random Gaussian noise ~N(0, 0.015)
        """
        rotate_range_in_degrees = 10  # applied as (-10, 10)
        rotate_range_in_radians = rotate_range_in_degrees * (np.pi / 180)
        transforms = monai.transforms.Compose([
            monai.transforms.RandRotate(range_x=rotate_range_in_radians,
                                        range_y=rotate_range_in_radians,
                                        range_z=rotate_range_in_radians,
                                        prob=0.5),
            monai.transforms.RandShiftIntensity(offsets=5, prob=0.5),  # applied as (-5, 5)
            monai.transforms.RandGaussianNoise(mean=0.0, std=0.015, prob=0.5),
        ])

        return transforms

    def __getitem__(self, index):
        row = self.metadata.loc[index]
        subject = row.Subject
        scan_path = os.path.join(self.data_dir, f"{subject}.npy")

        domain = self.domain_enum.get(row.ProjTitle, -1)

        age = np.float16(row.Age)

        # 0 is set as a fixed min age
        age_bin = int((age - 0) // self.classification_age_bin_size)

        scan = np.float32(np.load(scan_path))
        if self.scan_normalization_flag:
            site = row.ProjTitle
            scan = self.normalize_scan(scan, site)

        scan = transforms.ToTensor()(scan)

        if self.data_augmentation_flag:
            if self.transform is not None:
                scan = self.transform(scan)
            scan = torch.as_tensor(scan)

        scan = scan.permute(1, 2, 0)

        return scan[None, ...], domain, age, age_bin, subject


if __name__ == "__main__":
    from tqdm import tqdm

    src_data_dir = "<PLACEHOLDER_DATA_DIR>"
    train_data_dir = os.path.join(src_data_dir, "train")
    train_metadata_path = os.path.join(src_data_dir, "train_metadata.csv")
    train_dataset = MRIDataset(
        data_dir=train_data_dir,
        metadata_path=train_metadata_path,
        scan_normalization_flag=True,
        scan_normalization_type=1,
        subjects=None,
        data_augmentation_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    for scans_batch, genders_batch, ages_batch, _, _ in tqdm(train_loader):
        print(scans_batch.shape)
