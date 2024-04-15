from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DocImagesDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        image_size: int,
    ):
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.image_size = image_size
        self.image_paths, self.labels = self.get_dataset()
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        label = self.labels[idx]
        return (image, label)

    def get_dataset(self) -> Tuple[List[str], List[str]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            train_data, val_data = train_test_split(
                data, test_size=self.split_ratio, random_state=self.seed, shuffle=True
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "predict":
            csv_path = f"{self.data_path}/predict.csv"
            data = pd.read_csv(csv_path)
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.split in ["train", "predict"]:
            image_paths = [
                f"{self.data_path}/{self.split}/{file_name}" for file_name in data["ID"]
            ]
        else:
            image_paths = [
                f"{self.data_path}/train/{file_name}" for file_name in data["ID"]
            ]
        labels = data["target"].tolist()
        return (image_paths, labels)

    def get_transform(self):
        if self.split == "train":
            return A.Compose(
                [
                    A.Resize(256, 256),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(256, 256),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
