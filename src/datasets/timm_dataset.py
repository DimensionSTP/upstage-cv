from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class UpStageDocsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        target_column_name: str,
        num_devices: int,
        batch_size: int,
        image_size: int,
        augmentation_probability: float,
        augmentations: List[str],
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        dataset = self.get_dataset()
        self.image_paths = dataset["image_paths"]
        self.labels = dataset["labels"]
        self.transform = self.get_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        label = self.labels[idx]
        return {
            "image": image,
            "label": label,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
                stratify=data[self.target_column_name],
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
        elif self.split == "predict":
            csv_path = f"{self.data_path}/test.csv"
            data = pd.read_csv(csv_path)
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.split in ["train", "test"]:
            image_paths = [
                f"{self.data_path}/{self.split}/{file_name}" for file_name in data["ID"]
            ]
        elif self.split == "val":
            image_paths = [
                f"{self.data_path}/train/{file_name}" for file_name in data["ID"]
            ]
        else:
            image_paths = [
                f"{self.data_path}/test/{file_name}" for file_name in data["ID"]
            ]
        labels = data[self.target_column_name].tolist()
        return {
            "image_paths": image_paths,
            "labels": labels,
        }

    def get_transform(self) -> A.Compose:
        transforms = [A.Resize(self.image_size, self.image_size)]
        if self.split in ["train", "val"]:
            for aug in self.augmentations:
                if aug == "rotate30":
                    transforms.append(
                        A.Rotate(
                            limit=[30, 30],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate45":
                    transforms.append(
                        A.Rotate(
                            limit=[45, 45],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate90":
                    transforms.append(
                        A.Rotate(
                            limit=[90, 90],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "hflip":
                    transforms.append(
                        A.HorizontalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "vflip":
                    transforms.append(
                        A.VerticalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "noise":
                    transforms.append(
                        A.GaussNoise(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "blur":
                    transforms.append(
                        A.Blur(
                            blur_limit=7,
                            p=self.augmentation_probability,
                        )
                    )
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
