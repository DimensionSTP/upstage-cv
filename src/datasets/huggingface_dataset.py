from typing import Dict, Tuple, List
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoImageProcessor, AutoTokenizer, AutoProcessor

import albumentations as A
from albumentations.pytorch import ToTensorV2


class UpStageDocsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        modality: str,
        pretrained_model_name: str,
        augmentation_probability: float,
        augmentations: List[str],
        text_max_length: int,
    ):
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.modality = modality
        if self.modality == "image":
            self.data_encoder = AutoImageProcessor.from_pretrained(
                pretrained_model_name,
            )
        elif self.modality == "text":
            self.data_encoder = AutoTokenizer.from_pretrained(
                pretrained_model_name,
                use_fast=True,
            )
        elif self.modality == "multi-modality":
            self.data_encoder = AutoProcessor.from_pretrained(
                pretrained_model_name,
                apply_ocr=True,
                ocr_lang="kor",
            )
        else:
            raise ValueError(f"Invalid modality: {self.modality}")
        self.datas, self.labels = self.get_dataset()
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        self.transform = self.get_transform()
        self.text_max_length = text_max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        if self.modality in ["image", "multi-modality"]:
            data = np.array(Image.open(self.datas[idx]).convert("RGB"))
            data = self.transform(image=data)["image"]
            if self.modality == "image":
                encoded = self.encode_image(data)
            else:
                encoded = self.encode_text(data)
        else:
            data = self.normalize_string(self.datas[idx])
            encoded = self.encode_text(data)
        encoded["labels"] = torch.tensor([self.labels[idx]], dtype=torch.long).squeeze(
            0
        )
        return encoded

    def get_dataset(self) -> Tuple[List[str], List[str]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
                stratify=data["target"],
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "predict":
            csv_path = f"{self.data_path}/text.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.modality in ["image", "multi-modality"]:
            if self.split in ["train", "predict"]:
                datas = [
                    f"{self.data_path}/{self.split}/{file_name}"
                    for file_name in data["ID"]
                ]
            else:
                datas = [
                    f"{self.data_path}/train/{file_name}" for file_name in data["ID"]
                ]
        else:
            datas = data["text"].tolist()
        labels = data["target"].tolist()
        return (datas, labels)

    def get_transform(self):
        transforms = []
        if self.split in ["train", "val"]:
            for aug in self.augmentations:
                if aug == "rotate30":
                    transforms.append(
                        A.Rotate(limit=[30, 30], p=self.augmentation_probability)
                    )
                elif aug == "rotate45":
                    transforms.append(
                        A.Rotate(limit=[45, 45], p=self.augmentation_probability)
                    )
                elif aug == "rotate90":
                    transforms.append(
                        A.Rotate(limit=[90, 90], p=self.augmentation_probability)
                    )
                elif aug == "hflip":
                    transforms.append(A.HorizontalFlip(p=self.augmentation_probability))
                elif aug == "vflip":
                    transforms.append(A.VerticalFlip(p=self.augmentation_probability))
                elif aug == "noise":
                    transforms.append(A.GaussNoise(p=self.augmentation_probability))
                elif aug == "blur":
                    transforms.append(
                        A.Blur(blur_limit=7, p=self.augmentation_probability)
                    )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(ToTensorV2())
            return A.Compose(transforms)

    def encode_image(
        self,
        data: np.ndarray,
    ) -> torch.Tensor:
        encoded = self.data_encoder(
            data,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    @staticmethod
    def normalize_string(
        data: str,
    ) -> str:
        data = re.sub(r"[\s]", r" ", str(data))
        data = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", str(data))
        return data

    def encode_text(
        self,
        data: str,
    ) -> torch.Tensor:
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded
