from typing import Dict, Tuple, List
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoImageProcessor, AutoTokenizer

import albumentations as A
from albumentations.pytorch import ToTensorV2


class UpStageDocsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        image_pretrained_model_name: str,
        text_pretrained_model_name: str,
        augmentation_probability: float,
        augmentations: List[str],
        text_max_length: int,
    ):
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.image_encoder = AutoImageProcessor.from_pretrained(
            image_pretrained_model_name,
        )
        self.text_encoder = AutoTokenizer.from_pretrained(
            text_pretrained_model_name,
            use_fast=True,
        )
        self.image_paths, self.texts, self.labels = self.get_dataset()
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
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        image = self.transform(image=image)["image"]
        encoded_image = self.encode_image(image)
        encoded_image["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)

        text = self.normalize_string(self.texts[idx])
        encoded_text = self.encode_text(text)
        encoded_text["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)

        image_mask = self.get_padding_mask(
            attention_mask=encoded_text["attention_mask"],
            modality="image",
        )
        text_mask = self.get_padding_mask(
            attention_mask=encoded_text["attention_mask"],
            modality="text",
        )

        return (encoded_image, image_mask, encoded_text, text_mask, self.labels[idx])

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
        elif self.split == "test":
            csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.split in ["train", "test"]:
            image_paths = [
                f"{self.data_path}/{self.split}/{file_name}" for file_name in data["ID"]
            ]
        else:
            image_paths = [
                f"{self.data_path}/train/{file_name}" for file_name in data["ID"]
            ]
        texts = data["text"].tolist()
        labels = data["target"].tolist()
        return (image_paths, texts, labels)

    def get_transform(self):
        transforms = []
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
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(ToTensorV2())
            return A.Compose(transforms)

    def encode_image(
        self,
        data: np.ndarray,
    ) -> torch.Tensor:
        encoded = self.image_encoder(
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
        encoded = self.text_encoder(
            data,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def get_padding_mask(
        self,
        attention_mask: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        if modality == "image":
            output_length = self.text_max_length
        elif modality == "text":
            output_length = attention_mask.sum(-1).to(torch.long)
        else:
            raise ValueError(f"Invalid padding mask modality: {modality}")
        attention_mask = torch.zeros(
            (self.text_max_length,),
            dtype=attention_mask.dtype,
        )
        attention_mask[(output_length - 1,)] = 1
        attention_mask = attention_mask.cumsum(-1).bool()
        return attention_mask
