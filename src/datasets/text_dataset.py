from typing import Dict, Tuple, List
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class UpStageDocsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        pretrained_dataset: str,
        model_type: str,
        text_max_length: int,
    ):
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.pretrained_model_name = f"{pretrained_dataset}/{model_type}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, use_fast=True
        )
        self.text_max_length = text_max_length
        self.texts, self.labels = self.get_dataset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        normalized_text = self.normalize_string(self.texts[idx])
        tokenized_text = self.tokenize_text(normalized_text)
        tokenized_text["labels"] = torch.tensor(
            [self.labels[idx]], dtype=torch.long
        ).squeeze(0)
        return tokenized_text

    def get_dataset(self) -> Tuple[List[str], List[str]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train_text_ver0.csv"
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
            csv_path = f"{self.data_path}/predict_text_ver1.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        texts = data["text"].tolist()
        labels = data["target"].tolist()
        return (texts, labels)

    @staticmethod
    def normalize_string(
        text: str,
    ) -> str:
        text = re.sub(r"[\s]", r" ", str(text))
        text = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", str(text))
        return text

    def tokenize_text(
        self,
        text: str,
    ) -> torch.Tensor:
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
        }
