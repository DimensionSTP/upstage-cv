import torch
from torch import nn

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
)


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        modality: str,
        pretrained_model_name: str,
        num_labels: int,
        is_backbone: bool,
    ) -> None:
        super().__init__()
        if modality in ["text", "multi-modality"]:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                output_hidden_states=is_backbone,
                ignore_mismatched_sizes=True,
            )
        elif modality == "image":
            self.model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                output_hidden_states=is_backbone,
                ignore_mismatched_sizes=True,
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def forward(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(**encoded)
        return output
