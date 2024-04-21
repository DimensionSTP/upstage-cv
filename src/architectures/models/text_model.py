import torch
from torch import nn

from transformers import AutoModelForSequenceClassification


class TextModel(nn.Module):
    def __init__(
        self,
        pretrained_dataset: str,
        model_type: str,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = f"{pretrained_dataset}/{model_type}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name, num_labels=n_classes, output_hidden_states=False
        )

    def forward(
        self,
        tokenized_text: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(**tokenized_text)
        return output
