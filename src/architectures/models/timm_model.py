import torch
from torch import nn

import timm


class TimmModel(nn.Module):
    def __init__(self, model_type: str, pretrained: str, n_classes: int) -> None:
        super().__init__()
        if pretrained == "pretrained":
            is_pretrained = True
        elif pretrained == "raw":
            is_pretrained = False
        else:
            raise ValueError(f"Invalid pretrained: {pretrained}")
        self.model = timm.create_model(
            model_type, pretrained=is_pretrained, num_classes=n_classes
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(image)
        return output
