import torch
import torch.nn as nn
import torch.nn.functional as F

from .crossmodal_transformer import CrossModalTransformer


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        num_layers: int,
        image_dims: int,
        text_dims: int,
        text_max_length: int,
        num_labels: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
        emb_dropout: float,
        out_dropout: float,
        attn_mask: bool,
        scale_embedding: bool,
    ) -> None:
        super().__init__()
        combined_dims = model_dims * 2

        self.image_encoder = nn.Conv1d(
            image_dims,
            model_dims,
            3,
            padding=1,
            bias=True,
        )
        self.text_encoder = nn.Conv1d(
            text_dims,
            model_dims,
            3,
            padding=1,
            bias=True,
        )

        kwargs = {
            "model_dims": model_dims,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "text_max_length": text_max_length,
            "attn_dropout": attn_dropout,
            "relu_dropout": relu_dropout,
            "res_dropout": res_dropout,
            "emb_dropout": emb_dropout,
            "attn_mask": attn_mask,
            "scale_embedding": scale_embedding,
        }

        self.image_text = self.get_network(**kwargs)
        self.text_image = self.get_network(**kwargs)
        self.image_self = self.get_network(**kwargs)
        self.text_self = self.get_network(**kwargs)

        self.fc1 = nn.Linear(combined_dims, combined_dims)
        self.fc2 = nn.Linear(combined_dims, combined_dims)
        self.dropout = nn.Dropout(out_dropout)
        self.out_layer = nn.Linear(combined_dims, num_labels)

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        image = self.image_encoder(image.transpose(1, 2)).transpose(1, 2)
        text = self.text_encoder(text.transpose(1, 2)).transpose(1, 2)
        image = self.image_text(query=image, key=text, key_padding_mask=text_mask)
        text = self.text_image(query=text, key=image, key_padding_mask=image_mask)
        image = self.image_self(query=image, key=image, key_padding_mask=image_mask)
        text = self.text_self(query=text, key=text, key_padding_mask=text_mask)
        features = torch.cat([image, text], dim=2)
        pooler_output = features[:, 0, :].squeeze()
        out = pooler_output + self.fc2(self.dropout(F.relu(self.fc1(pooler_output))))

        return self.out_layer(out)

    @staticmethod
    def get_network(**kwargs) -> nn.Module:
        return CrossModalTransformer(
            model_dims=kwargs["model_dims"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            text_max_length=kwargs["text_max_length"],
            attn_dropout=kwargs["attn_dropout"],
            relu_dropout=kwargs["relu_dropout"],
            res_dropout=kwargs["res_dropout"],
            emb_dropout=kwargs["emb_dropout"],
            attn_mask=kwargs["attn_mask"],
            scale_embedding=["scale_embedding"],
        )
