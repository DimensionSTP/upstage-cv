from typing import Tuple, Dict, Any

import torch
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection, F1Score, Accuracy

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


class MultiModalArchitecture(LightningModule):
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: nn.Module,
        model: nn.Module,
        num_labels: int,
        average: str,
        strategy: str,
        multimodal_weight: float,
        modality_split_weight: float,
        dynamic_loss_weight: float,
        lr: float,
        t_max: int,
        eta_min: float,
        interval: str,
    ) -> None:
        super().__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.model = model
        self.multimodal_weight = multimodal_weight
        self.image_weight = (1 - self.multimodal_weight) * modality_split_weight
        self.text_weight = (1 - self.multimodal_weight) * (1 - modality_split_weight)
        self.dynamic_loss_weight = dynamic_loss_weight
        self.strategy = strategy
        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.interval = interval

        metrics = MetricCollection(
            [
                F1Score(
                    task="multiclass",
                    num_classes=num_labels,
                    average=average,
                ),
                Accuracy(
                    task="multiclass",
                    num_classes=num_labels,
                    average=average,
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        encoded_image: torch.Tensor,
        image_mask: torch.Tensor,
        encoded_text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_output = self.image_backbone(encoded_image)
        text_output = self.text_backbone(encoded_text)
        multimodal_output = self.model(
            image=image_output.hidden_states[0].squeeze(),
            image_mask=image_mask,
            text=text_output.hidden_states[0].squeeze(),
            text_mask=text_mask,
        )
        return (image_output, text_output, multimodal_output)

    def step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, image_mask, text, text_mask, label = batch
        image_output, text_output, multimodal_output = self(
            encoded_image=image,
            image_mask=image_mask,
            encoded_text=text,
            text_mask=text_mask,
        )

        image_loss = image_output.loss
        text_loss = text_output.loss
        multimodal_loss = F.cross_entropy(multimodal_output, label)

        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch

        weighted_multimodal_loss = self.multimodal_weight * multimodal_loss
        weighted_image_loss = (
            self.image_weight * image_loss
        ) + self.dynamic_loss_weight * (current_epoch / total_epochs)
        weighted_text_loss = (
            self.text_weight * text_loss
        ) - self.dynamic_loss_weight * (current_epoch / total_epochs)
        loss = weighted_multimodal_loss + weighted_image_loss + weighted_text_loss
        pred = torch.argmax(multimodal_output, dim=1)
        return (loss, pred, label)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(self.parameters(), lr=self.lr)
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": self.interval},
        }

    def training_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        loss, pred, label = self.step(batch)
        metrics = self.train_metrics(pred, label)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"loss": loss, "pred": pred, "label": label}

    def validation_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        loss, pred, label = self.step(batch)
        metrics = self.val_metrics(pred, label)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"loss": loss, "pred": pred, "label": label}

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        _, pred, _ = self.step(batch)
        gathered_pred = self.all_gather(pred)
        return gathered_pred

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.reset()