import os
from typing import Dict, Any
import json
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures.models.timm_model import TimmModel
from ..architectures.timm_architecture import TimmArchitecture


class TimmTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        num_trials: int,
        seed: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.num_trials = num_trials
        self.seed = seed
        self.hparams_save_path = hparams_save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(self.optuna_objective, n_trials=self.num_trials)
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(self.hparams_save_path, exist_ok=True)

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(best_params, json_file)

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        if self.hparams.model_type:
            params["model_type"] = trial.suggest_categorical(
                name="model_type",
                choices=self.hparams.model_type,
            )
        if self.hparams.pretrained:
            params["pretrained"] = trial.suggest_categorical(
                name="pretrained",
                choices=self.hparams.pretrained,
            )
        if self.hparams.lr:
            params["lr"] = trial.suggest_float(
                name="lr",
                low=self.hparams.lr.low,
                high=self.hparams.lr.high,
                log=self.hparams.lr.log,
            )
        if self.hparams.t_max:
            params["t_max"] = trial.suggest_int(
                name="t_max",
                low=self.hparams.t_max.low,
                high=self.hparams.t_max.high,
                log=self.hparams.t_max.log,
            )
        if self.hparams.eta_min:
            params["eta_min"] = trial.suggest_float(
                name="eta_min",
                low=self.hparams.eta_min.low,
                high=self.hparams.eta_min.high,
                log=self.hparams.eta_min.log,
            )

        model = TimmModel(
            model_type=params["model_type"],
            pretrained=params["pretrained"],
            n_classes=self.module_params.num_classes,
        )
        architecture = TimmArchitecture(
            model=model,
            num_classes=self.module_params.num_classes,
            average=self.module_params.average,
            strategy=self.module_params.strategy,
            lr=params["lr"],
            t_max=params["t_max"],
            eta_min=params["eta_min"],
            interval=self.module_params.interval,
        )

        self.logger.log_hyperparams(params)
        callbacks = EarlyStopping(
            monitor=self.module_params.monitor,
            mode=self.module_params.mode,
            patience=self.module_params.patience,
            min_delta=self.module_params.min_delta,
        )

        trainer = Trainer(
            devices=self.module_params.devices,
            accelerator=self.module_params.accelerator,
            strategy=self.module_params.strategy,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=self.logger,
        )

        try:
            trainer.fit(
                model=architecture,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            self.logger.experiment.alert(
                title="Tuning Complete",
                text="Tuning process has successfully finished.",
                level="INFO",
            )
        except Exception as e:
            self.logger.experiment.alert(
                title="Tuning Error",
                text="An error occurred during tuning",
                level="ERROR",
            )
            raise e

        return trainer.callback_metrics["val_MulticlassF1Score"].item()
