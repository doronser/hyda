import torch
import pandas as pd
from torch import nn
import lightning as pl
from typing import Any
from torchmetrics.classification import MultilabelAUROC
from chest_xray.models.loss import MultiLabelClassificationLoss


class CXRLitModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 task_weights: list = None,
                 lr: float = 0.0001,
                 w_decay: float = 0,
                 min_lr: float = 1e-6
                 ):
        """

        :param model: chest x-ray model
        :param task_weights: weights per task for loss function
        :param lr: optimizer lr param
        :param w_decay: optimizer decay param
        :param min_lr: scheduler min lr param
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 224, 224)
        self.criterion = MultiLabelClassificationLoss(weights=task_weights)

        # self.train_auc = MultilabelAUROC(num_labels=18, ignore_index=100)
        # self.val_auc = MultilabelAUROC(num_labels=18, ignore_index=100)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        if self.min_lr is not None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
                                                                      eta_min=self.min_lr)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def training_step(self, batch):
        xrays = batch['img']
        labels = batch['lab']

        logits = self.model(xrays)
        loss = self.criterion(logits, labels)
        # self.train_auc(logits,  torch.nan_to_num(labels, nan=100))
        # self.log('train_AUC', self.train_auc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    # def on_train_epoch_end(self) -> None:
    #     self.log('train_AUC_epoch', self.train_auc)

    def validation_step(self, batch):
        xrays = batch['img']
        labels = batch['lab']

        logits = self.model(xrays)
        loss = self.criterion(logits, labels)
        # self.val_auc(logits, torch.nan_to_num(labels, nan=100))
        # self.log('val_AUC', self.val_auc, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    # def on_validation_epoch_end(self) -> None:
    #     self.log('val_AUC_epoch', self.val_auc)