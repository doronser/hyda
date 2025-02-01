import torch
import pandas as pd
from torch import nn
import lightning as pl
from typing import Any
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
from torchmetrics.functional import mean_absolute_error


class BrainAgeLitModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr: float=0.0001,
                 w_decay: float=0):
        """

        :param model: brain age model
        :param lr: Adam learning rate
        :param w_decay: Adam weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 90, 120, 99)
        self.criterion = torch.nn.MSELoss()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.train_r = PearsonCorrCoef()
        self.val_r = PearsonCorrCoef()

        # for logging test metrics
        self.test_age_preds = []
        self.test_age_gts = []


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
                                                                  eta_min=1e-6)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch):
        scan, domain, age, age_bin, subject = batch
        age = age.float().unsqueeze(-1)
        age_predicted = self.model(scan)
        loss = self.criterion(age_predicted, age)
        self.log('train_loss', loss, prog_bar=True)

        for metric_name, metric in [("mae", self.train_mae), ("r", self.train_r)]:
            metric(age_predicted.squeeze(), age.squeeze())
            self.log(f"train_{metric_name}", metric, prog_bar=metric_name=="mae")

        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        for metric_name, metric in [("mae", self.train_mae), ("r", self.train_r)]:
            self.log(f"train_{metric_name}_epoch", metric)

    def validation_step(self, batch):
        scan, domain, age, age_bin, subject = batch
        age = age.float().unsqueeze(-1)
        age_predicted = self.model(scan)
        loss = self.criterion(age_predicted, age)
        self.log('val_loss', loss)

        for metric_name, metric in [("mae", self.val_mae), ("r", self.val_r)]:
            metric(age_predicted.squeeze(), age.squeeze())
            self.log(f"val_{metric_name}", metric)

        return loss

    def on_validation_epoch_end(self):
        for metric_name, metric in [("mae", self.val_mae), ("r", self.val_r)]:
            self.log(f"val_{metric_name}_epoch", metric)

    def test_step(self, batch):
        scan, domain, age, age_bin, subject = batch
        age = age.float().unsqueeze(-1)
        age_predicted = self.model(scan)
        self.test_age_gts.append(age.squeeze().cpu())
        self.test_age_preds.append(age_predicted.squeeze().cpu())

    def on_test_epoch_end(self):
        ages = torch.cat(self.test_age_gts).cpu().numpy()
        age_preds = torch.cat(self.test_age_preds).cpu().numpy()

        test_df = pd.DataFrame(dict(age_pred=age_preds, age=ages))
        test_df["abs_err"] = abs(test_df["age_pred"] - test_df["age"])
        self.logger.log_table("test_df", dataframe=test_df)

        test_mae = mean_absolute_error(age_preds, ages)
        self.log("test_mae", test_mae)


    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

