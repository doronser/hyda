import torch
from torch import nn
import lightning as pl
from torchmetrics import Accuracy


class DigitsLitModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr: float = 0.001,
                 w_decay: float = 0,
                 min_lr: float = 1e-6
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 32, 32)
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

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

    def training_step(self, batch, batch_idx):
        x, y, dom = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        acc = self.train_accuracy(logits, y)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, dom = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        acc = self.val_accuracy(logits, y)
        self.log('val_acc', acc)
        return loss

