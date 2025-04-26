import torch
import numpy as np
from torch import nn
import lightning as pl
import torchxrayvision as xrv
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_auroc
from typing import Optional

from hyda.layers import grad_reverse
from ..data.dataset import DOMAIN_ENUM, DOMAINS
from ..models.loss import MultiLabelClassificationLoss


class CXRDANN(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 target_domain: Optional[str],
                 num_domains: int = 3,
                 task_weights: list = None,
                 domain_loss_weight: float = 1,
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
        self.target_domain = target_domain
        self.target_domain_idx = DOMAIN_ENUM.get(self.target_domain, -1)
        self.num_domains = num_domains
        # self.domain_clf = nn.Linear(1024, num_domains)
        self.domain_clf = nn.Sequential(nn.Linear(1024, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, num_domains))
        self.domain_loss_weight = domain_loss_weight
        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 224, 224)
        self.criterion = MultiLabelClassificationLoss(weights=task_weights)
        if self.model.num_classes == 5:  # intersection of all pathologies
            self.pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Pneumothorax']
        elif self.model.num_classes == 18:  # union of all pathologies
            self.pathologies = xrv.models.DenseNet.targets
        else:
            raise ValueError('num_classes must be either 5 or 18 (intersection/union of all pathologies)')

        # aggregators for AUC calculation
        self.train_logits = []
        self.train_labels = []
        self.val_logits = []
        self.val_labels = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.w_decay)
        if self.min_lr is not None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
                                                                      eta_min=self.min_lr)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def log_auc(self, logits, labels, prefix='train'):
        valid_classes = ~torch.isnan(labels).all(0)
        if valid_classes.sum().item() == 0:
            print('wtf?!', prefix)
            return
        valid_logits = logits[:, valid_classes]
        valid_labels = labels[:, valid_classes]
        auc = multilabel_auroc(valid_logits, torch.nan_to_num(valid_labels, nan=100).long(),
                               num_labels=valid_classes.sum().item(), ignore_index=100, average='none')
        valid_pathos = [self.pathologies[i] for i, valid_cls in enumerate(valid_classes) if valid_cls]
        for i, auc_val in enumerate(auc):
            self.log(f'{prefix}_AUC/{valid_pathos[i]}', auc_val.item(), prog_bar=False)
        self.log(f'{prefix}_AUC', auc.mean().item(), prog_bar=prefix=='train')


    def _step(self, batch, prefix='train'):
        xrays = batch['img']
        labels = batch['lab']
        domain_labels = batch['domain']

        features = self.model.get_features(xrays)
        logits = self.model.classifier(features)

        # task branch
        task_mask = domain_labels != self.target_domain_idx
        task_loss = self.criterion(logits[task_mask, :], labels[task_mask, :])

        # domain branch
        epoch_progress = self.trainer.current_epoch / self.trainer.max_epochs
        grl_alpha = 2 / (1 + np.exp(-10 * epoch_progress)) - 1
        features = grad_reverse(features, alpha=grl_alpha)
        dom_logits = self.domain_clf(features)
        dom_loss = F.cross_entropy(dom_logits, domain_labels)

        # output + logging
        loss = task_loss + self.domain_loss_weight * dom_loss
        if prefix=='train':
            self.log('grl_alpha', grl_alpha, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f'{prefix}_base_loss', task_loss, prog_bar=False)
        self.log(f'{prefix}_domain_loss', dom_loss, prog_bar=False)
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, prefix='train')

        # We don't want to store the entire training set logits and labels in memory,
        # so we log AUC every 200 batches instead
        if batch_idx % 200 == 0 and batch_idx > 0:
            train_labels = torch.cat(self.train_labels)
            train_logits = torch.cat(self.train_logits)

            self.log_auc(train_logits, train_labels, prefix='train')

            self.train_logits.clear()
            self.train_labels.clear()
        else:
            self.train_logits.append(logits)
            self.train_labels.append(labels)
        return loss

    def validation_step(self, batch):
        loss, logits, labels = self._step(batch, prefix='val')
        self.val_logits.append(logits)
        self.val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        val_logits = torch.cat(self.val_logits)
        val_labels = torch.cat(self.val_labels)

        self.log_auc(val_logits, val_labels, prefix='val')

        # Clear the lists for the next epoch
        self.val_logits.clear()
        self.val_labels.clear()