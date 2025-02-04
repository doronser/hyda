import torch
import wandb
from torch import nn
import lightning as pl
from typing import Optional
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning import losses, miners
from wandb.plot import confusion_matrix

from ..data.dataset import DOMAIN_ENUM


class DomainClassifier(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 target_domain: Optional[str]=None,
                 lr: float=0.0001,
                 w_decay: float=0,
                 alpha=1,
                 msim_on_embeddings=False):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.classifier = classifier
        self.target_domain = target_domain
        self.lr = lr
        self.w_decay = w_decay
        self.alpha= alpha
        self.msim_on_embeddings = msim_on_embeddings

        # for model architecture logging
        self.example_input_array = torch.Tensor(1, 1, 224, 224)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)

        # map GT classes to training classes
        gt2train_ids = {}
        self.cls_names = []
        curr_idx = 0
        for k, v in DOMAIN_ENUM.items():
            if k != self.target_domain:
                gt2train_ids[v] = curr_idx
                curr_idx += 1
                self.cls_names.append(k)
            else:
                gt2train_ids[v] = -1
        gt2train_ids_tensor = torch.tensor([gt2train_ids[k] for k in sorted(gt2train_ids.keys())])
        self.register_buffer("gt2train_ids", gt2train_ids_tensor)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.classifier.num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.classifier.num_classes, top_k=1)

        # for logging val domain prediction confusion matrix
        self.val_domain_labels = []
        self.val_domain_preds = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) +
                                      list(self.classifier.parameters()),
                                      lr=self.lr, weight_decay=self.w_decay)
        return optimizer

    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        if return_feats:
            feats = self.classifier.model[:-1](feats)
            return feats
        else:
            domain_logits = self.classifier.model(feats)
            return domain_logits

    def _step(self, batch, return_feats=False):
        img = batch['img']
        domain = batch['domain']
        domain_labels = self.gt2train_ids[domain]
        domain_out = self(img, return_feats)
        return domain_out, domain_labels

    def training_step(self, batch):
        if self.msim_on_embeddings:
            domain_feats, domain_labels = self._step(batch, return_feats=True)
            domain_logits = self.classifier.model[-1](domain_feats)
            msim_feats = domain_feats
        else:
            domain_logits, domain_labels = self._step(batch)
            msim_feats = domain_logits

        # domain classification step
        clf_loss = self.criterion(domain_logits, domain_labels)
        self.log('train_clf_loss', clf_loss, prog_bar=False)


        hard_pairs  = self.miner(msim_feats, domain_labels)
        msim_loss = self.msim_loss(msim_feats, domain_labels, hard_pairs)
        self.log('train_msim_loss', msim_loss, prog_bar=False)

        # log metrics
        loss = clf_loss + msim_loss * self.alpha
        self.train_acc(domain_logits, domain_labels)
        self.log("train_acc", self.train_acc)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        if self.msim_on_embeddings:
            domain_feats, domain_labels = self._step(batch, return_feats=True)
            domain_logits = self.classifier.model[-1](domain_feats)
            msim_feats = domain_feats
        else:
            domain_logits, domain_labels = self._step(batch, return_feats=False)
            msim_feats = domain_logits

        # log loss
        clf_loss = self.criterion(domain_logits, domain_labels)
        self.log('val_clf_loss', clf_loss, prog_bar=True)

        hard_pairs = self.miner(msim_feats, domain_labels)
        msim_loss = self.msim_loss(msim_feats, domain_labels, hard_pairs)
        self.log('val_msim_loss', msim_loss, prog_bar=True)

        # log metrics
        loss = clf_loss + msim_loss
        self.val_acc(domain_logits, domain_labels)
        self.val_domain_labels.append(domain_labels)
        self.val_domain_preds.append(domain_logits.argmax(1))
        self.log("val_acc", self.val_acc)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc)

        # log confusion matrix
        val_domain_labels = torch.cat(self.val_domain_labels).cpu().numpy()
        val_domain_preds = torch.cat(self.val_domain_preds).cpu().numpy()

        if isinstance(self.logger, WandbLogger):
            wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

        # free memory
        self.val_domain_labels.clear()
        self.val_domain_preds.clear()