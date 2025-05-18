import torch
import wandb
from torch import nn
import lightning as pl
from typing import Optional
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning import losses, miners
from wandb.plot import confusion_matrix

from hyda.utils import load_submodule_from_checkpoint
from ..data.dataset import DOMAIN_ENUM
from ..models import HyResNet26


class DomainConditionedDigitsLitModule(HyResNet26, pl.LightningModule):
    def __init__(self,
                 # domain branch
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,
                 target_domain: Optional[str] = None,
                 dom_clf_ckpt: Optional[str] = None,

                 # ResNet params
                 in_channels=1,
                 num_classes=5,

                 # Hypernet params
                 hyper_in_size=64,
                 hyper_emb=0,
                 init_method=None,
                 input_var=None,

                 lr: float=0.0001,
                 min_lr: float=1e-6,
                 w_decay: float=0,
                 msim_loss_weight: int=0,
                 use_aux_msim_loss: bool=False,
                 ):
        """
        :param lr: Adam learning rate
        :param w_decay: Adam weight decay
        """
        super().__init__(domain_encoder,domain_classifier, in_channels, num_classes,
                         hyper_in_size, hyper_emb, init_method, input_var)
        self.save_hyperparameters()
        self.automatic_optimization = False


        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 32, 32)

        # losses
        self.criterion = nn.CrossEntropyLoss()
        self.domain_loss = torch.nn.CrossEntropyLoss() # TODO: add ignore index=-1

        self.msim_loss_weight = msim_loss_weight
        self.use_aux_msim_loss = use_aux_msim_loss
        if self.msim_loss_weight > 0:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
            self.msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)
            if self.use_aux_msim_loss:
                self.aux_msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)


        # map GT classes to training classes
        self.target_domain = target_domain
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


        if dom_clf_ckpt is not None:
            print('loading from', dom_clf_ckpt)
            ckpt = torch.load(dom_clf_ckpt, map_location='cuda:0')
            self.domain_encoder = load_submodule_from_checkpoint(ckpt, 'encoder')
            self.domain_classifier = load_submodule_from_checkpoint(ckpt, 'classifier')

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

        # aggregators for logging metrics
        self.val_domain_labels = []
        self.val_domain_preds = []


    def configure_optimizers(self):
        digit_optimizer = torch.optim.AdamW(list(self.model.parameters()) +
                                          list(self.hyper_encode.parameters()),
                                          lr=self.lr, weight_decay=self.w_decay)
        cxr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(digit_optimizer, T_max=self.trainer.max_epochs,
                                                                  eta_min=self.min_lr)
        domain_optimizer = torch.optim.AdamW(list(self.domain_encoder.parameters()) +
                                            list(self.domain_classifier.parameters()) +
                                            list(self.hyper_encode.parameters()), # for aux msim loss
                                            lr=self.lr, weight_decay=self.w_decay)

        domain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=self.trainer.max_epochs,
                                                               eta_min=self.min_lr)
        return [digit_optimizer, domain_optimizer], [cxr_sched, domain_sched]

    def _step(self, batch):
        img, labels, domains = batch
        domains = self.gt2train_ids[domains]
        logits, dom_logits, h_emb, h_out = self(img)
        return logits, dom_logits, labels, domains, h_emb, h_out

    def training_step(self, batch, batch_idx):
        digit_opt, dom_opt = self.optimizers()
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        acc = self.train_accuracy(logits, labels)
        self.log('train_acc', acc)

        # cxr classification optimization
        digits_loss = self.criterion(logits, labels)
        self.log('train_loss', digits_loss, prog_bar=True)

        if len(h_out) > 0 and self.current_epoch > 1:
            weight_decay_factor = digit_opt.param_groups[0]['weight_decay']
            hyper_l2reg = weight_decay_factor * sum(
                [torch.linalg.vector_norm(w.flatten(1), dim=1) for w in h_out if w is not None]).mean()
            self.log("hyper_l2reg", hyper_l2reg, batch_size=logits.shape[0])
            digits_loss = digits_loss #+ hyper_l2reg

        digit_opt.zero_grad()
        self.manual_backward(digits_loss, retain_graph=True)
        digit_opt.step()


        # domain features optimization
        domain_loss = self.domain_loss(dom_logits, domains)
        if self.msim_loss_weight > 0:
            hard_pairs = self.miner(dom_logits, domains)
            msim_loss = self.msim_loss(dom_logits, domains, hard_pairs)

            # calculate auxillary msim loss on hyper embedding
            if self.use_aux_msim_loss:
                aux_hard_pairs = self.miner(h_emb, domains)
                aux_msim_loss = self.aux_msim_loss(h_emb, domains, aux_hard_pairs)
                msim_loss = msim_loss + aux_msim_loss
                self.log('train_aux_msim_loss', aux_msim_loss)
            domain_loss = domain_loss + msim_loss * self.msim_loss_weight
            self.log('train_msim_loss', msim_loss)
        self.log('train_domain_loss', domain_loss, prog_bar=True)

        dom_opt.zero_grad()
        self.manual_backward(domain_loss)
        dom_opt.step()

    def on_train_epoch_end(self):
        # lr schedulers update
        age_sch, dom_sch = self.lr_schedulers()
        age_sch.step()
        dom_sch.step()

    def validation_step(self, batch):
        cxr_opt, _ = self.optimizers()
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        acc = self.val_accuracy(logits, labels)
        self.log('val_acc', acc)

        # digit classification
        digit_loss = self.criterion(logits, labels)
        self.log('val_loss', digit_loss, prog_bar=True)

        if len(h_out) > 0:
            weight_decay_factor = cxr_opt.param_groups[0]['weight_decay']
            hyper_l2reg = weight_decay_factor * sum(
                [torch.linalg.vector_norm(w.flatten(1), dim=1) for w in h_out if w is not None]).mean()
            self.log("val_hyper_l2reg", hyper_l2reg, batch_size=logits.shape[0])


        # domain features
        domain_loss = self.domain_loss(dom_logits, domains)
        if self.msim_loss_weight > 0:
            hard_pairs = self.miner(dom_logits, domains)
            msim_loss = self.msim_loss(dom_logits, domains, hard_pairs)

            # calculate auxillary msim loss on hyper embedding
            if self.use_aux_msim_loss:
                aux_hard_pairs = self.miner(h_emb, domains)
                aux_msim_loss = self.aux_msim_loss(h_emb, domains, aux_hard_pairs)
                msim_loss = msim_loss + aux_msim_loss
                self.log('val_aux_msim_loss', aux_msim_loss)
            self.log('val_msim_loss', msim_loss)
            domain_loss = domain_loss + msim_loss * self.msim_loss_weight
        self.log('val_domain_loss', domain_loss, prog_bar=True)

        if self.current_epoch % 10 == 0:
            self.val_domain_preds.append(dom_logits.argmax(1).detach())
            self.val_domain_labels.append(domains.detach())


    def on_validation_epoch_end(self):
        if self.current_epoch % 10 == 0:
            # log confusion matrix
            val_domain_labels = torch.cat(self.val_domain_labels).cpu().numpy()
            val_domain_preds = torch.cat(self.val_domain_preds).cpu().numpy()

            if isinstance(self.logger, WandbLogger):
                wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

        # free memory
        self.val_domain_preds.clear()
        self.val_domain_labels.clear()