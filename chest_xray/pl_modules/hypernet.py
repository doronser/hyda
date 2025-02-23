import torch
import wandb
from torch import nn
import lightning as pl
from typing import Optional
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning import losses, miners
from torchmetrics.functional.classification import multilabel_auroc
from torchxrayvision.models import DenseNet
from wandb.plot import confusion_matrix

from hyda.utils import load_submodule_from_checkpoint
from ..data.dataset import DOMAIN_ENUM
from ..models.cxr_hypernet import HyDenseNet
from ..models.loss import MultiLabelClassificationLoss


class DomainConditionedCXRLitModule(HyDenseNet, pl.LightningModule):
    def __init__(self,
                 # domain branch
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,
                 target_domain: Optional[str] = None,
                 dom_clf_ckpt: Optional[str] = None,

                 # DenseNet params
                 imagenet_pretrained=True,
                 num_classes=5,
                 task_weights: list = None,

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
                 sanity_check = False):
        """
        :param lr: Adam learning rate
        :param w_decay: Adam weight decay
        """
        super().__init__(domain_encoder,domain_classifier, imagenet_pretrained, num_classes,
                         hyper_in_size, hyper_emb, init_method, input_var)
        self.save_hyperparameters()
        self.sanity_check = sanity_check
        self.automatic_optimization = False


        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 224, 224)

        #save class list
        if self.num_classes == 5: # intersection of all pathologies
            self.pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Pneumothorax']
        elif self.num_classes == 18: # union of all pathologies
            self.pathologies = DenseNet.targets
        else:
            raise ValueError('num_classes must be either 5 or 18 (intersection/union of all pathologies)')


        # losses
        self.criterion = MultiLabelClassificationLoss(weights=task_weights)
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

        # self.model = HyDenseNet(domain_encoder=domain_encoder,
        #                         domain_classifier=domain_classifier,
        #                         imagenet_pretrained=imagenet_pretrained,
        #                         num_classes=num_classes,
        #                         hyper_in_size=hyper_in_size,
        #                         hyper_emb=hyper_emb,
        #                         )

        # aggregators for logging metrics
        self.train_logits = []
        self.train_labels = []
        self.val_logits = []
        self.val_labels = []
        self.val_domain_labels = []
        self.val_domain_preds = []

        # for logging test metrics
        self.test_age_preds = []
        self.test_age_gts = []

    def log_auc(self, logits, labels, prefix='train'):
        valid_classes = ~torch.isnan(labels).all(0)
        valid_logits = logits[:, valid_classes]
        valid_labels = labels[:, valid_classes]
        auc = multilabel_auroc(valid_logits, torch.nan_to_num(valid_labels, nan=100).long(),
                               num_labels=valid_classes.sum().item(), ignore_index=100, average='none')
        valid_pathos = [self.pathologies[i] for i, valid_cls in enumerate(valid_classes) if valid_cls]
        for i, auc_val in enumerate(auc):
            self.log(f'{prefix}_AUC/{valid_pathos[i]}', auc_val.item(), prog_bar=False)
        self.log(f'{prefix}_AUC', auc.mean().item(), prog_bar=prefix=='train')

    def configure_optimizers(self):
        cxr_optimizer = torch.optim.AdamW(list(self.model.features.parameters()) +
                                          list(self.model.classifier.parameters()) +
                                          list(self.hyper_encode.parameters()),
                                          lr=self.lr, weight_decay=self.w_decay)
        cxr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cxr_optimizer, T_max=self.trainer.max_epochs,
                                                                  eta_min=self.min_lr)
        domain_optimizer = torch.optim.AdamW(list(self.domain_encoder.parameters()) +
                                            list(self.domain_classifier.parameters()) +
                                            list(self.hyper_encode.parameters()), # for aux msim loss
                                            lr=self.lr, weight_decay=self.w_decay)

        domain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=self.trainer.max_epochs,
                                                               eta_min=self.min_lr)
        return [cxr_optimizer, domain_optimizer], [cxr_sched, domain_sched]

    def _step(self, batch):
        xray = batch['img']
        domains = self.gt2train_ids[batch['domain']]
        labels = batch['lab']

        logits, dom_logits, h_emb, h_out = self(xray)

        return logits, dom_logits, labels, domains, h_emb, h_out

    def training_step(self, batch, batch_idx):
        cxr_opt, dom_opt = self.optimizers()
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        # cxr classification optimization
        cxr_loss = self.criterion(logits, labels)

        self.log('train_loss', cxr_loss, prog_bar=True)
        if len(h_out) > 0 and self.current_epoch > 1:
            weight_decay_factor = cxr_opt.param_groups[0]['weight_decay']
            hyper_l2reg = weight_decay_factor * sum(
                [torch.linalg.vector_norm(w.flatten(1), dim=1) for w in h_out if w is not None]).mean()
            self.log("hyper_l2reg", hyper_l2reg, batch_size=logits.shape[0])
            cxr_loss = cxr_loss #+ hyper_l2reg

        cxr_opt.zero_grad()
        self.manual_backward(cxr_loss, retain_graph=True)
        cxr_opt.step()


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
            domain_loss = domain_loss + msim_loss * self.msim_loss_weight
            self.log('train_msim_loss', msim_loss)
        self.log('train_domain_loss', domain_loss, prog_bar=True)

        dom_opt.zero_grad()
        self.manual_backward(domain_loss)
        dom_opt.step()

        # We don't want to store the entire training set logits and labels in memory,
        # so we log AUC every 200 batches instead
        if batch_idx % 200 == 0 and batch_idx > 0:
            train_labels = torch.cat(self.train_labels)
            train_logits = torch.cat(self.train_logits)

            self.log_auc(train_logits, train_labels, prefix='train')

            self.train_logits.clear()
            self.train_labels.clear()
        else:
            self.train_logits.append(logits.detach())
            self.train_labels.append(labels.detach())

    def on_train_epoch_end(self):
        # lr schedulers update
        age_sch, dom_sch = self.lr_schedulers()
        age_sch.step()
        dom_sch.step()

    def validation_step(self, batch):
        cxr_opt, _ = self.optimizers()
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        # cxr classification
        cxr_loss = self.criterion(logits, labels)
        self.log('val_loss', cxr_loss, prog_bar=True)
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
            domain_loss = domain_loss + msim_loss * self.msim_loss_weight
        self.log('val_domain_loss', domain_loss, prog_bar=True)

        self.val_logits.append(logits.detach())
        self.val_labels.append(labels.detach())
        if self.current_epoch % 10 == 0:
            self.val_domain_preds.append(dom_logits.argmax(1).detach())
            self.val_domain_labels.append(domains.detach())


    def on_validation_epoch_end(self):
        val_logits = torch.cat(self.val_logits)
        val_labels = torch.cat(self.val_labels)
        self.log_auc(val_logits, val_labels, prefix='val')

        if self.current_epoch % 10 == 0:
            # log confusion matrix
            val_domain_labels = torch.cat(self.val_domain_labels).cpu().numpy()
            val_domain_preds = torch.cat(self.val_domain_preds).cpu().numpy()

            if isinstance(self.logger, WandbLogger):
                wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

        # free memory
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_domain_preds.clear()
        self.val_domain_labels.clear()