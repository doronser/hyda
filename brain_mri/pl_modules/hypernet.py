import torch
import pandas as pd
import wandb
from torch import nn
import lightning as pl
from typing import Optional, Any
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning import losses, miners
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, Accuracy
from torchmetrics.functional import mean_absolute_error
from wandb.plot import confusion_matrix

from brain_mri.data.dataset import DOMAIN_ENUM
from hyda.utils import load_submodule_from_checkpoint


class DomainConditionedBrainAgeLitModule(pl.LightningModule):
    def __init__(self,
                 age_encoder: nn.Module,
                 age_classifier: nn.Module,
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,
                 domain_conditioning_block: nn.Module,
                 lr: float=0.0001,
                 w_decay: float=0,
                 target_domain: Optional[str]=None,
                 dom_clf_ckpt: Optional[str]=None,
                 use_dom_clf_feats: bool=False,
                 freeze_domain_encoder: bool=False,
                 msim_loss_weight: int=0,
                 sanity_check = False):
        """
        :param lr: Adam learning rate
        :param w_decay: Adam weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr = lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 1, 90, 120, 99)
        self.criterion = torch.nn.MSELoss()
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.msim_loss_weight = msim_loss_weight
        if self.msim_loss_weight > 0:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
            self.msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)
            self.aux_msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)
        self.target_domain = target_domain
        self.use_dom_clf_feats = use_dom_clf_feats
        self.freeze_domain_encoder = freeze_domain_encoder
        self.sanity_check = sanity_check

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

        self.age_encoder = age_encoder
        self.age_classifier = age_classifier

        if dom_clf_ckpt is not None:
            print('loading from', dom_clf_ckpt)
            ckpt = torch.load(dom_clf_ckpt, map_location='cuda:0')
            self.domain_encoder = load_submodule_from_checkpoint(ckpt, 'encoder')
            self.domain_classifier = load_submodule_from_checkpoint(ckpt, 'classifier')
        else:
            self.domain_encoder = domain_encoder
            self.domain_classifier = domain_classifier
        if self.freeze_domain_encoder:
            for param in self.domain_encoder.parameters():
                param.requires_grad = False

        self.domain_conditioning_block = domain_conditioning_block

        num_domains = getattr(self.domain_classifier, 'num_classes', 20)
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_domains, top_k=1)
        self.train_r = PearsonCorrCoef()
        self.val_r = PearsonCorrCoef()
        self.val_acc = Accuracy(task="multiclass", num_classes=num_domains, top_k=1)

        # for logging val domain prediction confusion matrix
        self.val_domain_labels = []
        self.val_domain_preds = []

        # for logging test metrics
        self.test_age_preds = []
        self.test_age_gts = []

        self.apply(self.init_weights)

    def configure_optimizers(self):
        age_optimizer = torch.optim.AdamW(list(self.age_encoder.parameters()) +
                                          list(self.age_classifier.parameters()) +
                                          # list(self.domain_encoder.parameters()) +
                                          list(self.domain_conditioning_block.parameters()),
                                          lr=self.lr, weight_decay=self.w_decay)
        age_sched = torch.optim.lr_scheduler.CosineAnnealingLR(age_optimizer, T_max=self.trainer.max_epochs,
                                                                  eta_min=1e-6)
        if self.sanity_check:
            domain_optimizer = torch.optim.AdamW(list(self.age_encoder.parameters()),
                                                 lr=self.lr, weight_decay=self.w_decay)
        else:
            domain_optimizer = torch.optim.AdamW(list(self.domain_encoder.parameters()) +
                                                list(self.domain_classifier.parameters()) +
                                                list(self.domain_conditioning_block.hyper_encode.parameters()), # for aux msim loss
                                                lr=self.lr, weight_decay=self.w_decay)

        domain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=self.trainer.max_epochs,
                                                               eta_min=1e-6)
        return [age_optimizer, domain_optimizer], [age_sched, domain_sched]


    def _step(self, batch):
        scan, domain, age, age_bin, subject = batch
        age = age.float().unsqueeze(-1)
        domain = self.gt2train_ids[domain]  # map GT cls ids to training cls ids

        # extract features
        age_feats = self.age_encoder(scan)
        if self.sanity_check:
            domain_feats = torch.ones((scan.shape[0], 32), device=scan.device)
        else:
            domain_feats = self.domain_encoder(scan)
            if self.use_dom_clf_feats:
                domain_feats = self.domain_classifier.model[:-1](domain_feats.flatten(1))

        # conditioned age on domain
        dc_out = self.domain_conditioning_block(age_feats, domain_feats)

        # hypernetwork output handling
        h_out = ()
        h_emb = None
        if isinstance(dc_out, tuple):
            dc_out, h_emb, h_out = dc_out

        # predict age
        age_predicted = self.age_classifier(dc_out)

        # domain classification
        if self.use_dom_clf_feats and not self.sanity_check:
            domain_logits = self.domain_classifier.model[-1](domain_feats)
        else:
            domain_logits = self.domain_classifier(domain_feats.flatten(1))

        return age_predicted, domain_logits, age, domain, h_out, h_emb

    def training_step(self, batch):
        age_opt, dom_opt = self.optimizers()
        age_predicted, domain_logits, age, domain, h_out, h_emb = self._step(batch)


        # age regression step
        age_loss = self.criterion(age_predicted, age)
        self.log('train_loss', age_loss, prog_bar=True)
        if len(h_out) > 0:
            weight_decay_factor =age_opt.param_groups[0]['weight_decay']
            hyper_l2reg = weight_decay_factor * sum([torch.linalg.vector_norm(w.flatten(1), dim=1) for w in h_out if w is not None]).mean()
            self.log("hyper_l2reg", hyper_l2reg, batch_size=age.shape[0])
            age_loss = age_loss + hyper_l2reg

        age_opt.zero_grad()
        self.manual_backward(age_loss, retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        age_opt.step()

        # domain classification step
        if not self.sanity_check:
            domain_loss = self.domain_loss(domain_logits, domain)
            if self.msim_loss_weight > 0:
                hard_pairs = self.miner(domain_logits, domain)
                msim_loss = self.msim_loss(domain_logits, domain, hard_pairs)

                # calculate auxillary msim loss on hyper embedding
                aux_hard_pairs = self.miner(h_emb, domain)
                aux_msim_loss = self.aux_msim_loss(h_emb, domain, aux_hard_pairs)
                domain_loss = domain_loss + (msim_loss + aux_msim_loss) * self.msim_loss_weight
            self.log('train_domain_loss', domain_loss, prog_bar=True)

            dom_opt.zero_grad()
            self.manual_backward(domain_loss)
            dom_opt.step()

        # log metrics
        for metric_name, metric in [("mae", self.train_mae), ("r", self.train_r)]:
            metric(age_predicted.squeeze(), age.squeeze())
            self.log(f"train_{metric_name}", metric, prog_bar=metric_name=="mae")
        if not self.sanity_check:
            self.train_acc(domain_logits, domain)
            self.log("train_acc", self.train_acc)



    def on_train_epoch_end(self):
        # log epoch metric
        for metric_name, metric in [("mae", self.train_mae), ("r", self.train_r)]:
            self.log(f"train_{metric_name}_epoch", metric)

        # lr schedulers update
        age_sch, dom_sch = self.lr_schedulers()
        age_sch.step()
        dom_sch.step()

    def validation_step(self, batch):
        age_predicted, domain_logits, age, domain, h_out, h_emb = self._step(batch)

        # log loss
        age_loss = self.criterion(age_predicted, age)
        self.log('val_loss', age_loss, prog_bar=True)
        if len(h_out) > 0:
            age_opt, _ = self.optimizers()
            weight_decay_factor =age_opt.param_groups[0]['weight_decay']
            hyper_l2reg = weight_decay_factor * sum([torch.linalg.vector_norm(w.flatten(1), dim=1) for w in h_out if w is not None]).mean()
            self.log("val_hyper_l2reg", hyper_l2reg, batch_size=age.shape[0])


        if not self.sanity_check:
            domain_loss = self.domain_loss(domain_logits, domain)
            if self.msim_loss_weight > 0:
                hard_pairs = self.miner(domain_logits, domain)
                msim_loss = self.msim_loss(domain_logits, domain, hard_pairs)

                # calculate auxiliary msim loss on hyper embedding
                aux_hard_pairs = self.miner(h_emb, domain)
                aux_msim_loss = self.aux_msim_loss(h_emb, domain, aux_hard_pairs)
                domain_loss = domain_loss + (msim_loss + aux_msim_loss) * self.msim_loss_weight
            self.log('val_domain_loss', domain_loss, prog_bar=True)

        # log metrics
        for metric_name, metric in [("mae", self.val_mae), ("r", self.val_r)]:
            metric(age_predicted.squeeze(), age.squeeze())
            self.log(f"val_{metric_name}", metric, prog_bar=metric_name == "mae", on_step=True, on_epoch=False)
        if not self.sanity_check:
            self.val_acc(domain_logits, domain)
            self.val_domain_labels.append(domain.detach().cpu())
            self.val_domain_preds.append(domain_logits.argmax(1).detach().cpu())
            self.log("val_acc", self.val_acc)

    def on_validation_epoch_end(self):
        for metric_name, metric in [("mae", self.val_mae), ("r", self.val_r)]:
            self.log(f"val_{metric_name}_epoch", metric, on_epoch=True)
        if not self.sanity_check:
            self.log("val_acc_epoch", self.val_acc)
            if self.current_epoch % 10 == 0:
                # log confusion matrix
                val_domain_labels = torch.cat(self.val_domain_labels).numpy()
                val_domain_preds = torch.cat(self.val_domain_preds).cpu().numpy()

                if isinstance(self.logger, WandbLogger):
                    wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

            # free memory
            self.val_domain_preds.clear()
            self.val_domain_labels.clear()

    def test_step(self, batch):
        age_predicted, _, age, _, _, _ = self._step(batch)
        self.test_age_gts.append(age.squeeze(1).detach().cpu())
        self.test_age_preds.append(age_predicted.squeeze(1).detach().cpu())

    def on_test_epoch_end(self):
        print([x.shape for x in self.test_age_gts])
        ages = torch.cat(self.test_age_gts).cpu().float()
        age_preds = torch.cat(self.test_age_preds).cpu().float()

        test_df = pd.DataFrame(dict(age_pred=age_preds.numpy(), age=ages.numpy()))
        test_df["abs_err"] = abs(test_df["age_pred"] - test_df["age"])
        self.logger.log_table("test_df", dataframe=test_df)

        test_mae = mean_absolute_error(age_preds, ages)
        self.log("test_mae", test_mae)

    def forward(self, x, predict_domain=False) -> Any:
        # extract features
        age_feats = self.age_encoder(x)
        if self.sanity_check:
            domain_feats = torch.ones((x.shape[0], 32), device=x.device)
        else:
            domain_feats = self.domain_encoder(x)
            if self.use_dom_clf_feats and not self.sanity_check:
                domain_feats = self.domain_classifier.model[:-1](domain_feats.flatten(1))

        # predict age (conditioned on domain)
        dc_out = self.domain_conditioning_block(age_feats, domain_feats)
        if isinstance(dc_out, tuple):
            dc_out, h_emb, h_out = dc_out
        age_pred = self.age_classifier(dc_out)

        if predict_domain:
            if self.use_dom_clf_feats:
                domain_logits = self.domain_classifier.model[-1](domain_feats)
            else:
                domain_logits = self.domain_classifier(domain_feats.flatten(1))
            return age_pred, domain_logits
        else:
            return age_pred

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

