import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional.classification import multilabel_auroc

from chest_xray.data import CXRDataModule
from chest_xray.pl_modules import CXRLitModule
from chest_xray.models import CXRDenseNet
from chest_xray.models.tent import Tent, collect_params, configure_model

def eval_tent(dataloader, ckpt_path, steps=10, episodic=True, lr=1e-3, device='cpu'):
    lm = CXRLitModule.load_from_checkpoint(ckpt_path, model=CXRDenseNet(num_classes=5),
                                           task_weights=[1]*18, map_location=device)
    lm.eval()
    # baseline evaluation
    labels = []
    orig_outputs = []
    for batch in tqdm(dataloader, ncols=100, desc='baseline inference'):
        xrays = batch['img'].to(device)
        out = lm.model(xrays)

        labels.append(batch['lab'])
        orig_outputs.append(out.detach().cpu())

    # tent adaptation
    model = configure_model(lm.model)
    params, param_names = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    tented_model = Tent(model, optimizer, steps=steps, episodic=episodic)

    tented_model = tented_model.to(device)
    tent_outputs = []
    for batch in tqdm(dataloader, ncols=100, desc='TENT inference'):
        xrays = batch['img'].to(device)
        out = tented_model(xrays)
        tent_outputs.append(out.detach().cpu())

    orig_outputs = torch.concat(orig_outputs)
    tent_outputs = torch.concat(tent_outputs)
    labels = torch.concat(labels)

    valid_classes = ~torch.isnan(labels).all(0)
    base_logits = orig_outputs[:, valid_classes]
    tent_logits = tent_outputs[:, valid_classes]
    valid_labels = torch.nan_to_num(labels[:, valid_classes], nan=100).long()
    base_auc = multilabel_auroc(base_logits, valid_labels, num_labels=valid_classes.sum().item(), ignore_index=100,
                                average='none')
    tent_auc = multilabel_auroc(tent_logits, valid_labels, num_labels=valid_classes.sum().item(), ignore_index=100,
                                average='none')
    df = pd.DataFrame(dict(baseline=base_auc, tent=tent_auc), index=nih_ds.pathologies).T
    return df

if __name__ == '__main__':
    # load data
    base_dir = '<PLACEHOLDER_DATA_DIR>'
    dm = CXRDataModule(data_dir=base_dir, target_domain=None, batch_size=32, num_workers=16, unique_patients=False)
    dm.setup()
    nih_ds, chx_ds, vin_ds = dm.val_dataset.datasets
    nih_dl = DataLoader(nih_ds, batch_size=32, num_workers=8, shuffle=False)
    chx_dl = DataLoader(chx_ds, batch_size=32, num_workers=8, shuffle=False)
    vin_dl = DataLoader(vin_ds, batch_size=32, num_workers=8, shuffle=False)


    # model ckpts
    nih_ckpt = '<PLACEHOLDER_CKPT_DIR>/cxr_baseline_nih.ckpt'
    chx_ckpt = '<PLACEHOLDER_CKPT_DIR>/cxr_baseline_chexpert.ckpt'
    vin_ckpt = '<PLACEHOLDER_CKPT_DIR>/cxr_baseline_vin.ckpt'
    tent_dict = dict(NIH=[nih_dl, nih_ckpt],
                     CheXpert=[chx_dl, chx_ckpt],
                     VinDr=[vin_dl, vin_ckpt])

    # evaluate
    dfs = []
    for ds_name, ds_args in tent_dict.items():
        df = eval_tent(*ds_args, steps=10, episodic=False, lr=1e-4, device='cuda:0')
        df['dataset'] = ds_name
        dfs.append(df)

    df = pd.concat(dfs)[['dataset', *nih_ds.pathologies]]
    df.to_csv('tent_eval.csv')
    print(df)
