# adapted from https://github.com/DequanWang/tent/blob/master/tent.py
# added custom losses: sigmoid_softmax,  conj_pseudo_label
# more info on conjugate pseudo labels: https://papers.neurips.cc/paper_files/paper/2022/file/28e9eff897f98372409b40ae1ed3ea4c-Paper-Conference.pdf
from copy import deepcopy

import torch
import torch.jit
from torch import nn
import torch.nn.functional as F


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, tta_loss='softmax_entropy'):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.tta_loss = tta_loss

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.tta_loss)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.sigmoid() * torch.log(x.sigmoid())).mean(1)

def conj_pseudo_label(x, eps=1e-6, num_classes=5):
    """Conjunctive pseudo-labeling for model adaptation.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    softmax_prob = F.softmax(x, dim=1)
    smax_inp = softmax_prob

    eye = torch.eye(num_classes).to(x.device)
    eye = eye.reshape((1, num_classes, num_classes))
    eye = eye.repeat(x.shape[0], 1, 1)
    t2 = eps * torch.diag_embed(smax_inp)
    smax_inp = torch.unsqueeze(smax_inp, 2)
    t3 = eps * torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
    matrix = eye + t2 - t3
    y_star = torch.linalg.solve(matrix, smax_inp)
    y_star = torch.squeeze(y_star)

    pseudo_prob = y_star
    tta_loss = torch.logsumexp(x, dim=1) - (pseudo_prob * x - eps * pseudo_prob * (1 - softmax_prob)).sum(
        dim=1)
    return tta_loss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, tta_loss='softmax_entropy'):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    if tta_loss == 'softmax_entropy':
        loss = softmax_entropy(outputs).mean(0)
    elif tta_loss == 'sigmoid_entropy':
        loss = sigmoid_entropy(outputs).mean(0)
    elif tta_loss == 'conj_pseudo_label':
        loss = conj_pseudo_label(outputs).mean(0)
    else:
        raise ValueError(f"unknown tta_loss: {tta_loss}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"