import torch
from torch import nn
import torch.nn.functional as F

class MultiLabelClassificationLoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for multi-label classification"""
    def __init__(self, weights=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        if weights is not None:
            weights = torch.tensor(weights).float()
            self.register_buffer('weights', weights)

    def forward(self, logits, labels):
        total_loss = torch.zeros(1, device=logits.device).float()
        for task in range(labels.shape[-1]):
            task_logits = logits[:, task]
            task_labels = labels[:, task]
            valid_mask = ~torch.isnan(task_labels)
            if valid_mask.any():
                task_labels = task_labels[valid_mask]
                task_logits = task_logits[valid_mask]
                loss = self.criterion(task_logits, task_labels)
                if hasattr(self, 'weights'):
                    loss = loss * self.weights[task]
                total_loss += loss
        return total_loss


