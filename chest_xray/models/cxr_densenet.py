import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchxrayvision.models import DenseNet


class CXRDenseNet(nn.Module):
    def __init__(self, imagenet_pretrained=True, num_classes=18):
        super().__init__()
        self.imagenet_pretrained = imagenet_pretrained
        self.num_classes = num_classes
        if self.imagenet_pretrained:
            # load imagenet pre-trained densenet121, replace input and output layers to suite CXR classification task
            self.model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
            self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.classifier = nn.Linear(1024, num_classes)
        else:
            self.model = DenseNet(num_classes=num_classes, apply_sigmoid=False)

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out

    def get_features(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def classifier(self, x):
        return self.model.classifier(x)