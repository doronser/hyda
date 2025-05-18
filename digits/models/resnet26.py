import timm
from torch import nn


class DigitsResNet26(nn.Module):
    def __init__(self, in_channels=1 , num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('resnet26', pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels, self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding,
                                     bias=self.model.conv1.bias is not None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        return self.model.forward_features(x)