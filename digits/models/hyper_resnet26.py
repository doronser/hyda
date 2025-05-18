import torch
from torch import nn
import torch.nn.functional as F

from hyda.layers import HyperLinearBMM
from ..models import DigitsResNet26

class HyResNet26(DigitsResNet26):
    def __init__(self,
                 # domain branch params
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,

                 in_channels=1 ,
                 num_classes=10,

                 # hypernet params
                 hyper_in_size=64,
                 hyper_emb=0,
                 init_method=None,
                 input_var=None,
                 ):
        super().__init__(in_channels, num_classes)
        self.num_classes = num_classes
        self.domain_encoder = domain_encoder
        self.domain_classifier = domain_classifier
        self.hyper_emb = hyper_emb
        self.hyper_in_size = hyper_in_size
        if self.hyper_emb > 0:
            self.hyper_encode = nn.Sequential(
                nn.Linear(self.hyper_in_size, self.hyper_emb),
                nn.ReLU(),
            )
        else:
            self.hyper_encode = nn.Identity()
            self.hyper_emb = hyper_in_size

        self.model.fc = HyperLinearBMM(in_features=self.model.fc.in_features, out_features=num_classes,
                                       hyper_size=self.hyper_emb, init_method=init_method, input_var=input_var)

    def forward(self, x):
        """

        :param x: input tensor
        :return: tuple of primary output, hypernet input and hypernet output
        """
        # create pathology features
        feats = self.get_features(x)
        feats = torch.flatten(feats, 1)

        # create domain features
        dom_feats = self.domain_encoder(x)
        dom_feats = self.domain_classifier.model[:-1](dom_feats)
        dom_logits = self.domain_classifier.model[-1](dom_feats)

        # domain conditioning via hypernet
        h_in = self.hyper_encode(dom_feats)
        logits, h_out = self.model.fc(feats, h_in)
        return logits, dom_logits, h_in, h_out