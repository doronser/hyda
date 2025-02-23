import torch
from torch import nn
import torch.nn.functional as F

from hyda.layers import HyperLinearBMM
from ..models import CXRDenseNet

class HyDenseNet(CXRDenseNet):
    def __init__(self,
                 # domain branch params
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,

                 # DenseNet params
                 imagenet_pretrained=True,
                 num_classes=5,

                 # Hypernet params
                 hyper_in_size=64,
                 hyper_emb=0,
                 init_method=None,
                 input_var=None,

                 ):
        super().__init__(imagenet_pretrained, num_classes=num_classes)
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


        # overwrite existing classifier with a hyper layer
        self.model.classifier = HyperLinearBMM(in_features=1024, out_features=num_classes, hyper_size=32,
                                               init_method=init_method, input_var=input_var)

    def forward(self, x):
        """

        :param x: input tensor
        :return: tuple of primary output, hypernet input and hypernet output
        """
        # create pathology features
        path_feats = self.model.features(x)
        path_feats = F.relu(path_feats, inplace=True)
        path_feats = F.adaptive_avg_pool2d(path_feats, (1, 1))
        path_feats = torch.flatten(path_feats, 1)

        # create domain features
        dom_feats = self.domain_encoder(x)
        dom_feats = self.domain_classifier.model[:-1](dom_feats)
        dom_logits = self.domain_classifier.model[-1](dom_feats)

        # domain conditioning via hypernet
        h_in = self.hyper_encode(dom_feats)
        logits, h_out = self.model.classifier(path_feats, h_in)
        return logits, dom_logits, h_in, h_out