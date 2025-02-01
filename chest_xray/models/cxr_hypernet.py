from torch import nn
import torchxrayvision as xrv
from torchxrayvision.models import DenseNet

from hyda.layers import HyperLinearBMM, HyperGroupedConv


class HyDenseNet(DenseNet):
    def __init__(self,
                 # DenseNet params
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=len(xrv.datasets.default_pathologies),
                 in_channels=1,
                 weights=None,
                 cache_dir=None,
                 op_threshs=None,
                 apply_sigmoid=False,

                 # HyperNet params
                 hyper_in_size=32,
                 hyper_emb=0,

                 ):
        super().__init__(growth_rate,
                         block_config,
                         num_init_features,
                         bn_size,
                         drop_rate,
                         num_classes,
                         in_channels,
                         weights,
                         cache_dir,
                         op_threshs,
                         apply_sigmoid)

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
        self.classifier = HyperLinearBMM(in_features=self.classifier.in_features,
                                         out_features=self.classifier.out_features,
                                         hyper_size=32,
                                         )

    def forward(self, x):
        pass