from torch import nn

from hyda.layers import HyperLinearBMM

class BrainAgeHypernetwork(nn.Module):
    def __init__(self, in_features=1024,
                 hyper_in_size=32,
                 dropout=0.1,
                 activation_layer='LeakyReLU',
                 layers=(1024, 256, 64),
                 use_hyper=(False, True, True, False),
                 hyper_emb=16,
                 init_method=None,
                 input_var=None,
                 normalize_weights=False
                 ):
        super().__init__()
        self.hyper_emb = hyper_emb
        act_cls = getattr(nn, activation_layer)
        self.activation = act_cls()
        if self.hyper_emb > 0:
            self.hyper_encode = nn.Sequential(nn.Linear(hyper_in_size, self.hyper_emb),
                                              act_cls())
        else:
            # HACK: use domain features directly
            self.hyper_encode = nn.Identity()
            self.hyper_emb = hyper_in_size

        self.layers = nn.ModuleList()
        in_feats = in_features
        layers.append(1) # add output layer
        for i, out_feats in enumerate(layers):
            if use_hyper[i]:
                self.layers.append(HyperLinearBMM(hyper_size=self.hyper_emb, in_features=in_feats, out_features=out_feats,
                                                  init_method=init_method, input_var=input_var, normalize_weights=normalize_weights))
            else:
                self.layers.append(nn.Linear(in_features=in_feats, out_features=out_feats))
            in_feats = out_feats
        self.dropout = nn.Dropout(dropout)

    def forward(self, age_feats, dom_feats):
        h_in = self.hyper_encode(dom_feats.flatten(1))
        out = age_feats.flatten(1)
        h_out = ()

        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, HyperLinearBMM):
                out, h = layer(out, h_in)
                h_out += h
            else:
                out = layer(out)
            out = self.activation(out)
            if layer_idx == 0:
                out = self.dropout(out)
        return out, h_in, h_out