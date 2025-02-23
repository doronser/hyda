import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class HyperLinearBMM(nn.Module):
    def __init__(self,
                 hyper_size: int,
                 in_features: int,
                 out_features: int,
                 normalize_weights: bool = True,
                 init_method = 'uniform',
                 input_var = 0.0847 # input MRI var: 0.0847 domain embeddings var: 44.6123
                 ):
        """hypernetwork for a single Linear layer using batch matrix multiplication (BMM) to support batch_size >1


        The hypernetwork generated a set of weights and biases for every sample in the batch.
        Weight generation:
            1. Linear layer: (batch_size, input_size) -> (batch_size, out_features*in_features)
            2. Reshape to (batch_size, out_features, in_features)

        Bias generation:
            1. Linear layer: (batch_size,  input_size) ->(batch_size, out_features)
            2. Reshape to (batch_size, out_features, 1)

        With this set of W&B, the final output is calculated using BMM:
        BMM((batch_size, out_features, in_features), (batch_size, in_features, 1)) + (batch_size, out_features, 1)


        :param hyper_size: size of input to hypernetwork FC layer
        :param in_features: nn.Linear param
        :param out_features: nn.Linear param
        :param normalize_weights: if True, normalize weights relative to input variance
        """
        super().__init__()
        self.input_size = hyper_size
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_weights = normalize_weights
        self.init_method = init_method
        self.input_var = input_var

        self.hyper_w_layer = nn.Linear(self.input_size,  self.in_features*self.out_features)
        self.hyper_b_layer = nn.Linear(self.input_size,  self.out_features)

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.hyper_w_layer.bias.data, 0)
        nn.init.constant_(self.hyper_b_layer.bias.data, 0)

        if self.input_var is not None:
            # calc weight and bias variance given input variance
            # based on 'principled weight initialization for hypernetworks' by Lipson et al.
            main_net_relu = True
            main_net_biasses = True
            dk = self.input_size  # hypernet_input_size  # both dk and dl
            dj = self.in_features # main net input size
            w_var = (2 ** main_net_relu) / ((2 ** main_net_biasses) * dj * dk * self.input_var)
            b_var = (2 ** main_net_relu) / (2 * dk * self.input_var)


            if self.init_method == 'uniform':
                w_var = np.sqrt(3 * w_var)
                b_var = np.sqrt(3 * b_var)
                nn.init.uniform_(self.hyper_w_layer.weight, -w_var, w_var)
                nn.init.uniform_(self.hyper_b_layer.weight, -b_var, b_var)

            elif self.init_method == 'normal':
                nn.init.normal_(self.hyper_w_layer.weight, 0, w_var)
                nn.init.normal_(self.hyper_b_layer.weight, 0, b_var)

    def forward(self, x, hyper_emb, use_bmm=True):
        w, b = self.hyper_forward(hyper_emb)
        if self.normalize_weights:
            w = F.tanh(w) * 5
            b = F.tanh(b) * 5
        if use_bmm:
            fc_out = torch.bmm(w, x.unsqueeze(-1)) + b
        else:
            fc_out = torch.empty((x.shape[0], 1, w.shape[1]), device=x.device)
            for i in range(x.shape[0]):
                fc_out[i] = F.linear(x[i].unsqueeze(0), weight=w[i], bias=b[i].T)
        return fc_out.squeeze(-1), (w,b)

    def hyper_forward(self, hyper_emb):
        """generate weights and bias for a Linear layer"""
        w = self.hyper_w_layer(hyper_emb).reshape(-1, self.out_features, self.in_features)
        b = self.hyper_b_layer(hyper_emb.unsqueeze(1)).transpose(-2,-1)
        return w, b


class HyperGroupedConv(nn.Module):
    def __init__(self, hyper_size: int, batch_size: int,
                 in_channels: int, out_channels: int,
                 ndims = 2, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        """hypernetwork conv layer with grouped conv for batch dimension.
        A naive hyper conv layer would generate a single set of weights for all samples in the batch.
        This layer generates a set of weights for each sample in the batch.

        :param hyper_size: size of input to hypernetwork FC layer
        :param batch_size: batch size of conv layer input (needed for group conv)
        :param in_channels: conv layer param
        :param out_channels: conv layer param
        :param ndims: conv layer param
        :param kernel_size: conv layer param
        :param stride: conv layer param
        :param padding: conv layer param
        """
        super().__init__()
        self.hyper_size = hyper_size
        self.batch_size = batch_size
        self.ndims = ndims

        self.conv = F.conv2d if ndims == 2 else F.conv3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # for grouped conv layer
        self.w_size = batch_size * out_channels * in_channels * (kernel_size ** self.ndims)
        self.b_size = out_channels
        self.hyper_w_layer = nn.Linear(self.hyper_size, self.w_size)
        # self.hyper_b_layer = nn.Linear(self.hyper_size, self.b_size)
        print(f'HyperGroupedConv: w_size={self.w_size}, b_size={self.b_size}')

    def forward(self, x):
        x_in, h = x
        if self.ndims==2:
            B, C, H, W = x_in.size()
            x_in = x_in.view(1, B*C, H, W)
        else:
            B, C, D, H, W = x_in.size()
            x_in = x_in.view(1, B*C, D, H, W)

        w, b = self.hyper_forward(h)
        out = self.conv(x_in, weight=w, bias=None, groups=B,
                        stride=self.stride, padding=self.padding)
        if self.ndims == 2:
            out = out.view(self.batch_size, -1, out.shape[-2], out.shape[-1])
        else:
            out = out.view(self.batch_size, -1,  out.shape[-3], out.shape[-2], out.shape[-1])
        return out, (w,b)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None # return same num of inputs to forward

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)