import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from inspect import isfunction
from collections import deque
import copy
from types import SimpleNamespace
import scipy.io as scio

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


def clones(module, N):
    """Cloning model blocks"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DenseLayer(nn.Module):

    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(c_in=c_in + layer_idx * growth_rate,
                           # Input channels are original plus the feature maps from previous layers
                           bn_size=bn_size,
                           growth_rate=growth_rate,
                           act_fn=act_fn)
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(nn.Module):  # TODO data consistency

    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, in_filters=16, num_layers=[4, 4, 4, 4], bn_size=2, growth_rate=16,  data_path=None, act_fn_name="relu", **kwargs):
        super().__init__()
        self.data_path = data_path
        self.hparams = SimpleNamespace(in_filters=in_filters,
                                       num_layers=num_layers,
                                       bn_size=bn_size,
                                       growth_rate=growth_rate,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.in_filters  # The start number of hidden channels
        self.conv = clones(nn.Conv1d(in_channels=1, out_channels=c_hidden, kernel_size=3, padding=1, bias=False), 5)
        self.denseblock = clones(DenseBlock(c_in=c_hidden,
                                            num_layers=8,
                                            bn_size=self.hparams.bn_size,
                                            growth_rate=self.hparams.growth_rate,
                                            act_fn=self.hparams.act_fn), 5)
        self.transitionlayer = clones(
            TransitionLayer(c_in=c_hidden + 8 * self.hparams.growth_rate, c_out=1, act_fn=self.hparams.act_fn), 5)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, xx_real, xx_imag):
        y = torch.complex(xx_real, xx_imag)
        for block_idx, num_layers in enumerate(self.hparams.num_layers):
            x = self.conv[block_idx](x)
            x = self.denseblock[block_idx](x)
            x = self.transitionlayer[block_idx](x)
            # data consistency
            x = torch.fft.ifft(x, dim=-1)
            x[torch.nonzero(y, as_tuple=True)] = (x[torch.nonzero(y, as_tuple=True)]
                                                          + 1e3 * y[
                                                              torch.nonzero(y, as_tuple=True)]) / (
                                                                 1 + 1e3)
            x = torch.fft.fft(x, dim=-1)
            x = torch.abs(x)
        return x