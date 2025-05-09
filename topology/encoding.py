import math

import numpy as np
import torch
import torch.nn as nn

from torch_topological.nn.data import make_tensor
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn.layers import StructureElementLayer


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class Topo(nn.Module):
    def __init__(self, dims=0):
        super(Topo, self).__init__()
        self.vr = VietorisRipsComplex(dim=dims)
        self.pl = StructureElementLayer(n_elements=64)
        self.relu = nn.ReLU()
    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=1) # H, 1, V
        return weight_norm

    def forward(self, x):
        x = x.mean(1)
        x = x.unsqueeze(-1) - x.unsqueeze(-2)
        x = x.mean(-3)
        x = self.L2_norm(x)
        x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
        x = self.vr(x)
        x = make_tensor(x)
        x = self.pl(x)
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1
        self.fc1 = nn.Parameter(torch.stack([torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0) for _ in range(3)], dim=0), requires_grad=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        # # k-hop
        h1 = A.sum(0)
        h1[h1 != 0] = 1
       

        h = [None for _ in range(A.shape[-1])]
        h[0] = np.eye(A.shape[-1])
        h[1] = h1
        self.hops = 0*h[0]
        for i in range(2, A.shape[-1]):
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(A.shape[-1]-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()

        # hop connection
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, self.hops.max() + 1,)))

        self.in_channels = in_channels
        self.hidden_channels = in_channels if in_channels > 3 else 64

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


