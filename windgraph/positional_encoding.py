from math import pi, sqrt
from math import tau as two_pi

import torch
import torch.nn as nn


class SinCosPositionalEncoding(nn.Module):
    """ Sine-Cosine Positional Encoding à la *Attention is all you need* [1].

    The encodings are given by :

    [..., cos(2π σ^(j/n) x), sin(2π σ^(j/n) x), ...], j = 0,...,m-1
    
    Where σ should be grid searched for all dataset as suggested here [2]
    [1] https://arxiv.org/abs/1706.03762
    [2] https://arxiv.org/abs/2006.10739 p.7
    """

    def __init__(self, sigma, n):
        super().__init__()

        j = torch.arange(n).repeat_interleave(2).float() / n
        phase = (torch.arange(2 * n) % 2) * (pi / 2)

        self.register_buffer("sigmas", sigma ** j)
        self.register_buffer("phase", phase)

    def forward(self, x):
        x = x.float()
        pe = torch.cos(two_pi * self.sigmas * x[..., None] + self.phase)
        scale = sqrt(1 / (2 * len(self.sigmas)))
        return pe * scale
