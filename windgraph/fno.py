import torch
import torch.nn as nn


class NeuralOperator(nn.Module):
    """
    Implementation of Fourier Neural Operator
    https://arxiv.org/abs/2010.08895
    """

    def __init__(self):
        pass


class FourierLayer(nn.Module):
    def __init__(self, dv):
        self.dv = dv
        self.lin = nn.Linear(dv, dv)

    def forward(self, input):
        f = torch.fft.fft(input)
        f = f[:, : self.dv].transpose(1, 2)
        f = self.lin(f)
        return torch.fft.ifft(f, s=input.shape)
