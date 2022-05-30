import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_inx, d_iny, d_tx, d_out, d_model, nheads=4, nlayers=2):
        super().__init__()

        self.c_encoder = nn.Linear(d_inx + d_iny, d_model)
        self.t_encoder = nn.Linear(d_tx, d_model)

        self.transformer = nn.Transformer(
            d_model, nheads, nlayers, nlayers, d_model, 0.1, batch_first=True
        )
        self.decoder = nn.Linear(d_model, d_out)

    def forward(self, cx, cy, tx):
        inputs = self.c_encoder(torch.cat((cx, cy), dim=-1))
        hidden = self.transformer(inputs, self.t_encoder(tx))
        return self.decoder(hidden)
