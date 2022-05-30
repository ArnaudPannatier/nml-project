from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class GraphNetBlock(nn.Module):
    def __init__(self, dim=128, l=2):
        super().__init__()
        self.message = MLP(2 * dim, dim, dim, l)
        self.node = MLP(2 * dim, dim, dim, l)

    def forward(self, nodes, senders, receivers):
        messages = self.message(
            torch.cat((nodes[:, receivers], nodes[:, senders]), dim=-1)
        )
        inbox = torch.zeros(nodes.shape, device=nodes.device)
        inbox = inbox.index_add(1, receivers, messages)
        return nodes + self.node(torch.cat((nodes, inbox), dim=-1))


class GEN(nn.Module):
    def __init__(
        self,
        graph_structure,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        message_passing_steps,
        share_blocks=False,
    ):
        super().__init__()
        self.g = graph_structure
        self.encoder = MLP(output_size, hidden_size, hidden_size, num_layers)
        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(hidden_size, num_layers)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(hidden_size, num_layers)
                    for _ in range(message_passing_steps)
                ]
            )
        self.decoder = MLP(
            hidden_size + input_size, output_size, hidden_size, num_layers
        )

    def forward(self, x, s, q):
        # (B, C, N)
        scores = self.g(x)
        # (B, C, D)
        emb = self.encoder(s)
        # (B, N, D)
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            # (B, N, D)
            latents = block(latents, self.g.senders, self.g.receivers)

        # (B, T, N)
        scores = self.g(q)
        # (B, T, D)
        z = scores.bmm(latents)

        return self.decoder(torch.cat((z, q), dim=-1))
