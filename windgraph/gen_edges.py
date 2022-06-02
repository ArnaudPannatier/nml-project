import torch
import torch.nn as nn

from .mlp import MLP


class MeshNetBlock(nn.Module):
    def __init__(self, in_dim=128, dim=128, l=2):
        super().__init__()
        self.node = MLP(2 * in_dim, in_dim, dim, l)
        self.edge = MLP(3 * in_dim, in_dim, l)

    def forward(self, nodes, edges, senders, receivers):
        edges = self.edge(
            torch.cat((edges, nodes[:, receivers], nodes[:, senders]), dim=-1)
        )
        inbox = torch.zeros(nodes.shape, device=nodes.device)
        inbox = inbox.index_add(1, receivers, edges)
        return nodes + self.node(torch.cat((nodes, inbox), dim=-1)), edges


class GEN(nn.Module):
    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_y,
        dim_h,
        nlayers,
        message_passing_steps,
        share_blocks=False,
    ):
        super().__init__()
        self.g = graph_structure
        self.encoder = MLP(dim_x + dim_y, dim_h, dim_h, nlayers)
        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [MeshNetBlock(dim_h, dim_h, nlayers)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    MeshNetBlock(dim_h, dim_h, nlayers)
                    for _ in range(message_passing_steps)
                ]
            )
        self.decoder = MLP(dim_h + dim_x, dim_y, dim_h, nlayers)

    def forward(self, x, s, q):
        scores = self.g(x)
        emb = self.encoder(s)
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents, self.g.edges = block(
                latents, self.g.senders, self.g.receivers, self.g.edges
            )
        scores = self.g(q)
        z = scores.bmm(latents)
        return self.decoder(torch.cat((z, q), dim=-1))
