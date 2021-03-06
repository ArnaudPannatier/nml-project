import torch
import torch.nn as nn

from .mlp import MLP


class GraphNetBlock(nn.Module):
    def __init__(self, in_dim=128, dim=128, l=2):
        super().__init__()
        self.message = MLP(2 * in_dim, in_dim, dim, l)
        self.node = MLP(2 * in_dim, in_dim, dim, l)

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
        dim_x,
        dim_y,
        dim_h,
        dim_out,
        nlayers,
        message_passing_steps,
        share_blocks=False,
        pe=None,
    ):
        super().__init__()
        self.g = graph_structure
        self.pe = pe
        self.encoder = MLP(dim_x + dim_y, dim_h, dim_h, nlayers)
        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h, dim_h, nlayers)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_h, dim_h, nlayers)
                    for _ in range(message_passing_steps)
                ]
            )
        self.decoder = MLP(dim_h + dim_x, dim_out, dim_h, nlayers)

    def forward(self, x, s, q):
        # (B, C, N)
        scores = self.g(x)
        # (B, C, D)
        if self.pe:
            c = torch.cat((self.pe(x), self.pe(s)), dim=-1)
        else:
            c = torch.cat((x, s), dim=-1)
        emb = self.encoder(c)
        # (B, N, D)
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            # (B, N, D)
            latents = block(latents, self.g.senders, self.g.receivers)

        # (B, T, N)
        scores = self.g(q)
        # (B, T, D)
        z = scores.bmm(latents)
        # Decoder uses q as well (not in the paper but in their code.)
        # So it really is a CNP with
        if self.pe:
            q = self.pe(q)
        return self.decoder(torch.cat((z, q), dim=-1))


class GENwoenc(nn.Module):
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
        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_x + dim_y, dim_h, nlayers)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_x + dim_y, dim_h, nlayers)
                    for _ in range(message_passing_steps)
                ]
            )
        self.decoder = MLP(2 * dim_x + dim_y, dim_y, dim_h, nlayers)

    def forward(self, x, s, q):
        scores = self.g(x)
        latents = scores.transpose(1, 2).bmm(torch.cat((x, s), dim=-1))
        for block in self.gn_blocks:
            latents = block(latents, self.g.senders, self.g.receivers)

        scores = self.g(q)
        z = scores.bmm(latents)
        return self.decoder(torch.cat((z, q), dim=-1))
