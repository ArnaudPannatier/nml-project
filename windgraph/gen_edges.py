import torch
import torch.nn as nn

from .mlp import MLP


class MeshNetBlock(nn.Module):
    def __init__(self, in_dim=128, dim=128, l=2):
        super().__init__()
        self.node = MLP(2 * in_dim, in_dim, dim, l)
        self.edge = MLP(3 * in_dim, in_dim, dim, l)

    def forward(self, nodes, edges, senders, receivers):
        edges = self.edge(
            torch.cat((edges, nodes[:, receivers], nodes[:, senders]), dim=-1)
        )
        inbox = torch.zeros(nodes.shape, device=nodes.device)
        inbox = inbox.index_add(1, receivers, edges)
        return nodes + self.node(torch.cat((nodes, inbox), dim=-1)), edges


class GENEdge(nn.Module):
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
        self.edge = MLP(1, dim_h, dim_h, nlayers)
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
        self.decoder = MLP(dim_h + dim_x, dim_out, dim_h, nlayers)

    def forward(self, x, s, q):
        scores = self.g(x)
        if self.pe:
            c = torch.cat((self.pe(x), self.pe(s)), dim=-1)
        else:
            c = torch.cat((x, s), dim=-1)

        emb = self.encoder(c)
        latents = scores.transpose(1, 2).bmm(emb)
        # self.g.edges = (latents[:, self.g.receivers] - latents[:, self.g.senders]) ** 2
        self.g.edges = self.g.pos[self.g.receivers] - self.g.pos[self.g.senders]
        self.g.edges = self.edge(self.g.edges.pow(2).sum(-1).unsqueeze(1)).unsqueeze(0)

        for block in self.gn_blocks:
            latents, self.g.edges = block(
                latents, self.g.edges, self.g.senders, self.g.receivers
            )
        scores = self.g(q)
        z = scores.bmm(latents)
        if self.pe:
            q = self.pe(q)
        return self.decoder(torch.cat((z, q), dim=-1))


class GENfd(nn.Module):
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
        self.edge = MLP(3, dim_h, dim_h, nlayers)
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
        self.decoder = MLP(dim_h + dim_x, dim_out, dim_h, nlayers)

    def forward(self, x, s, q):
        scores = self.g(x)
        if self.pe:
            c = torch.cat((self.pe(x), self.pe(s)), dim=-1)
        else:
            c = torch.cat((x, s), dim=-1)

        emb = self.encoder(c)
        latents = scores.transpose(1, 2).bmm(emb)
        # self.g.edges = (latents[:, self.g.receivers] - latents[:, self.g.senders])

        diff = self.g.pos[self.g.receivers] - self.g.pos[self.g.senders]
        dist = diff.pow(2).sum(-1)
        self.g.edges = self.edge(diff / dist.unsqueeze(1)).unsqueeze(0)

        for block in self.gn_blocks:
            latents, self.g.edges = block(
                latents, self.g.edges, self.g.senders, self.g.receivers
            )
        scores = self.g(q)
        z = scores.bmm(latents)
        if self.pe:
            q = self.pe(q)
        return self.decoder(torch.cat((z, q), dim=-1))
