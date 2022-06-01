from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class GraphStructure(nn.Module):
    def __init__(self, pos, senders, receivers, fixed):
        super().__init__()

        if fixed:
            self.register_buffer("pos", pos)
        else:
            self.pos = nn.Parameter(pos)
        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)

    def forward(self, x):
        pseudo_dist = torch.norm(self.pos, dim=-1) ** 2 - 2 * x @ self.pos.t()
        return F.softmax(-pseudo_dist, dim=-1)

    @classmethod
    def from_file(cls, fixed, graph=None):
        if graph is None:
            graph = Path(__file__).parent / "graph.pt"

        pos, senders, receivers = torch.load(graph)
        return cls(pos, senders, receivers, fixed)

    def to_networkx(self):
        edges = (
            torch.cat((self.senders[:, None], self.receivers[:, None]), dim=-1)
            .cpu()
            .numpy()
        )
        pos = self.pos.detach().cpu().numpy()
        g = nx.Graph()
        g.add_edges_from(edges)
        return g, dict(enumerate(pos))


def grid(n):
    x = torch.linspace(0, 1, n)
    I, J = torch.meshgrid(x, x, indexing="ij")
    pos = torch.cat((I.reshape(-1, 1), J.reshape(-1, 1)), dim=1)
    idx = torch.arange(n**2).view(n, n)

    senders = torch.cat(
        (
            idx[1:].flatten(),  # ↑
            idx[:, :-1].flatten(),  # →
            idx[:-1].flatten(),  # ↓
            idx[:, 1:].flatten(),  # ←
        )
    )
    receivers = torch.cat(
        (
            idx[:-1].flatten(),  # ↑
            idx[:, 1:].flatten(),  # →
            idx[1:].flatten(),  # ↓
            idx[:, :-1].flatten(),  # ←
        )
    )
    return pos, senders, receivers


def kmeans_from_dataset(dataset, k=1000, path=Path("kmeans.pt")):
    if path.exists():
        return torch.load(path)

    choices = torch.randperm(len(dataset))[:100000]
    inputs, _ = dataset[choices]
    inputs = inputs[:, :3]

    print(inputs.shape)
    kmeans = KMeans(n_clusters=k, verbose=3).fit(inputs)
    pos = torch.tensor(kmeans.cluster_centers_).float()
    torch.save(pos, path)
    return pos


def neighbors_edges(pos, n=3):
    dists = torch.norm(pos[None, :, :] - pos[:, None, :], dim=2)
    receivers = dists.argsort()[:, 1 : n + 1].flatten()
    senders = torch.arange(len(pos)).repeat(n)
    return senders, receivers


def ba_edges(pos, m=3):
    g = nx.barabasi_albert_graph(len(pos), m)
    edge_index = torch.tensor(g.edges()).long()
    return edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()


def random_graph(p, k=1000):
    pos = torch.rand(3, k) * 2 - 1
    edge_index = torch.full((k, k), p).bernoulli().non_zero().long()
    return pos, edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()
