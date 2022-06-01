import os

import matplotlib.pyplot as plt
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from windgraph.datasets import datasets
from windgraph.experiment import run_exp
from windgraph.gen import GEN, GENwoenc
from windgraph.graphs import GraphStructure, kmeans_from_dataset, neighbors_edges
from windgraph.mlp import MLP
from windgraph.positional_encoding import SinCosPositionalEncoding

N = 1000

if __name__ == "__main__":
    load_dotenv()
    train_dataset, val_dataset = datasets(os.getenv("ROOT_FOLDER"))

    kmeans_pos = kmeans_from_dataset(train_dataset, N)
    gs = GraphStructure(kmeans_pos, *neighbors_edges(kmeans_pos, 3), fixed=True)

    model = GENwoenc(gs, 3, 2, 32, 2, 7)

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model, name = run_exp(model, train_dl, val_dl)

    plt.figure()
    fig, axs = plt.subplots(5, 3, figsize=(6, 10))

    (cxs, cys, txs), targets = next(iter(val_dl))
    outputs = model(cxs.cuda(), cys.cuda(), txs.cuda()).cpu()

    for ax, cy, target, output in zip(axs, cys, targets, outputs):
        ax[1].imshow(cy.view(14, 28).int(), cmap="Greys_r")
        ax[1].set_xlim([0, 26])
        ax[1].set_ylim([26, 0])
        ax[0].imshow(target.view(28, 28).int(), cmap="Greys_r")
        ax[2].imshow(output.view(28, 28).int(), cmap="Greys_r")
    fig.subplots_adjust(wspace=0.1, hspace=0.2, left=0.1, right=1.0)
    fig.savefig(name + ".pdf")
