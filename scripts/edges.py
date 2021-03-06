import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from windgraph.datasets import datasets
from windgraph.experiment import add_exp_args, run_exp
from windgraph.gen_edges import GENEdge
from windgraph.graphs import GraphStructure, ba_edges, kmeans_from_dataset
from windgraph.mlp import MLP
from windgraph.positional_encoding import PosEncCat

rel_path = Path(__file__).parent.parent

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="Windspeed Pipeline")
    add_exp_args(parser)

    parser.add_argument("--nodes", type=int, default=20)

    args = parser.parse_args()
    if not args.name:
        args.name = f"edges-{args.nodes}N-noemb{args.seed}"

    train_dataset, val_dataset = datasets(os.getenv("ROOT_FOLDER"))

    kmeans_pos = kmeans_from_dataset(k=args.nodes)
    gs = GraphStructure(kmeans_pos, *ba_edges(kmeans_pos), fixed=False)
    model = GENEdge(gs, 60, 40, 32, 2, 2, 3, pe=PosEncCat(10))

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = run_exp(model, train_dl, val_dl, args)

    plt.figure()
    fig, axs = plt.subplots(3, 6, figsize=(15, 8))

    def quiver_plot(lats, model, ax):
        lats = lats.detach().cpu()
        p = model.g.pos.detach().cpu()
        ax.plot(p[:, 0], p[:, 1], ".")
        ax.quiver(
            p[:, 0],
            p[:, 1],
            lats[0, :, 3],
            lats[0, :, 4],
        )

    for ax in axs:
        (cx, cy, tx), target = next(iter(val_dl))
        output = model(cx.cuda(), cy.cuda(), tx.cuda()).cpu()
        g, pos = model.g.to_networkx()
        nx.draw(g, pos, ax=ax[0], width=0.4, node_size=50)
        xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
        ax[1].quiver(cx[0, :, 0], cx[0, :, 1], cy[0, :, 0], cy[0, :, 1])

        scores = model.g(cx.cuda())
        latents = scores.transpose(1, 2).bmm(torch.cat((cx.cuda(), cy.cuda()), dim=-1))

        for block, a in zip(model.gn_blocks, ax[2:5]):
            latents = block(latents, model.g.senders, model.g.receivers)
            quiver_plot(latents, model, a)

        scores = model.g(tx.cuda())
        z = scores.bmm(latents)
        out = model.decoder(torch.cat((z, tx.cuda()), dim=-1)).detach().cpu()
        ax[5].quiver(tx[0, :, 0], tx[0, :, 1], out[0, :, 0], out[0, :, 1], color="blue")
        ax[5].quiver(
            tx[0, :, 0], tx[0, :, 1], target[0, :, 0], target[0, :, 1], color="red"
        )

        for a in ax:
            a.set_xlim(xlim)
            a.set_ylim(ylim)

    fig.savefig(rel_path / f"results/{args.name}.pdf")
