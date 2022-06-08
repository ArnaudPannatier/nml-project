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
from windgraph.gen_edges import GENfd
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
        args.name = f"fd-{args.nodes}N-noemb{args.seed}"

    train_dataset, val_dataset = datasets(os.getenv("ROOT_FOLDER"))

    kmeans_pos = kmeans_from_dataset(k=args.nodes)
    gs = GraphStructure(kmeans_pos, *ba_edges(kmeans_pos), fixed=False)
    model = GENfd(gs, 60, 40, 32, 2, 2, 3, pe=PosEncCat(10))

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = run_exp(model, train_dl, val_dl, args)
