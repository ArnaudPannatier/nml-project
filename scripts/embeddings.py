import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from windgraph.datasets import datasets
from windgraph.experiment import add_exp_args, run_exp
from windgraph.gen import GEN, GENwoenc
from windgraph.graphs import GraphStructure, kmeans_from_dataset, neighbors_edges
from windgraph.mlp import MLP
from windgraph.positional_encoding import PosEncAdd, PosEncCat

rel_path = Path(__file__).parent.parent

if __name__ == "__main__":
    load_dotenv()
    train_dataset, val_dataset = datasets(os.getenv("ROOT_FOLDER"))

    parser = ArgumentParser(description="Windspeed Pipeline")
    add_exp_args(parser)

    parser.add_argument(
        "--emb",
        choices=["noemb", "default", "sincos", "3dsincos"],
        default="noemb",
    )
    parser.add_argument("--nodes", type=int, default=10)

    args = parser.parse_args()
    if not args.name:
        args.name = f"{args.emb}-{args.nodes}N{args.seed}"

    train_dataset, val_dataset = datasets(os.getenv("ROOT_FOLDER"))

    kmeans_pos = kmeans_from_dataset(k=args.nodes)
    gs = GraphStructure(kmeans_pos, *neighbors_edges(kmeans_pos, 3), fixed=True)

    if args.emb == "noemb":
        model = GENwoenc(gs, 3, 2, 32, 2, 3)
    elif args.emb == "default":
        model = GEN(gs, 3, 2, 32, 2, 2, 3)
    elif args.emb == "sincos":
        model = GEN(gs, 60, 40, 32, 2, 2, 3, pe=PosEncCat(10))
    else:
        model = GEN(gs, 20, 20, 32, 2, 2, 3, pe=PosEncAdd(10))

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = run_exp(model, train_dl, val_dl, args)
