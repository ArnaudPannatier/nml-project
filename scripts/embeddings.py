import os

import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from windgraph import GEN, GraphStructure
from windgraph.datasets import FourWeeksDataset
from windgraph.experiment import run_exp
from windgraph.graphs import kmeans_from_dataset, neighbors_edges
from windgraph.mlp import MLP
from windgraph.positional_encoding import SinCosPositionalEncoding

N = 1000

if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("ROOT_FOLDER")    

    dataset = FourWeeksDataset(path, stage="train")
    val_dataset = FourWeeksDataset(path, stage="val")

    kmeans_pos = kmeans_from_dataset(dataset, N)
    gs = GraphStructure(kmeans_pos, *neighbors_edges(kmeans_pos, 3)),
    model = GEN(gs, 4,2, 32, 2, 7)

    encoders = [
        nn.Identity(),
        None,
        SinCosPositionalEncoding(1/10000.0, 32),
        nn.Sequential(SinCosPositionalEncoding(1/10000.0, 32), MLP(32,32,32,2))
    ]

    train_dl = DataLoader(dataset, batch_size=12, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=12, shuffle=True)

    model, name = run_exp(model, train_dl)
