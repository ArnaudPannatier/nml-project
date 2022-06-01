import os

import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from windgraph import GEN, GraphStructure
from windgraph.datasets import FourWeeksDataset
from windgraph.experiment import run_exp
from windgraph.graphs import (ba_edges, kmeans_from_dataset, neighbors_edges,
                              random_graph)

N = 1000

if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("ROOT_FOLDER")    

    dataset = FourWeeksDataset(path, stage="train")
    val_dataset = FourWeeksDataset(path, stage="val")

    kmeans_pos = kmeans_from_dataset(dataset, N)
    graph_structures = [
        GraphStructure(kmeans_pos, *neighbors_edges(kmeans_pos, 3)),
        GraphStructure(*random_graph(0.5, N)),
        GraphStructure(kmeans_pos, *ba_edges(kmeans_pos))
    ]
    model = GEN(graph_structures[0], 4,2, 32, 2, 7)


    train_dl = DataLoader(dataset, batch_size=12, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=12, shuffle=True)

    model, name = run_exp(model, train_dl)
