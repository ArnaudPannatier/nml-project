import torch
import torch.utils.data as data
from more_itertools import pairwise, windowed


class SequentialDataset(data.Dataset):
    def __init__(self, dataset, time_interval):
        inputs, _ = dataset[:]
        self.time_interval = time_interval / dataset.stds[3]

        time = inputs[:, 3].clone()
        mint, maxt = time.min(), time.max()
        intervals = int((maxt - mint) / self.time_interval)

        splits = torch.tensor(
            [time[0] + self.time_interval * i for i in range(intervals + 1)]
        )

        splits_index = torch.searchsorted(time, splits)
        splits_index = splits_index.tolist() + [-1]
        self.slices = [slice(a, b) for a, b in pairwise(splits_index) if a != b]
        self.sequences = tuple(dataset[sl] for sl in self.slices)

        assert all([len(s[0]) for s in self.sequences])

        assert all(
            [
                (s[0][-1, 3] - s[0][0, 3]) <= self.time_interval + 5e-4
                for s in self.sequences
            ]
        )

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return len(self.sequences)


class CombineSequentialDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lenghts = [len(d) for d in self.datasets]
        self.bins = torch.tensor(self.lenghts).cumsum(0)

    def __getitem__(self, index):
        j = torch.searchsorted(self.bins, index + 1)
        i = index if j == 0 else index - self.bins[j - 1]

        return self.datasets[j][i]

    def __len__(self):
        return sum(self.lenghts)


class PercentageDataset(data.Dataset):
    def __init__(self, dataset, percentage):
        self.dataset = dataset
        max_i = int(percentage * len(dataset))
        self.idxs = torch.randperm(len(dataset))[:max_i]

    def __getitem__(self, index):
        return self.dataset[self.idxs[index]]

    def __len__(self):
        return len(self.idxs)


class WindSeqDataset(data.Dataset):
    def __init__(self, dataset, time_interval, time_window, context):
        self.means = dataset.means
        self.stds = dataset.stds
        assert time_window % time_interval == 0
        seq_dataset = SequentialDataset(dataset, time_interval)
        offset = time_window // time_interval + 1

        if context == 1:
            self.pairs = list(zip(seq_dataset[:-offset], seq_dataset[offset:]))
        else:
            c_tup = [sum(s, ()) for s in windowed(seq_dataset[:-offset], context)]
            contexts = [(torch.cat(c[::2]), torch.cat(c[1::2])) for c in c_tup]
            self.pairs = list(zip(contexts, seq_dataset[(offset + context - 1) :]))

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def scale(self, y):
        device = y.device
        s = self.stds.to(device)
        m = self.means.to(device)

        return y * s[4:] + m[4:]
