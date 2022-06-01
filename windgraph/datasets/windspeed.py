from collections import namedtuple

import pandas as pd
import torch

from .malat import MalatDataset
from .windprocessing import (
    load_means_stds,
    process_df,
    save_count,
    save_means_stds,
    select_day,
)

Weeks = namedtuple("Weeks", ["files"])
Week = namedtuple("Week", ["file"])
Day = namedtuple("Day", ["file", "day"])


class BaseDataset(MalatDataset):
    mirrors = ["https://zenodo.org/record/5074237/files/"]

    resources = [("SkySoft_WindSpeed.tar.gz", "87b5f8c8e7faad810b43af6d67a49428")]
    _fid_dicts = None

    @property
    def folder(self):
        return self.root / "windspeed_raw"

    @property
    def files(self):
        return sorted([f for f in (self.folder / "SkySoft_WindSpeed").glob("*.csv")])

    def _process_raw_files(self):
        print("Starting Processing.")
        index = self.stage_index[self.stage]

        if isinstance(index, Weeks):
            files = [self.files[i] for i in index.files]
            data_list = []
            fid2indices = []
            for f in files:
                print("  - Processing file : ", str(f))
                df = pd.read_csv(f)
                data_in_file, fid = process_df(df)
                data_list.append(data_in_file)
                fid2indices.append(fid)

            save_count(data_list, self.path("_count.csv"))

            data = torch.cat(data_list)
        else:
            df = pd.read_csv(self.files[index.file])
            if isinstance(index, Week):
                data, fid2indices = process_df(df)
            else:
                df = select_day(df, index.day)
                data, fid2indices = process_df(df)

        means, std = data.mean(0), data.std(0)
        print("Saving means.")
        save_means_stds(means, std, self.path("_means_std.yml"))

        print("Saving data.")
        data = (data - means) / std
        torch.save(data, self.path(".pt"))

        print("Saving fid2indices")
        torch.save(fid2indices, self.path("_fid_dicts.pt"))

    def _load_data(self):
        self.data = torch.load(self.path(".pt"))
        self.means, self.stds = load_means_stds(self.path("_means_std.yml"))

        return self.data[:, :4], self.data[:, 4:]

    @property
    def fid_dicts(self):
        if self._fid_dicts is None:
            self._fid_dicts = torch.load(self.path("_fid_dicts.pt"))
        return self._fid_dicts


class FourWeeksDataset(BaseDataset):
    stage_index = {"train": Weeks([0, 1, 2, 4]), "test": Day(4, 2), "val": Day(3, 3)}


class Week0(BaseDataset):
    stage_index = {"train": Week(0), "test": Day(4, 2), "val": Day(3, 3)}


class Week1(BaseDataset):
    stage_index = {"train": Week(1), "test": Day(4, 2), "val": Day(3, 3)}


class Week2(BaseDataset):
    stage_index = {"train": Week(2), "test": Day(4, 2), "val": Day(3, 3)}


class Week4(BaseDataset):
    stage_index = {"train": Week(4), "test": Day(4, 2), "val": Day(3, 3)}
