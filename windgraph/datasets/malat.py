from pathlib import Path
from urllib.error import URLError

import torch.utils.data as data
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive)

# Largerly inspired from torchvision


class MalatDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 stage="train",
                 download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        if download:
            self.download()

        if not self.check_processed_exists():
            self.processed_folder.mkdir(exist_ok=True)
            self._process_raw_files()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.inputs, self.targets = self._load_data()

    def _process_raw_files(self):
        raise NotImplementedError

    def path(self, fn):
        return self.processed_folder / (self.stage + fn)

    def check_processed_exists(self):
        return (self.processed_folder.exists() and self.path(".pt").exists())

    def __getitem__(self, index):
        inputs, targets = self.inputs[index], self.targets[index]

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform

        return inputs, targets

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(str(self.root)))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return self.root / self.__class__.__name__

    def _check_exists(self):
        return all(check_integrity(self.folder / r) for r, _ in self.resources)

    def download(self) -> None:
        """Download the data if it doesn't exist already."""

        if self._check_exists():
            return

        self.folder.mkdir(exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(url,
                                                 download_root=self.folder,
                                                 filename=filename,
                                                 md5=md5)
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self):
        return "Split: {}".format(self.stage)
