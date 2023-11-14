import pickle
from dataclasses import dataclass
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from ml_tools.datasets.base import BaseDataset
from ml_tools.utils.utils import Config, Phase, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


@dataclass
class Item(object):
    data: np.ndarray
    label: int


ItemSet = Dict[int, Item]


class CIFAR10(BaseDataset):
    def __init__(self, config: Config, transform: transforms.Compose, class_filter: List[int] = []):
        super().__init__(config, Phase.TRAIN)
        self.transform = transform
        self.class_filter = class_filter

        self.train_data, self.valid_data, self.test_data = self.__load_data__()
        self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}
        self.class2index, self.index2class = self.__load_meta__()

    def __load_meta__(self):
        meta_path = Path(self.config.data.data_path) / "cifar-10-batches-py" / "batches.meta"
        with open(meta_path, mode="rb") as rf:
            data = pickle.load(rf, encoding="latin1")
            classes = data["label_names"]
            cls2idx = {c: i for i, c in enumerate(classes)}
            idx2cls = {i: c for i, c in enumerate(classes)}
        return cls2idx, idx2cls

    def __load_data__(
        self, dataset_path: PathLike = Path("./cifar10.tar.gz"), test_size: float = 0.2, valid_size: float = 0.2
    ) -> Tuple[ItemSet, ItemSet, ItemSet]:
        """load imdb corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, Item]
        """
        base_dir = "cifar-10-batches-py"
        data_path = Path(self.config.data.data_path)

        if not (data_path / base_dir).exists():
            torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)

        train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_list = ["test_batch"]

        # load train data
        _train_data, train_labels = [], []
        for filename in tqdm(train_list, desc="Loading Train Data", leave=False):
            filepath = data_path / base_dir / filename
            with open(filepath, mode="rb") as rf:
                entry = pickle.load(rf, encoding="latin1")
                _train_data.append(entry["data"])
                if "labels" in entry:
                    train_labels.extend(entry["labels"])
                else:
                    train_labels.extend(entry["fine_labels"])
        raw_data = np.vstack(_train_data).reshape(-1, 3, 32, 32)
        raw_data = raw_data.transpose((0, 2, 3, 1))
        train_data = [Item(data, label) for data, label in zip(raw_data, train_labels)]
        train_data, valid_data = train_test_split(train_data, test_size=self.config.train.valid_size)
        train_dict = {index: item for index, item in enumerate(train_data)}
        valid_dict = {index: item for index, item in enumerate(valid_data)}

        # load test data
        _test_data, test_labels = [], []
        for filename in tqdm(test_list, desc="Loading Test Data", leave=False):
            filepath = data_path / base_dir / filename
            with open(filepath, mode="rb") as rf:
                entry = pickle.load(rf, encoding="latin1")
                _test_data.append(entry["data"])
                if "labels" in entry:
                    test_labels.extend(entry["labels"])
                else:
                    test_labels.extend(entry["fine_labels"])
        test_data = [
            Item(data, label) for data, label in zip(np.vstack(_test_data).reshape(-1, 3, 32, 32), test_labels)
        ]
        test_dict = {index: item for index, item in enumerate(test_data)}

        return train_dict, valid_dict, test_dict

    def __getitem__(self, index):
        data: Item
        if self.phase == Phase.TRAIN:
            data = self.train_data[index]
        elif self.phase == Phase.VALID:
            data = self.valid_data[index]
        elif self.phase == Phase.TEST:
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid Phase: {self.phase}")

        images = self.transform(Image.fromarray(data.data))

        return images, data.label
