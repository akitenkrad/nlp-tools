import gzip
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from utils.logger import Logger
from utils.utils import Config, Phase, is_notebook

from datasets.base import BaseDataset

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


@dataclass
class MnistItem(object):
    data: np.ndarray
    label: int


class MnistDataset(BaseDataset):
    """

    examples:
        >>> transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]
        )
        >>> dataset = MnistDataset(config, transform)
    """

    def __init__(self, config: Config, transform: transforms.Compose, class_filter: List[int] = []):
        super().__init__(config, Phase.TRAIN)
        self.transform = transform
        self.class_filter = class_filter

        self.train_data, self.valid_data, self.test_data = self.__load_data__()
        self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, class_filter: List[int] = []):
        train_images, train_labels = self.load_mnist(Phase.TRAIN)
        test_images, test_labels = self.load_mnist(Phase.TEST)

        # np.ndarray -> MnistItem
        train_data = [MnistItem(data.reshape(28, 28), label) for data, label in zip(train_images, train_labels)]
        test_data = [MnistItem(data.reshape(28, 28), label) for data, label in zip(test_images, test_labels)]

        # train valid split
        train_data, valid_data = train_test_split(
            train_data, test_size=self.config.train.valid_size, random_state=self.config.train.seed
        )

        # list -> dict
        train_dict = {idx: item for idx, item in enumerate(train_data)}
        valid_dict = {idx: item for idx, item in enumerate(valid_data)}
        test_dict = {idx: item for idx, item in enumerate(test_data)}

        return train_dict, valid_dict, test_dict

    def __getitem__(self, index):

        if self.phase == Phase.TRAIN or self.phase == Phase.VALID:
            data: MnistItem = self.train_data[index]
            label = data.label

            data = self.transform(np.array(data.data))
            return data, label

        elif self.phase == Phase.DEV:
            data: MnistItem = self.dev_data[index]
            label = data.label

            data = self.transform(np.array(data.data))
            return data, label

        elif self.phase == Phase.TEST:
            data: MnistItem = self.test_data[index]
            data = self.transform(np.array(data.data))
            return data

        raise RuntimeError(f"Unknown phase: {self.pahse}")

    def load_mnist(self, kind: Phase = Phase.TRAIN) -> Tuple[np.ndarray, np.ndarray]:
        """Load MNIST data from `path`"""

        kind_name_map = {Phase.TRAIN: "train", Phase.TEST: "t10k"}
        kind_name = kind_name_map[kind]

        path = self.config.data.data_path / "mnist"
        path.mkdir(parents=True, exist_ok=True)
        labels_path = path / f"{kind_name}-labels-idx1-ubyte.gz"
        images_path = path / f"{kind_name}-images-idx3-ubyte.gz"

        if not labels_path.exists() or not images_path.exists():
            urls = [
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            ]
            for url in tqdm(urls, desc="loading mnist data..."):
                urlData = requests.get(url).content
                with open(path / url.split("/")[-1], "wb") as f:
                    f.write(urlData)

        with gzip.open(str(labels_path), "rb") as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(str(images_path), "rb") as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels
