import tarfile
from collections import namedtuple
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split

from nlp_tools.datasets.base import BaseDataset
from nlp_tools.embeddings.base import Embedding
from nlp_tools.utils.utils import Config, Phase, download

LivedoorNewsItem = namedtuple("LivedoorNewsItem", ("filepath", "label"))
LivedoorDs = Dict[int, LivedoorNewsItem]


class LivedoorNewsDataset(BaseDataset):
    def __init__(self, config: Config, embedding: Embedding, test_size: float = 0.1, valid_size: float = 0.1):
        self.config = config
        config.add_logger("dataset_log")
        self.embedding = embedding
        self.dataset_path = Path("data")
        self.test_size = test_size
        self.valid_size = valid_size

        self.train_data, self.valid_data, self.test_data = self.__load_data__(
            str(self.dataset_path), self.test_size, self.valid_size
        )

    def __load_data__(self, dataset_path: str, test_size: float = 2.0, valid_size: float = 0.2, **kwargs):
        """load livedoor news corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, LivedoorNewsItem]
        """
        ds_path = Path(dataset_path)
        if not (ds_path / "livedoor-corpus").exists():
            url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
            self.config.log.dataset_log.info(f"download livedoor corpus from {url}")

            (ds_path / "livedoor-news-corpus").mkdir(parents=True, exist_ok=True)
            download(url, str(ds_path / "livedoor-news-corpus.tar.gz"))

            with tarfile.open(ds_path / "livedoor-news-corpus.tar.gz", "r:gz") as tar:
                tar.extractall(path=str(ds_path / "livedoor-news-corpus"))

        exclude_files = ["CHANGES.txt", "README.txt", "LICENSE.txt"]
        files = [
            Path(f)
            for f in glob(str(ds_path / "livedoor-news-corpus" / "**" / "*.txt"))
            if Path(f).name not in exclude_files
        ]
        labels = [f.parent.name for f in files]
        items = [LivedoorNewsItem(f, l) for f, l in zip(files, labels)]

        # split data -> train, valid, test
        train_data, test_data = train_test_split(items, test_size=test_size)
        train_data, valid_data = train_test_split(train_data, test_size=valid_size)

        # list -> dict
        train_data = {idx: item for idx, item in enumerate(train_data)}
        valid_data = {idx: item for idx, item in enumerate(valid_data)}
        test_data = {idx: item for idx, item in enumerate(test_data)}

        return train_data, valid_data, test_data

    def __load_text__(self, path: str):
        lines = open(path, "rt").read().splitlines()
        # 0: url
        # 1: timestamp
        # 2: title
        # 3-: text
        text = "\n".join(lines[3:])
        return text

    def __getitem__(self, index):
        data: LivedoorNewsItem
        if self.phase == Phase.TRAIN or self.phase == Phase.DEV:
            data = self.train_data[index]
        elif self.phase == Phase.VALID:
            data = self.valid_data[index]
        elif self.phase == Phase.TEST or self.phase == Phase.SUBMISSION:
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid Phase: {self.phase}")

        text = self.__load_text__(data.filepath)
        tokens = self.embedding.tokenize(text)
        indices = [self.embedding.token2index(token) for token in tokens]
        embedding = self.embedding.embed(indices)
        return embedding, data.label
