import tarfile
from collections import namedtuple
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from nlp_tools.datasets.base import BaseDataset
from nlp_tools.embeddings.base import Embedding
from nlp_tools.utils.utils import Config, Phase, download

Item = namedtuple("Item", ("filepath", "label"))
ItemSet = Dict[int, Item]


class ImdbDataset(BaseDataset):
    def __init__(self, config: Config, embedding: Embedding):
        super().__init__(config, Phase.TRAIN)
        self.embedding = embedding
        self.n_class = 2

        self.train_data, self.valid_data, self.test_data = self.__load_data__(
            str(self.dataset_path), self.config.train.test_size, self.config.train.valid_size
        )
        self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path: str, test_size: float = 0.2, valid_size: float = 0.2, **kwargs):
        """load imdb corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, Item]
        """
        ds_path = Path(dataset_path)
        if not (ds_path / "imdb").exists():
            url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            self.config.log.dataset_log.info(f"download imdb corpus from {url}")

            (ds_path / "imdb").mkdir(parents=True, exist_ok=True)
            download(url, str(ds_path / "aclImdb_v1.tar.gz"))

            with tarfile.open(ds_path / "aclImdb_v1.tar.gz", "r:gz") as tar:
                tar.extractall(path=str(ds_path / "imdb"))

        train_pos_files = [Path(f) for f in glob(str(ds_path / "imdb" / "aclImdb" / "train" / "pos" / "*.txt"))]
        train_neg_files = [Path(f) for f in glob(str(ds_path / "imdb" / "aclImdb" / "train" / "neg" / "*.txt"))]
        test_pos_files = [Path(f) for f in glob(str(ds_path / "imdb" / "aclImdb" / "test" / "pos" / "*.txt"))]
        test_neg_files = [Path(f) for f in glob(str(ds_path / "imdb" / "aclImdb" / "test" / "neg" / "*.txt"))]

        train_data_pos = [Item(p, 1) for p in train_pos_files]
        train_data_neg = [Item(p, 0) for p in train_neg_files]
        train_data_pos, valid_data_pos = train_test_split(train_data_pos, test_size=self.config.train.valid_size)
        train_data_neg, valid_data_neg = train_test_split(train_data_neg, test_size=self.config.train.valid_size)
        train_data: List[Item] = train_data_pos + train_data_neg
        valid_data: List[Item] = valid_data_pos + valid_data_neg

        test_data_pos = [Item(p, 1) for p in test_pos_files]
        test_data_neg = [Item(p, 0) for p in test_neg_files]
        test_data: List[Item] = test_data_pos + test_data_neg

        # list -> dict
        train_dict = {idx: item for idx, item in enumerate(train_data)}
        valid_dict = {idx: item for idx, item in enumerate(valid_data)}
        test_dict = {idx: item for idx, item in enumerate(test_data)}

        return train_dict, valid_dict, test_dict

    def __load_text__(self, path: PathLike) -> str:
        text = open(path, "rt").read().strip()
        return text

    def __getitem__(self, index):
        data: Item
        if self.phase == Phase.TRAIN or self.phase == Phase.DEV:
            data = self.train_data[index]
        elif self.phase == Phase.VALID:
            data = self.valid_data[index]
        elif self.phase == Phase.TEST or self.phase == Phase.SUBMISSION:
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid Phase: {self.phase}")

        text = self.__load_text__(data.filepath)
        embedding = self.embedding.embed(text)
        return embedding, data.label
