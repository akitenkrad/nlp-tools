import tarfile
from collections import namedtuple
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

from embeddings.base import Embedding
from sklearn.model_selection import train_test_split
from utils.utils import Config, Phase, download

from datasets.base import BaseDataset

ImdbItem = namedtuple('ImdbItem', ('filepath', 'label'))
ImdbDs = Dict[int, ImdbItem]


class ImdbDataset(BaseDataset):
    def __init__(self, config: Config, embedding: Embedding):
        super().__init__(config, Phase.TRAIN)
        self.embedding = embedding
        self.n_class = 2

        self.train_data, self.valid_data, self.test_data = self.__load_data__(self.dataset_path)
        self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path: Path) -> Tuple[ImdbDs, ImdbDs, ImdbDs]:
        '''load imdb corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, ImdbItem]
        '''
        if not (dataset_path / 'imdb').exists():
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            self.config.log.dataset_log.info(f'download imdb corpus from {url}')

            (dataset_path / 'imdb').mkdir(parents=True, exist_ok=True)
            download(url, str(dataset_path / 'aclImdb_v1.tar.gz'))

            with tarfile.open(dataset_path / 'aclImdb_v1.tar.gz', 'r:gz') as tar:
                tar.extractall(path=str(dataset_path / 'imdb'))

        train_pos_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'aclImdb' / 'train' / 'pos' / '*.txt'))]
        train_neg_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'aclImdb' / 'train' / 'neg' / '*.txt'))]
        test_pos_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'aclImdb' / 'test' / 'pos' / '*.txt'))]
        test_neg_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'aclImdb' / 'test' / 'neg' / '*.txt'))]

        train_data_pos = [ImdbItem(p, 1) for p in train_pos_files]
        train_data_neg = [ImdbItem(p, 0) for p in train_neg_files]
        train_data_pos, valid_data_pos = train_test_split(train_data_pos, test_size=self.config.train.valid_size)
        train_data_neg, valid_data_neg = train_test_split(train_data_neg, test_size=self.config.train.valid_size)
        train_data: List[ImdbItem] = train_data_pos + train_data_neg
        valid_data: List[ImdbItem] = valid_data_pos + valid_data_neg

        test_data_pos = [ImdbItem(p, 1) for p in test_pos_files]
        test_data_neg = [ImdbItem(p, 0) for p in test_neg_files]
        test_data: List[ImdbItem] = test_data_pos + test_data_neg

        # list -> dict
        train_dict = {idx: item for idx, item in enumerate(train_data)}
        valid_dict = {idx: item for idx, item in enumerate(valid_data)}
        test_dict = {idx: item for idx, item in enumerate(test_data)}

        return train_dict, valid_dict, test_dict

    def __load_text__(self, path: PathLike) -> str:
        text = open(path, 'rt').read().strip()
        return text

    def __getitem__(self, index):
        if self.phase == Phase.TRAIN or self.phase == Phase.DEV:
            data: ImdbItem = self.train_data[index]
        elif self.phase == Phase.VALID:
            data: ImdbItem = self.valid_data[index]
        elif self.phase == Phase.TEST or self.phase == Phase.SUBMISSION:
            data: ImdbItem = self.test_data[index]
        else:
            raise ValueError(f'Invalid Phase: {self.phase}')

        text = self.__load_text__(data.filepath)
        embedding = self.embedding.embed(text)
        return embedding, data.label
