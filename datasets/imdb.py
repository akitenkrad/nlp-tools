from typing import Tuple, Dict
from os import PathLike
from pathlib import Path
from collections import namedtuple
import urllib.request
from glob import glob
from tqdm import tqdm
import tarfile
import numpy as np
from sklearn.model_selection import train_test_split

from utils.logger import Logger
from utils.utils import Phase, Config, download
from embeddings.base import Embedding
from datasets.base import BaseDataset

ImdbItem = namedtuple('ImdbItem', ('filepath', 'label'))
ImdbDs = Dict[int, ImdbItem]

class ImdbDataset(BaseDataset):
    def __init__(self, config:Config, embedding:Embedding, valid_size:float=0.1):
        self.config = config
        config.add_logger('dataset_log')
        self.embedding = embedding
        self.dataset_path = Path('data')
        self.valid_size = valid_size

        self.train_data, self.valid_data, self.test_data = self.__load_data__(self.dataset_path, self.valid_size)

    def __load_data__(self, dataset_path:Path, valid_size:float) -> Tuple[ImdbDs, ImdbDs, ImdbDs]:
        '''load imdb corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, LivedoorNewsItem]
        '''
        if not (dataset_path / 'livedoor-corpus').exists():
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            self.config.log.dataset_log.info(f'download livedoor corpus from {url}')

            (dataset_path / 'imdb').mkdir(parents=True, exist_ok=True)
            download(url, str(dataset_path / 'aclImdb_v1.tar.gz'))

            with tarfile.open(dataset_path / 'aclImdb_v1.tar.gz', 'r:gz') as tar:
                tar.extractall(path=str(dataset_path / 'imdb'))
        
        train_pos_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'train' / 'pos' / '*.txt'))]
        train_neg_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'train' / 'neg' / '*.txt'))]
        test_pos_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'test' / 'pos' / '*.txt'))]
        test_neg_files = [Path(f) for f in glob(str(dataset_path / 'imdb' / 'test' / 'neg' / '*.txt'))]

        train_data = []
        train_data += [ImdbItem(p, 1) for p in train_pos_files]
        train_data += [ImdbItem(p, 0) for p in train_neg_files]

        test_data = []
        test_data += [ImdbItem(p, 1) for p in test_pos_files]
        test_data += [ImdbItem(p, 0) for p in test_neg_files]

        # split data -> train, valid, test
        train_data, valid_data = train_test_split(train_data, test_size=valid_size)

        # list -> dict
        train_data = {idx: item for idx, item in enumerate(train_data)}
        valid_data = {idx: item for idx, item in enumerate(valid_data)}
        test_data = {idx: item for idx, item in enumerate(test_data)}

        return train_data, valid_data, test_data

    def __load_text__(self, path:PathLike) -> str:
        path: Path = Path(path)
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
        tokens = self.embedding.tokenize(text)
        indices = [self.embedding.token2index(token) for token in tokens]
        embedding = self.embedding.embed(indices)
        return embedding, data.label
