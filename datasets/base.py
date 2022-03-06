from typing import Any
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
import numpy as np

from torchvision.transforms import transforms
from torch.utils.data import Dataset

from utils.logger import get_logger, Logger
from utils.utils import Phase, is_colab, Config

if is_colab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class BaseDataset(ABC, Dataset):
    def __init__(self, config:Config, phase:Phase=Phase.TRAIN):
        super().__init__()
        self.config = config
        self.config.add_logger('dataset_log')
        self.dataset_path = Path(config.dataset_path)
        self.valid_size = config.valid_size
        self.test_size = config.test_size
        self.phase = phase

        # self.train_data, self.valid_data, self.test_data = self.__load_data__(self.dataset_path)
        # self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path:PathLike, test_size:float, valid_size:float):
        '''load train and test data
        
        Returns:
            train_data, valid_data, test_data: dict[index, namedtuple[data, label]]
        '''
        # return train_data, test_data, label_data
        raise NotImplementedError()

    def __len__(self) -> int:
        if self.phase == Phase.TRAIN:
            return len(self.train_data)
        elif self.phase == Phase.VALID:
            return len(self.valid_data)
        elif self.phase == Phase.TEST:
            return len(self.test_data)
        elif self.phase == Phase.DEV:
            return len(self.dev_data)
        elif self.phase == Phase.SUBMISSION:
            return len(self.test_data)
        raise RuntimeError(f'Unknown phase: {self.phase}')

    def __getitem__(self, index) -> Any:
        if self.phase == Phase.TRAIN:
            raise NotImplementedError()
        elif self.phase == Phase.TEST:
            raise NotImplementedError()
        raise RuntimeError(f'Unknown phase: {self.phase}')

    # phase change functions
    def to_train(self):
        self.phase = Phase.TRAIN
    def to_valid(self):
        self.phase = Phase.VALID
    def to_test(self):
        self.phase = Phase.TEST
    def to_dev(self):
        self.phase = Phase.DEV
    def to_submittion(self):
        self.phase = Phase.SUBMISSION

    def refresh(self):
        pass

    @staticmethod
    def collate_fn():
        pass
