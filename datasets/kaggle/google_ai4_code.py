import json
import os
import subprocess
import zipfile
from collections import namedtuple
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets.base import BaseDataset
from datasets.kaggle.utils import check_kaggle_configure, kaggle_configure
from embeddings.base import Embedding
from sklearn.model_selection import train_test_split

from utils.utils import Config, Phase, download

Item = namedtuple("Item", ("id", "anchor", "target", "context", "score", "title"))
ItemSet = Dict[int, Item]


class GoogleAI4Code(BaseDataset):
    def __init__(self, config: Config, embedding: Embedding):
        super().__init__(config, Phase.TRAIN)
        self.embedding = embedding
        self.n_class = 2

        self.train_data, self.valid_data, self.test_data = self.__load_data__(self.dataset_path)
        self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path: Path) -> Tuple[ItemSet, ItemSet, ItemSet]:
        """load kaggle - patent phrase-to-phrase matching dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, ImdbItem]
        """
        ds_dir = dataset_path / "kaggle" / "google-ai4-code"
        if not ds_dir.exists():
            # check kaggle configuration
            if not check_kaggle_configure():
                kaggle_configure()
            assert check_kaggle_configure(), "Invalid Kaggle configuration -> check ~/.kaggle/kaggle.json"

            # download kaggle data
            ds_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                f"kaggle competitions download -c AI4Code -p {str(ds_dir.expanduser().absolute())}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with zipfile.ZipFile(str(ds_dir / "AI4Code.zip")) as zf:
                zf.extractall(ds_dir)

    def __getitem__(self, index):
        if self.phase == Phase.TRAIN or self.phase == Phase.DEV:
            data: Item = self.train_data[index]
        elif self.phase == Phase.VALID:
            data: Item = self.valid_data[index]
        elif self.phase == Phase.TEST or self.phase == Phase.SUBMISSION:
            data: Item = self.test_data[index]
        else:
            raise ValueError(f"Invalid Phase: {self.phase}")

        # TODO: convert to tensor
        # embedding = self.embedding.embed(text)
        # return embedding, data.label
