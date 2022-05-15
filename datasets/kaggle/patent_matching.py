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


class PatentMatchingDataset(BaseDataset):
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
        ds_dir = dataset_path / "kaggle" / "patent-matching"
        cpc_path = ds_dir / "CPC" / "cpc_titles.csv"
        if not ds_dir.exists():
            # check kaggle configuration
            if not check_kaggle_configure():
                kaggle_configure()
            assert check_kaggle_configure(), "Invalid Kaggle configuration -> check ~/.kaggle/kaggle.json"

            # download kaggle data
            ds_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                f"kaggle competitions download -c us-patent-phrase-to-phrase-matching -p {str(ds_dir.expanduser().absolute())}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with zipfile.ZipFile(str(ds_dir / "us-patent-phrase-to-phrase-matching.zip")) as zf:
                zf.extractall(ds_dir)

            # download CPC title list
            cpc_zip_path = ds_dir / "CPC" / "CPCTitleList202202.zip"
            cpc_zip_path.parent.mkdir(parents=True, exist_ok=True)
            download(
                "https://www.cooperativepatentclassification.org/sites/default/files/cpc/bulk/CPCTitleList202202.zip",
                cpc_zip_path.parent.expanduser().absolute(),
            )
            with zipfile.ZipFile(str(cpc_zip_path.expanduser().absolute())) as zf:
                cpc_title_data = []
                for cpc_section_file in zf.filelist:
                    if cpc_section_file.filename.endswith(".txt"):
                        with zf.open(cpc_section_file) as section_f:
                            for line in section_f:
                                items = str(line).strip().split("\t")
                                if len(items) == 2:
                                    cpc_title_data.append({"code": items[0], "title": items[1]})
                                elif len(items) == 3:
                                    cpc_title_data.append({"code": items[0], "title": items[2]})
                for item in cpc_title_data:
                    code: str = item["code"]
                    item["main_group"] = code.split("/")[-1] if "/" in code else ""
                    item["group"] = code.split("/")[0][4:] if len(code) >= 5 else ""
                    item["subclass"] = code[3] if len(code) >= 4 else ""
                    item["code_class"] = code[1:3] if len(code) >= 3 else ""
                    item["section"] = code[0] if len(code) >= 1 else ""
                pd.DataFrame(cpc_title_data).to_csv(str(cpc_path), header=True, index=False)

        train_df = pd.read_csv(str(ds_dir / "train.csv"), header=0)
        test_df = pd.read_csv(str(ds_dir / "test.csv"), header=0)
        cpc_titles = pd.read_csv(cpc_path, header=0)

        train_df = train_df.merge(cpc_titles, left_on="context", right_on="code")
        test_df = test_df.merge(cpc_titles, left_on="context", right_on="code")

        train_set = [Item(row["id"], row["anchor"], row["target"], row["context"], row["score"], row["title"]) for _, row in train_df.iterrows()]
        test_set = [Item(row["id"], row["anchor"], row["target"], row["context"], row["score"], row["title"]) for _, row in test_df.iterrows()]

        train_set, valid_set = train_test_split(train_set, test_size=self.config.train_valid_size)

        # list -> dict
        train_dict = {idx: item for idx, item in enumerate(train_set)}
        valid_dict = {idx: item for idx, item in enumerate(valid_set)}
        test_dict = {idx: item for idx, item in enumerate(test_set)}

        return train_dict, valid_dict, test_dict

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
