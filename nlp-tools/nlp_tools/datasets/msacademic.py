import bz2
import dataclasses
import json
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from torch.utils.data import Dataset
from utils.data import ConferenceText
from utils.google_drive import GDriveObjects, download_from_google_drive
from utils.tokenizers import Tokenizer
from utils.utils import Lang, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


@dataclasses.dataclass
class Table(object):
    name: str
    max_digit: int


class MsAcademicTable(Enum):
    Affiliations = Table("Affiliations", -1)
    Authors = "Authors"
    ConferenceInstances = "ConferenceInstances"
    ConferenceSeries = "ConferenceSeries"
    Journals = "Journals"
    PaperAuthorAffiliations = "PaperAuthorAffiliations"
    PaperExtendedAttributes = "PaperExtendedAttributes"
    PaperReferences = "PaperReferences"
    PaperResources = "PaperResources"
    Papers = "Papers"
    PaperUrls = "PaperUrls"
    EntityRelatedEntities = "EntityRelatedEntities"
    FieldOfStudyChildren = "FieldOfStudyChildren"
    FieldOfStudyExtendedAttributes = "FieldOfStudyExtendedAttributes"
    FieldsOfStudy = Table("FieldsOfStudy", 10)
    PaperFieldsOfStudy = "PaperFieldsOfStudy"
    PaperRecommendations = "PaperRecommendations"
    RelatedFieldOfStudy = "RelatedFieldOfStudy"
    PaperCitationContexts = "PaperCitationContexts"
    PaperTags = "PaperTags"


def triple2hdf5(src_triples: PathLike, dst_dir: PathLike):
    """convert MS-ACADEMIC triples into hdf5"""

    def get_field(text: str):
        field = text.split("/")[-1]
        if "#" in field:
            field = field.split("#")[-1]
        return field

    type_map = {
        "integer": int,
        "string": str,
        "date": str,
    }
    hdf5_type_map = {
        "integer": int,
        "string": h5py.string_dtype(encoding="utf-8"),
        "date": h5py.string_dtype(encoding="utf-8"),
    }

    src_path = Path(src_triples)
    out_dir = Path(dst_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_name = src_path.stem.replace(".nt", "")

    ptn = re.compile(r'(".+?"\^\^)?<(.+?)>')
    # write into hdf5
    # 1. calculate max digit
    with bz2.open(src_path, mode="rt", encoding="utf-8") as rf:
        total = 0
        keys = []
        for triple in tqdm(rf, leave=False, desc="Counting keys"):
            items = ptn.findall(triple)
            num_key = int(get_field(items[0][1]))
            keys.append(num_key)
            total += 1
        keys = list(set(keys))
        max_digit = int(np.ceil(np.log10(max(keys))))

    # 2. convert triples -> hdf5
    with bz2.open(src_path, mode="rt", encoding="utf-8") as rf:
        for triple in tqdm(rf, leave=False, desc="Converting triples -> htf5", total=total):
            items = ptn.findall(triple)
            if not items[2][0]:
                continue
            _key = get_field(items[0][1])
            field_name = get_field(items[1][1])
            value = items[2][0][1:-3]
            data_type = get_field(items[2][1])

            # configure key
            key: str = "0" * max_digit + str(_key)
            key = key[-max_digit:]
            assert _key in key

            # write into hdf5
            out_hdf5 = out_dir / table_name / key[0] / key[1] / key[2] / f"{key[:3]}.hdf5"
            out_hdf5.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_hdf5, mode="w") as h5wf:
                # create group
                group = h5wf.require_group(key)
                # create dataset
                dataset = group.require_dataset(name=field_name, shape=(1,), dtype=hdf5_type_map[data_type])
                dataset[0] = type_map[data_type](value.replace('"', ""))
