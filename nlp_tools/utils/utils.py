from __future__ import annotations

import json
import os
import random
import shutil
import string
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from glob import glob
from logging import Logger
from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Callable, Optional

import cpuinfo
import mlflow
import nltk
import numpy as np
import torch
import unidic
import yaml
from attrdict import AttrDict
from colorama import Fore, Style
from IPython import get_ipython
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME, MLFLOW_USER
from PIL import Image
from pyunpack import Archive
from torchinfo import summary
from wordcloud import STOPWORDS, WordCloud

from ml_tools.utils.utils import is_notebook, Config
from ml_tools.utils.google_drive import download_from_google_drive, GDriveObjects

if sys.version_info.minor < 11:
    import toml
else:
    import tomllib

if not (Path(nltk.downloader.Downloader().download_dir) / "tokenizers" / "punkt").exists():
    nltk.download("punkt", quiet=True)

if not (Path(nltk.downloader.Downloader().download_dir) / "taggers" / "averaged_perceptron_tagger").exists():
    nltk.download("averaged_perceptron_tagger", quiet=True)

if not Path(unidic.DICDIR).exists():
    subprocess.run(
        "python -m unidic download",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Lang(Enum):
    ENGLISH = "english"
    JAPANESE = "japanese"


class WordCloudMask(Enum):
    RANDOM = [
        "circle.png",
        "doragoslime.png",
        "goldenslime.png",
        "haguremetal.png",
        "haguremetal2.png",
        "kingslime.png",
        "slime.png",
        "slimetower.png",
        "slimetsumuri.png",
    ]
    CIRCLE = ["circle.png"]
    DQ = [
        "doragoslime.png",
        "goldenslime.png",
        "haguremetal.png",
        "haguremetal2.png",
        "kingslime.png",
        "slime.png",
        "slimetower.png",
        "slimetsumuri.png",
    ]


def word_cloud(input_text: str, out_path: str, mask_type=WordCloudMask.RANDOM, **kwargs):
    mask: np.ndarray = get_mask(mask_type)

    font_path = Path(__file__).parent / "resources/fonts/CodeM/CodeM-Regular.ttf"
    if not font_path.exists():
        font_dir = Path(__file__).parent / "resources/fonts/CodeM/"
        font_dir.mkdir(parents=True, exist_ok=True)
        font_name = font_dir / "CodeM-Regular.ttf"
        download_from_google_drive(
            GDriveObjects.CODEM_REGULAR.value,
            str(font_name),
        )

    wc = WordCloud(
        font_path=str(font_path),
        background_color="white",
        max_words=200,
        stopwords=set(STOPWORDS),
        collocations=False,
        contour_width=3,
        contour_color="steelblue",
        mask=mask,
        **kwargs,
    )
    wc.generate(input_text)
    wc.to_file(str(out_path))


def get_mask(mask_type: WordCloudMask) -> np.ndarray:
    mask_dir = Path(__file__).parent / "resources/mask_images"
    mask_file = random.choice(mask_type.value)
    mask_path = mask_dir / mask_file
    mask_image = Image.open(str(mask_path)).convert("L")
    mask = np.array(mask_image, "f")
    mask = (mask > 128) * 255
    return mask


def isint(s: str) -> bool:
    """Check the argument string is integer or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is integer or not.
    """
    try:
        int(s, 10)
    except ValueError:
        return False
    else:
        return True


def isfloat(s: str) -> bool:
    """Check the argument string is float or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is float or not.
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__iter__"):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.strftime("%Y%m%d %H:%M:%S.%f")
        elif isinstance(obj, date):
            return datetime(obj.year, obj.month, obj.day, 0, 0, 0).strftime("%Y%m%d %H:%M:%S.%f")
        else:
            return super().default(obj)
