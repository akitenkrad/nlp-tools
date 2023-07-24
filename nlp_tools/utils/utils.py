from __future__ import annotations

import json
import os
import random
import shutil
import string
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from glob import glob
from logging import Logger
from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Optional

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

from nlp_tools.utils.logger import get_logger, kill_logger

NVIDIA_SMI_DEFAULT_ATTRIBUTES = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
)

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


def now() -> datetime:
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMInteractiveShell":
            return True  # Jupyter notebook qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal ipython
        elif "google.colab" in sys.modules:
            return True  # Google Colab
        else:
            return False
    except NameError:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Lang(Enum):
    ENGLISH = "english"
    JAPANESE = "japanese"


class Phase(Enum):
    DEV = "dev"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    SUBMISSION = "submission"


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


def get_enum_from_value(enum_class, value: Any):
    value_map = {item.value: item for item in enum_class}
    return value_map[value]


def filepath_to_uri(path: Path) -> str:
    return urllib.parse.urljoin("file:", urllib.request.pathname2url(str(path.resolve().absolute())))


class MlflowWriter:
    def __init__(self, exp_name: str, tracking_uri: str, logger: Optional[Logger] = None):
        self.__exp_name = exp_name
        self.__tracking_uri = filepath_to_uri(Path(tracking_uri))
        self.__logger = logger
        self.__print = lambda x: print(x) if self.__logger is None else lambda x: self.__logger.info(x)

    def __initialize__(self):
        self.client = MlflowClient(tracking_uri=self.__tracking_uri)
        self.run = None
        self.run_id = None
        try:
            self.exp_id = self.client.create_experiment(self.__exp_name)
        except Exception as e:
            self.__print(e)
            self.exp_id = self.client.get_experiment_by_name(self.__exp_name).experiment_id

        self.experiment = self.client.get_experiment(self.exp_id)

        self.__print("New experiment started")
        self.__print(f"Name: {self.experiment.name}")
        self.__print(f"Experiment ID: {self.experiment.experiment_id}")
        self.__print(f"Artifact Location: {self.experiment.artifact_location}")

        mlflow.set_tracking_uri(self.__tracking_uri)
        mlflow.set_experiment(self.__exp_name)

    def initialize(self, tags=None):
        self.__initialize__()
        self.create_new_run(tags=tags)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.exp_id, tags=tags)
        assert self.run is not None

        self.run_id = self.run.info.run_id
        self.__print(f"New run started: {self.run.info.run_name}")

        mlflow.tracking.fluent._active_run_stack.append(self.run)

    def terminate(self):
        self.client.set_terminated(self.run_id, RunStatus.to_string(RunStatus.FINISHED))

    def log_param(self, key: str, value: Any):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key: str, value: Any, step=None):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path: str):
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str = ""):
        self.client.log_dict(self.run_id, dictionary, artifact_file)

    def log_text(self, text: str, artifact_file: str):
        self.client.log_text(self.run_id, text, artifact_file)

    def log_pytorch_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)


@dataclass
class LrFinderSettings(object):
    initial_value: float = 1e-8
    final_value: float = 10
    beta: float = 0.98

    @classmethod
    def from_dict(cls, config: dict):
        settings = cls()
        if "initial_value" in config:
            settings.initial_value = float(config["initial_value"])
            settings.final_value = float(config["final_value"])
            settings.beta = float(config["beta"])
        return settings


@dataclass
class TrainSettings(object):
    device: torch.device = torch.device("cpu")
    exp_name: str = "default_exp"
    k_folds: int = 3
    epochs: int = 250
    batch_size: int = 32
    valid_size: float = 0.1
    test_size: float = 0.1
    lr: float = 0.001
    lr_decay: float = 0.9
    early_stop_patience: int = 10
    logging_per_batch: int = 5
    lr_finder_settings: LrFinderSettings = field(default_factory=LrFinderSettings)
    ex_args: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> TrainSettings:
        settings = cls()
        if "device" in config:
            settings.device = torch.device(config["device"])
        elif torch.cuda.is_available():
            settings.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            settings.device = torch.device("mps")
        else:
            settings.device = torch.device("cpu")

        if "exp_name" in config:
            settings.exp_name = str(config["exp_name"])
        if "k_fold" in config:
            settings.k_folds = int(config["k_folds"])
        if "epochs" in config:
            settings.epochs = int(config["epochs"])
        if "batch_size" in config:
            settings.batch_size = int(config["batch_size"])
        if "valid_size" in config:
            settings.valid_size = float(config["valid_size"])
        if "test_size" in config:
            settings.test_size = float(config["test_size"])
        if "lr" in config:
            settings.lr = float(config["lr"])
        if "lr_decay" in config:
            settings.lr_decay = float(config["lr_decay"])
        if "early_stop_patience" in config:
            settings.early_stop_patience = int(config["early_stop_patience"])
        if "logging_per_batch" in config:
            settings.logging_per_batch = int(config["logging_per_batch"])
        if "lr_finder_settings" in config:
            settings.lr_finder_settings = LrFinderSettings.from_dict(config["lr_finder_settings"])

        default_fields = list(settings.__dataclass_fields__.keys())
        for key, value in config.items():
            if key not in default_fields:
                settings.ex_args[key] = value

        return settings


@dataclass
class LogSettings(object):
    log_dir: Path = Path("__logs__")
    log_filename: str = "system.log"
    log_file: Path = Path("")
    mlflow_dir: Path = Path("__logs__/mlflow")
    backup: bool = False
    backup_dir: Path = Path("__backup__")

    @classmethod
    def from_dict(cls, config: dict, exp_name: str = "exp", timestamp: datetime = datetime.now()) -> LogSettings:
        settings = cls()
        if "log_dir" in config:
            settings.log_dir = Path(config["log_dir"])
        if "log_filename" in config:
            settings.log_filename = str(config["log_filename"])
        if "mlflow_dir" in config:
            settings.mlflow_dir = Path(config["mlflow_dir"])
        if "backup" in config:
            settings.backup = config["backup"]
        if "backup_dir" in config:
            settings.backup_dir = Path(config["backup_dir"])

        settings.log_dir = settings.log_dir / f"{exp_name}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        settings.log_file = settings.log_dir / settings.log_filename

        return settings


@dataclass
class DataSettings(object):
    data_dir: Path = Path("__data__")
    cache_dir: Path = Path("__cache__")
    weights_dir: Path = Path("__weights__")
    output_dir: Path = Path("__output__")

    @classmethod
    def from_dict(cls, config: dict) -> DataSettings:
        settings = cls()
        if "data_dir" in config:
            settings.data_dir = Path(config["data_dir"])
        if "cache_dir" in config:
            settings.cache_dir = Path(config["cache_dir"])
        if "weights_dir" in config:
            settings.weights_dir = Path(config["weights_dir"])
        if "output_dir" in config:
            settings.output_dir = Path(config["output_dir"])

        return settings


@dataclass
class Config(object):
    logger: Logger
    mlflow_writer: MlflowWriter
    config_path: Path = Path("")
    timestamp: datetime = field(default_factory=datetime.now)
    train_settings: TrainSettings = field(default_factory=TrainSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    log_settings: LogSettings = field(default_factory=LogSettings)
    ex_logger: AttrDict = field(default_factory=lambda: AttrDict({}))

    __TEXT = "config: {KEY_1:15s} - {KEY_2:20s}: {VALUE}"

    @classmethod
    def get_hash(cls, size: int = 12) -> str:
        chars = string.ascii_lowercase + string.digits
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    @classmethod
    def now(cls) -> datetime:
        JST = timezone(timedelta(hours=9))
        return datetime.now(JST)

    @classmethod
    def generate(cls, config_path: str, silent: bool = False, extra_config: dict[str, Any] = {}) -> Config:
        settings = cls(logger=Logger(""), mlflow_writer=MlflowWriter("", ""), config_path=Path(config_path))

        with open(config_path, mode="rb") as f:
            config: dict = tomllib.load(f)
            if len(extra_config) > 0:
                config.update(extra_config)

        # set attributes
        settings.timestamp = settings.now()
        if "yaml_path" in config:
            settings.config_path = Path(config["yaml_path"])
        if "train_settings" in config:
            settings.train_settings = TrainSettings.from_dict(config["train_settings"])
        if "log_settings" in config:
            settings.log_settings = LogSettings.from_dict(
                config["log_settings"], exp_name=settings.train_settings.exp_name, timestamp=settings.timestamp
            )
        if "data_settings" in config:
            settings.data_settings = DataSettings.from_dict(config["data_settings"])

        # set logger
        if hasattr(settings, "logger") and isinstance(settings.logger, Logger):
            kill_logger(settings.logger)
        settings.logger = get_logger(name="config", logfile=str(settings.log_settings.log_file), silent=silent)

        # show config
        settings.logger.info("====== show config =========")
        settings.logger.info(settings.__TEXT.format(KEY_1="root", KEY_2="config_path", VALUE=settings.config_path))
        settings.logger.info(settings.__TEXT.format(KEY_1="root", KEY_2="timestamp", VALUE=settings.timestamp))
        for key in settings.log_settings.__dataclass_fields__.keys():
            settings.logger.info(
                settings.__TEXT.format(KEY_1="log_settings", KEY_2=key, VALUE=getattr(settings.log_settings, key))
            )
        for key in settings.data_settings.__dataclass_fields__.keys():
            settings.logger.info(
                settings.__TEXT.format(KEY_1="data_settings", KEY_2=key, VALUE=getattr(settings.data_settings, key))
            )
        for key in settings.train_settings.__dataclass_fields__.keys():
            settings.logger.info(
                settings.__TEXT.format(KEY_1="train_settings", KEY_2=key, VALUE=getattr(settings.train_settings, key))
            )
        settings.logger.info("============================")

        # CPU info
        settings.describe_cpu()

        # NVIDIA GPU info
        if torch.cuda.is_available():
            settings.describe_gpu()

        # mkdir
        settings.__mkdirs()

        return settings

    def __mkdirs(self):
        self.log_settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.mlflow_dir.mkdir(parents=True, exist_ok=True)
        self.data_settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_settings.weights_dir.mkdir(parents=True, exist_ok=True)
        self.data_settings.output_dir.mkdir(parents=True, exist_ok=True)

    def print(self, x):
        if self.logger is None:
            print(x)
        else:
            self.logger.info(x)

    def init_mlflow(self):
        self.mlflow_writer = MlflowWriter(
            exp_name=self.train_settings.exp_name,
            tracking_uri=str(self.log_settings.mlflow_dir.resolve().absolute()),
            logger=self.logger,
        )
        self.mlflow_writer.initialize()

    def close_mlflow(self):
        self.mlflow_writer.terminate()

    def describe_cpu(self):
        self.print("====== cpu info ============")
        for key, value in cpuinfo.get_cpu_info().items():
            self.print(f"CPU INFO: {key:20s}: {value}")
        self.print("============================")

    def describe_gpu(self, nvidia_smi_path="nvidia-smi", no_units=True):
        try:
            keys = NVIDIA_SMI_DEFAULT_ATTRIBUTES
            nu_opt = "" if not no_units else ",nounits"
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True)
            raw_lines = [line.strip() for line in output.decode().split("\n") if line.strip() != ""]
            lines = [{k: v for k, v in zip(keys, line.split(", "))} for line in raw_lines]

            self.print("====== show GPU information =========")
            for line in lines:
                for k, v in line.items():
                    self.print(f"{k:25s}: {v}")
            self.print("=====================================")
        except CalledProcessError:
            self.print("====== show GPU information =========")
            self.print("  No GPU was found.")
            self.print("=====================================")

    def describe_m1_silicon(self):
        self.log.logger.info("====== show GPU information =========")
        self.log.logger.info("  Mac-M1 GPU is available.")
        self.log.logger.info("=====================================")

    def describe_model(self, model: torch.nn.Module, input_size: Optional[tuple[int]] = None, input_data=None):
        if input_data is None:
            summary_str = summary(
                model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )
        else:
            summary_str = summary(
                model,
                input_data=input_data,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )

        for line in summary_str.__str__().split("\n"):
            self.print(line)

    def backup_logs(self):
        """copy log directory to config.backup"""
        backup_dir = Path(self.log_settings.backup_dir)
        if backup_dir.exists():
            shutil.rmtree(str(backup_dir))
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.log_settings.log_dir, self.log_settings.backup_dir)

    def add_logger(self, name: str, silent: bool = False):
        self.ex_logger[name] = get_logger(name=name, logfile=str(self.log_settings.log_file), silent=silent)

    def fix_seed(self, seed=42):
        self.print(self.__TEXT.format(KEY_1="root", KEY_2="seed", VALUE=seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def timedelta2HMS(total_sec: int) -> str:
    h = total_sec // 3600
    m = total_sec % 3600 // 60
    s = total_sec % 60
    return f"{h:2d}h {m:2d}m {s:2d}s"


def __show_progress__(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = "=" * bar_num
    if bar_num != max_bar:
        progress_element += ">"
    bar_fill = " "
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        Fore.LIGHTCYAN_EX,
        f"[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r",
        end="",
    )


def download(url: str, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    print(Fore.LIGHTGREEN_EX, "download from:", end="")
    print(Fore.WHITE, url)
    urllib.request.urlretrieve(url, filepath, __show_progress__)
    print("")  # 改行
    print(Style.RESET_ALL, end="")


def un7zip(src_path: str, dst_path: str):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    Archive(src_path).extractall(dst_path)
    for dirname, _, filenames in os.walk(dst_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def word_cloud(input_text: str, out_path: str, mask_type=WordCloudMask.RANDOM):
    mask: np.ndarray = get_mask(mask_type)

    font_path = Path(__file__).parent / "resources/fonts/Utatane_v1.1.0/Utatane-Regular.ttf"
    if not font_path.exists():
        font_dir = Path(__file__).parent / "resources/fonts"
        utatane_7z = font_dir / "Utatane_v1.1.0.7z"
        download(
            "https://github.com/nv-h/Utatane/releases/download/Utatane_v1.1.0/Utatane_v1.1.0.7z",
            str(utatane_7z),
        )
        un7zip(str(utatane_7z), str(font_dir))

    wc = WordCloud(
        font_path=str(font_path),
        background_color="white",
        max_words=200,
        stopwords=set(STOPWORDS),
        collocations=False,
        contour_width=3,
        contour_color="steelblue",
        mask=mask,
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
