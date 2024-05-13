import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import nltk
from nltk.corpus import stopwords

from nlp_tools.utils.data import Token
from nlp_tools.utils.utils import Lang

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

PAD = "<PAD>"

JA_PUNCTUATIONS = [c for c in string.punctuation]
JA_PUNCTUATIONS += ["．", "。", "，", "、"]
JA_PUNCTUATIONS += [
    "（",
    "）",
    "「",
    "」",
    "【",
    "】",
    "『",
    "』",
    "［",
    "］",
    "｛",
    "｝",
    "〈",
    "〉",
    "〔",
    "〕",
    "《",
    "》",
    "＜",
    "＞",
]
JA_PUNCTUATIONS += [
    "！",
    "＠",
    "？",
    "：",
    "；",
    "”",
    "’",
    "ー",
    "〜",
    "-",
    "~",
    "−",
    "・",
    "＄",
    "％",
    "＾",
    "＆",
    "／",
    "＝",
    "＼",
]

JA_STOPWORDS = stopwords.words(Lang.ENGLISH.value)
JA_STOPWORDS += [
    ss.strip() for ss in open(Path(__file__).parent / "resources/texts/JA_STOPWORDS_SLOTHLIB.txt") if ss.strip() != ""
]


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
        pass
