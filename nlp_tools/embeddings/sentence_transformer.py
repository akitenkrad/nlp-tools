import pickle
import zipfile
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from embeddings.base import Embedding
from sentence_transformers import SentenceTransformer as _SentenceTransformer
from tqdm import tqdm

from nlp_tools.utils.tokenizers import WordTokenizer
from nlp_tools.utils.utils import Config


class SentenceTransformer(Embedding):
    def __init__(self, config: Config, model: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.model = _SentenceTransformer(model)

        self._embedding_dim = self.model("This is a test sentence.").shape[0]
        self.__idx2word = self.model.tokenizer.vocab
        self.__word2idx = {v: i for i, v in self.__idx2word.items()}

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def tokens(self) -> List[str]:
        return list(self.model.tokenizer.vocab.keys())

    def tokenize(self, text: str):
        return self.model.tokenize(text)

    def index2token(self, index: int) -> str:
        if index not in self.__idx2word:
            return "<unk>"
        else:
            return self.__idx2word[index]

    def token2index(self, token: str) -> int:
        if token not in self.__word2idx:
            return -1
        else:
            return self.__word2idx[token]

    def embed(self, input_text: str) -> np.ndarray:
        return self.model.encode(input_text)
