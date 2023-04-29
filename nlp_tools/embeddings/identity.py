from collections import Counter
from typing import List

import numpy as np
from embeddings.base import Embedding
from utils.tokenizers import WordTokenizer
from utils.utils import Lang


class IdentityEmbedding(Embedding):
    def __init__(
        self,
        pad="<pad>",
        unk="<unk>",
        sos="<sos>",
        eos="<eos>",
        max_sent_len=-1,
        language=Lang.ENGLISH,
        remove_stopwords=False,
        remove_punctuations=True,
        stemming=False,
        add_tag=False,
    ):
        self.__token2idx = {}
        self.__idx2token = {}
        self.counter = Counter()
        self.PAD = pad
        self.UNK = unk
        self.SOS = sos
        self.EOS = eos
        self.tokenizer = WordTokenizer(
            pad=pad,
            max_sent_len=max_sent_len,
            language=language,
            remove_stopwords=remove_stopwords,
            remove_punctuations=remove_punctuations,
            stemming=stemming,
            add_tag=add_tag,
        )
        self.add_token(pad)
        self.add_token(unk)
        self.add_token(sos)
        self.add_token(eos)

    def __len__(self):
        return len(self.__token2idx)

    @property
    def __next_token_idx(self):
        if len(self.__idx2token) < 1:
            return 0
        return max(self.__idx2token) + 1

    @property
    def embedding_dim(self) -> int:
        """embedding dim"""
        return -1

    @property
    def tokens(self) -> List[str]:
        """token list"""
        return list(self.__token2idx.keys())

    def add_token(self, token):
        """used to add a token from corpus"""
        if token not in self.__token2idx:
            new_idx = self.__next_token_idx
            self.__token2idx[token] = new_idx
            self.__idx2token[new_idx] = token
        self.counter.update(token)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def token2index(self, token: str) -> int:
        if token not in self.__token2idx:
            return self.__token2idx[self.UNK]
        else:
            return self.__token2idx[token]

    def index2token(self, index: int) -> str:
        return self.__idx2token[index]

    def embed(self, input_text: str) -> np.ndarray:
        return np.array([self.token2index(token) for token in self.tokenize(input_text)])
