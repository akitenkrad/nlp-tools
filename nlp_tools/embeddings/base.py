from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union

import numpy as np

from nlp_tools.utils.data import Token


class Embedding(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """embedding dim"""
        pass

    @property
    @abstractmethod
    def tokens(self) -> List[str]:
        """token list"""
        pass

    @abstractmethod
    def word_tokenize(self, text: str) -> List[Token]:
        """tokenize the input text into a list of words

        Args:
            text (str): input sentence.

        Returns:
            tokenized list of tokens. List[Token].
        """
        pass

    @abstractmethod
    def char_tokenize(self, text: str) -> List[List[Token]]:
        """tokenize the input text into lists of characters

        Args:
            text (Text): input sentence.

        Returns:
            tokenized list of tokens. List[List[Token]].
        """
        pass

    @abstractmethod
    def token2index(self, token: str) -> int:
        """convert token to index

        Args:
            token (str): input token

        Returns:
            index (int).
        """
        pass

    @abstractmethod
    def index2token(self, index: int) -> str:
        """convert index to token

        Args:
            index (int): input index

        Returns:
            token (str).
        """
        pass

    @abstractmethod
    def word_embed(self, input_text: str) -> np.ndarray:
        """embed token index list into vector

        Args:
            input_text (Text): input text as Text object

        Returns:
            embedded vector. np.array: (text_length, embedding_dim)
        """
        pass

    @abstractmethod
    def char_embed(self, input_text: str) -> List[np.ndarray]:
        """embed token index list into vector

        Args:
            input_text (Text): input text as Text object

        Returns:
            embedded vector. np.array: (text_length, word_length, embedding_dim)
        """
        pass
