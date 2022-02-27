from typing import List
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class Embedding(ABC):

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        '''embedding dim'''
        pass

    @property
    @abstractmethod
    def tokens(self) -> List[str]:
        '''token list'''
        pass

    @abstractmethod
    def tokenize(self, text:str) -> List[str]:
        '''tokenize the input text
        
        Args:
            text (str): input sentence.

        Returns:
            tokenized list of tokens. list[str].
        '''
        pass

    @abstractmethod
    def token2index(self, token:str) -> int:
        '''convert token to index
        
        Args:
            token (str): input token

        Returns:
            index (int).
        '''
        pass

    @abstractmethod
    def index2token(self, index:int) -> str:
        '''convert index to token
        
        Args:
            index (int): input index

        Returns:
            token (str).
        '''
        pass

    @abstractmethod
    def embed(self, index_list:List[int]) -> np.ndarray:
        '''embed token index list into vector
        
        Args:
            index_list (list[int]): input index list which was converted from tokens
        
        Returns:
            embedded vector. np.array: (text_length, embedding_dim)
        '''
        pass
