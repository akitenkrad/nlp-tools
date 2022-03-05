from typing import Iterable, List
from collections import namedtuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.utils import download, Config
from utils.tokenizers import WordTokenizer
from embeddings.base import Embedding

class Tfidf(Embedding):

    def __init__(self, config:Config, max_sent_len=-1, padding='<PAD>', stemming=False, stop_words=None):
        self.config = config
        self.config.add_logger('tfidf_log')
        self.tokenizer = WordTokenizer(pad=padding, max_sent_len=max_sent_len, stemming=stemming)
        self.vectorizer = TfidfVectorizer(encoding='utf-8',
                                          tokenizer=self.tokenizer.tokenize,
                                          analyzer='word',
                                          stop_words=stop_words)

    @property
    def embedding_dim(self) -> int:
        return len(self.tokens)

    @property
    def tokens(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def tokenize(self, text:str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def index2token(self, index: int) -> str:
        raise NotImplementedError('Tfidf does not have "index2token()".')

    def token2index(self, token: str) -> int:
        raise NotImplementedError('Tfidf does not have "token2index()".')

    def embed(self, input_text:str) -> np.ndarray:
        '''embed token index list into vector
        
        Args:
            input_text (str): input text
        
        Returns:
            embedded vector. np.array: (embedding_dim)
        '''

        embed_vector = self.vectorizer.transform([input_text]).squeeze()
        return embed_vector

    def fit(self, raw_documents:Iterable[str]):
        self.vectorizer.fit(raw_documents)
