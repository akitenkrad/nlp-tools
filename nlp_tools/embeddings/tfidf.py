from collections import namedtuple
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_tools.embeddings.base import Embedding
from nlp_tools.utils.data import Token
from nlp_tools.utils.tokenizers import WordTokenizer, WordTokenizerFactory
from nlp_tools.utils.utils import Config, Lang, download


class Tfidf(Embedding):
    def __init__(
        self,
        config: Config,
        max_sent_len=-1,
        padding="<PAD>",
        stemming=False,
        stop_words=None,
        language=Lang.ENGLISH,
        **kwargs
    ):
        self.config = config
        self.config.add_logger("tfidf_log")
        self.tokenizer: WordTokenizer = WordTokenizerFactory.get_tokenizer(
            pad=padding, max_sent_len=max_sent_len, stemming=stemming, language=language, **kwargs
        )
        self.vectorizer = TfidfVectorizer(
            encoding="utf-8", tokenizer=self.tokenizer.tokenize, analyzer="word", stop_words=stop_words
        )

    @property
    def embedding_dim(self) -> int:
        return len(self.tokens)

    @property
    def tokens(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def index2token(self, index: int) -> str:
        raise NotImplementedError('Tfidf does not have "index2token()".')

    def token2index(self, token: str) -> int:
        raise NotImplementedError('Tfidf does not have "token2index()".')

    def word_tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def word_embed(self, input_text):
        """embed token index list into vector

        Args:
            input_text (str): input text

        Returns:
            embedded vector. np.array: (embedding_dim)
        """

        embed_vector = self.vectorizer.transform([input_text]).squeeze()
        return embed_vector

    def char_tokenize(self, text):
        pass

    def char_embed(self, input_text):
        pass

    def fit(self, raw_documents: Iterable[str]):
        self.vectorizer.fit(raw_documents)
