import gzip
import pickle
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm

from embeddings.base import Embedding
from utils.data import Sentence, Token
from utils.tokenizers import CharTokenizer, CharTokenizerFactory, WordTokenizer, WordTokenizerFactory
from utils.utils import Config, Lang, download

FastTextInfo = namedtuple("FastTextInfo", ("filename", "language", "embedding_dim"))


class FastTextType(Enum):
    CC_EN_300 = FastTextInfo("cc.en.300.vec.gz", Lang.ENGLISH, 300)
    CC_JA_300 = FastTextInfo("cc.ja.300.vec.gz", Lang.JAPANESE, 300)


class FastText(Embedding):
    def __init__(
        self,
        config: Config,
        fast_text_type: FastTextType,
        max_sent_len=-1,
        max_word_len=-1,
        remove_punctuations=True,
        remove_stopwords=False,
        no_cache=False,
        filter=None,
    ):
        """
        filter example:
            >>> filter=lambda tk: tk.pos_tag=="名詞" or tk.pos_tag=="形容詞"    # CC_JA_300
            >>> filter=lambda tok: tok.pos_tag.startswith("N") or tok.pos_tag.startswith("J")   # CC_EN_300
        """
        self.config = config
        self.config.add_logger("fast_text_log")
        self.fast_text_type: FastTextType = fast_text_type
        self.weights_path: Path = Path(config.weights.global_weights_dir) / "fast_text"
        self.word_tokenizer: WordTokenizer = WordTokenizerFactory.get_tokenizer(
            language=fast_text_type.value.language,
            pad="PAD",
            max_sent_len=max_sent_len,
            remove_punctuations=remove_punctuations,
            remove_stopwords=remove_stopwords,
            filter=filter,
        )
        self.char_tokenizer: CharTokenizer = CharTokenizerFactory.get_tokenizer(
            language=fast_text_type.value.language,
            pad="PAD",
            max_sent_len=max_sent_len,
            max_word_len=max_word_len,
            remove_punctuations=remove_punctuations,
            remove_stopwords=remove_stopwords,
            filter=filter,
        )

        self.__vectors, self.__words, self.__word2idx = self.__load_fast_text__(no_cache)
        self.__idx2word = {i: w for w, i in self.__word2idx.items()}

    @property
    def embedding_dim(self) -> int:
        return self.fast_text_type.value.embedding_dim

    @property
    def tokens(self) -> List[str]:
        return self.__words

    @property
    def weights(self) -> np.ndarray:
        return self.__vectors

    def word_tokenize(self, text: str) -> List[Token]:
        return self.word_tokenizer.tokenize(text)

    def char_tokenize(self, text: str) -> List[List[Token]]:
        return self.char_tokenizer.tokenize(text)

    def index2token(self, index: int) -> str:
        if index not in self.__idx2word:
            return "UNKNOWN"
        else:
            return self.__idx2word[index]

    def token2index(self, token: str) -> int:
        if token not in self.__word2idx:
            return -1
        else:
            return self.__word2idx[token]

    def word_embed(self, input_text: str) -> np.ndarray:
        tokens: List[Token] = self.word_tokenize(input_text)
        indices: List[int] = [self.token2index(token.surface) for token in tokens]
        embed_vector: np.ndarray = np.array([self.__vectors[idx] for idx in indices])
        return embed_vector

    def char_embed(self, input_text: str) -> List[np.ndarray]:
        words: List[List[Token]] = self.char_tokenize(input_text)
        indices: List[List[int]] = [[self.token2index(token.surface) for token in word] for word in words]
        embed_vector: List[np.ndarray] = [np.array([self.__vectors[idx] for idx in index]) for index in indices]
        return embed_vector

    def __load_fast_text__(self, no_cache=False) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """load pretrained FastText weights from https://dl.fbaipublicfiles.com

        Args:
            no_cache (bool): if True, download weights from the original source.

        Returns:
            vectors: np.ndarray (vocab_size, embedding_dim)
            words: list[str]
            word2idx: dict[word, index]
        """
        weights_path = self.weights_path / self.fast_text_type.value.filename

        if not weights_path.exists():
            self.config.log.fast_text_log.info("download FastText weights from the Internet.")
            url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{self.fast_text_type.value.filename}"
            download(url, weights_path)

        # cache path
        cache_name = self.fast_text_type.value.filename.replace(".vec.gz", "")
        cache_dir = self.config.data.cache_path / "fast_text" / f"fast_text.{cache_name}"
        vector_cache = cache_dir / f"fast_text.{cache_name}.vectors.pickle"
        words_cache = cache_dir / f"fast_text.{cache_name}.words.pickle"
        word2idx_cache = cache_dir / f"fast_text.{cache_name}.word2idx.pickle"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not no_cache and vector_cache.exists() and words_cache.exists() and word2idx_cache.exists():
            self.config.log.fast_text_log.info("restore weights from cache.")
            vectors = pickle.load(open(str(vector_cache), "rb"))
            words = pickle.load(open(str(words_cache), "rb"))
            word2idx = pickle.load(open(str(word2idx_cache), "rb"))
        else:
            self.config.log.fast_text_log.info("construct weights from the weights file... this takes a few minutes...")
            weights: KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(weights_path, binary=False)

            vectors = weights.vectors
            words = [""] * len(weights.index_to_key)
            word2idx = {}
            for index, word in enumerate(weights.index_to_key):
                words[index] = word
                word2idx[word] = index

            # cache weights
            pickle.dump(vectors, open(str(vector_cache), "wb"))
            pickle.dump(words, open(str(words_cache), "wb"))
            pickle.dump(word2idx, open(str(word2idx_cache), "wb"))

        self.config.log.fast_text_log.info(f"Finished loading fast_text tokens: total={len(words)} words.")
        return np.array(vectors), words, word2idx

    def get_gensim_model(self) -> KeyedVectors:
        weights_path = self.weights_path / self.fast_text_type.value.filename
        weights = gensim.models.KeyedVectors.load_word2vec_format(weights_path, binary=False)
        return weights
