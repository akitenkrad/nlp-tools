from typing import List
from pathlib import Path
import pickle
from tqdm import tqdm
from enum import Enum
from collections import namedtuple
import numpy as np
import zipfile

from utils.utils import download, Config
from utils.tokenizers import WordTokenizer
from embeddings.base import Embedding

GloVeInfo = namedtuple('GloveInfo', ('filename', 'zipname', 'embedding_dim'))

class GloVeType(Enum):
    B6_D50 = GloVeInfo('glove.6B.50d.txt', 'glove.6B.zip', 50)
    B6_D100 = GloVeInfo('glove.6B.100d.txt', 'glove.6B.zip', 100)
    B6_D200 = GloVeInfo('glove.6B.200d.txt', 'glove.6B.zip', 200)
    B6_D300 = GloVeInfo('glove.6B.300d.txt', 'glove.6B.zip', 300)
    B42_D300 = GloVeInfo('glove.42B.300d.txt', 'glove.42B.300d.zip', 300)
    B840_D300 = GloVeInfo('glove.840B.300d.txt', 'glove.840B.300d.zip', 300)
    B27_Twitter_D25 = GloVeInfo('glove.twitter.27B.25d.txt', 'glove.twitter.27B.zip', 25)
    B27_Twitter_D50 = GloVeInfo('glove.twitter.27B.50d.txt', 'glove.twitter.27B.zip', 50)
    B27_Twitter_D100 = GloVeInfo('glove.twitter.27B.100d.txt', 'glove.twitter.27B.zip', 100)
    B27_Twitter_D200 = GloVeInfo('glove.twitter.27B.200d.txt', 'glove.twitter.27B.zip', 200)

class GloVe(Embedding):

    def __init__(self, config:Config, glove_type:GloVeType, max_sent_len=-1, no_cache:bool=False):
        self.config = config
        self.config.add_logger('glove_log')
        self.glove_type: GloVeType = glove_type
        self.weights_path = Path(config.weights.global_weights_dir) / 'glove'
        self.tokenizer = WordTokenizer(pad='padding', max_sent_len=max_sent_len)

        self.__vectors, self.__words, self.__word2idx = self.__load_glove__(no_cache)
        self.__idx2word = {i:w for w, i in self.__word2idx.items()}

    @property
    def embedding_dim(self) -> int:
        return self.glove_type.value.embedding_dim

    @property
    def tokens(self) -> List[str]:
        return self.__words

    def tokenize(self, text:str) -> List[str]:
        return self.tokenizer.tokenize(text, ex_punc=True)

    def index2token(self, index: int) -> str:
        if index not in self.__idx2word:
            return 'unknonw'
        else:
            return self.__idx2word[index]

    def token2index(self, token: str) -> int:
        if token not in self.__word2idx:
            return -1
        else:
            return self.__word2idx[token]

    def embed(self, input_text:str) -> np.ndarray:
        tokens = self.tokenize(input_text)
        indices = [self.token2index(token) for token in tokens]
        embed_vector = np.array([self.__vectors[idx] for idx in indices])
        return embed_vector

    def __load_glove__(self, no_cache:bool=False):
        '''load pretrained glove from http://nlp.stanford.edu

        Args:
            no_cache (bool): if True, download weights from the original source.

        Returns:
            vectors: np.ndarray (vocab_size, embedding_dim)
            words: list[str]
            word2idx: dict[word, index]
        '''
        weights_zip_path = self.weights_path / self.glove_type.value.zipname

        if not weights_zip_path.exists():
            self.config.log.glove_log.info('download glove weights from the Internet.')
            url = f'http://nlp.stanford.edu/data/{self.glove_type.value.zipname}'
            download(url, weights_zip_path)

        # cache path
        tokens, dim = self.glove_type.value.filename.split('.')[-3:-1]
        cache_dir = self.config.data.cache_path / 'glove' / f'glove.{tokens}'
        vector_cache = cache_dir / f'glove.{tokens}.{dim}_vectors.pickle'
        words_cache = cache_dir / f'glove.{tokens}.{dim}_words.pickle'
        word2idx_cache = cache_dir / f'glove.{tokens}.{dim}_word2idx.pickle'
        cache_dir.mkdir(parents=True, exist_ok=True)

        if no_cache == False and vector_cache.exists() and words_cache.exists() and word2idx_cache.exists():
            self.config.log.glove_log.info('restore weights from cache.')
            vectors = pickle.load(open(str(vector_cache), 'rb'))
            words = pickle.load(open(str(words_cache), 'rb'))
            word2idx = pickle.load(open(str(word2idx_cache), 'rb'))
        else:
            self.config.log.glove_log.info('construct weights from the weights file.')
            words = []
            idx = 0
            word2idx = {}
            vectors = []
            with zipfile.ZipFile(str(weights_zip_path)) as zip_f:
                with zip_f.open(self.glove_type.value.filename) as f:

                    # get file size
                    f_len = sum([1 for _ in f])
                    f.seek(0)

                    # load weights
                    for l in tqdm(f, desc='loading glove weights', total=f_len, leave=False):
                        line = [i.strip() for i in l.strip().split()]
                        word = line[0].decode('utf-8')
                        words.append(word)
                        word2idx[word] = idx
                        idx += 1         
                        vect = np.array(line[1:]).astype(np.float)
                        vectors.append(vect)
                
                # for unknown token
                vectors.append(np.mean(vectors, axis=0))
                words.append('unknonw')
                word2idx['unknonw'] = idx

                vectors = np.array(vectors)

            # cache weights
            pickle.dump(vectors, open(str(vector_cache), 'wb'))
            pickle.dump(words, open(str(words_cache), 'wb'))
            pickle.dump(word2idx, open(str(word2idx_cache), 'wb'))

        self.config.log.glove_log.info(f'Finished loading glove tokens: total={len(words)} words.')
        return np.array(vectors), words, word2idx

    def get_weights_matrix(self) -> np.ndarray:
        '''load pretrained glove weights matrix from
    
        Returns:
            weights matrix: np.ndarray (vocab_size, embedding_dim)
        '''
        glove = {w: self.__vectors[self.__word2idx[w]] for w in self.__words}
        embedding_dim = self.__vectors.shape[-1]

        # construct weights_matrix
        vocab = list(set(self.__words))
        matrix_len = len(vocab)
        weights_matrix = np.zeros((matrix_len, embedding_dim))

        # load weights
        for i, word in enumerate(tqdm(vocab, desc='loading glove weights', leave=False)):
            if word in glove:
                weights_matrix[i] = glove[word]
            else:
                weights_matrix[i] = np.zeros(embedding_dim)

        return weights_matrix
