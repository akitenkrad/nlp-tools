import string
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from utils.utils import Lang

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass


class WordTokenizer(Tokenizer):
    def __init__(
        self,
        pad="<pad>",
        max_sent_len=-1,
        language=Lang.ENGLISH,
        remove_stopwords=False,
        remove_punctuations=True,
        stemming=False,
        add_tag=False,
    ):
        self.PAD: str = pad
        self.max_sent_len: int = max_sent_len
        self.language: Lang = language
        self.remove_punctuations: bool = remove_punctuations
        self.remove_stopwords: bool = remove_stopwords
        self.add_tag: bool = add_tag
        self.porter: PorterStemmer = PorterStemmer() if stemming else None

    def tokenize(
        self, text: str, disable_max_len=False
    ) -> List[Union[str, Tuple[str, str]]]:
        """tokenize a sentence

        Args:
            ex_punc (bool): if True, remove punctuations
            disable_max_len (bool): if True, ignore the "max_sent_len" setting

        Return:
            [t_1, t_2, t_3, ...]
        """
        # lowercase
        text = text.lower()

        # tokenize
        words = [word for word in word_tokenize(text)]

        # remove punctuation
        if self.remove_punctuations:
            words = [word for word in words if word not in list(string.punctuation)]

        # remove stopwords
        if self.remove_stopwords:
            stop_words = stopwords.words(self.language.value)
            words = [word for word in words if word not in stop_words]

        # add tag
        if self.add_tag:
            words = nltk.pos_tag(words)

        # stemming
        if self.porter is not None:
            if self.add_tag:
                words = [(self.porter.stem(word[0]), word[1]) for word in words]
            else:
                words = [self.porter.stem(word) for word in words]

        # pad sentence
        if self.max_sent_len > 0 and not disable_max_len:
            pad = (self.PAD, "") if self.add_tag else self.PAD
            words = words + [pad] * self.max_sent_len
            words = words[: self.max_sent_len]

        return words


class CharTokenizer(Tokenizer):
    def __init__(
        self,
        pad="<pad>",
        max_sent_len=-1,
        max_word_len=-1,
        language=Lang.ENGLISH,
        remove_stopwords=False,
        remove_punctuations=True,
        stemming=False,
    ):
        self.PAD: str = pad
        self.max_sent_len: int = max_sent_len
        self.max_word_len: int = max_word_len
        self.language: Lang = language
        self.remove_stopwords: bool = remove_stopwords
        self.remove_punctuations: bool = remove_punctuations
        self.porter: PorterStemmer = PorterStemmer() if stemming else None

    def tokenize(self, text: str) -> List[List[str]]:
        """tokenize a sentence

        Return:
            [[c_1_1, c_1_2, ...], [c_2_1, c_2_2, ...], ...]
        """
        # lowercase
        text = text.lower()

        # remove punctuation
        if self.remove_punctuations:
            words = [
                word
                for word in word_tokenize(text)
                if word not in list(string.punctuation)
            ]

        words = [word for word in word_tokenize(text)]

        # remove stopwords
        if self.remove_stopwords:
            stop_words = stopwords.words(self.language.value)
            words = [word for word in words if word not in stop_words]

        # stemming
        if self.porter is not None:
            words = [self.porter.stem(word) for word in words]

        chars = []
        for word in words:
            word = list(word)

            # pad words
            if self.max_word_len > 0:
                word = word + [self.PAD] * self.max_word_len
                word = word[: self.max_word_len]

            chars.append(word)

        # pad sentence
        if self.max_sent_len > 0:
            chars = chars + [[self.PAD] * self.max_word_len] * self.max_sent_len
            chars = chars[: self.max_sent_len]

        return chars
