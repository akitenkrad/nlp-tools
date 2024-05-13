import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

import ipadic
import MeCab
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nlp_tools.tokenizers.base import JA_PUNCTUATIONS, JA_STOPWORDS, PAD, BaseTokenizer
from nlp_tools.utils.data import Token
from nlp_tools.utils.utils import Lang, isfloat, isint


class CharTokenizer(BaseTokenizer):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
        pass


class EnglishCharTokenizer(CharTokenizer):
    def __init__(
        self,
        pad: str = PAD,
        max_sent_len: int = -1,
        max_word_len: int = -1,
        remove_stopwords: bool = False,
        remove_punctuations: bool = True,
        remove_numbers: bool = False,
        stemming: bool = False,
        filter: Optional[Callable] = None,
    ):
        self.language: Lang = Lang.ENGLISH
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.max_word_len: int = max_word_len
        self.remove_stopwords: bool = remove_stopwords
        self.remove_punctuations: bool = remove_punctuations
        self.remove_numbers: bool = remove_numbers
        self.stemming: bool = stemming
        self.porter: PorterStemmer = PorterStemmer()
        self.filter: Optional[Callable] = filter

    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
        """tokenize a sentence

        Return:
            [[c_1_1, c_1_2, ...], [c_2_1, c_2_2, ...], ...]
        """
        # lowercase
        lower_text = text.lower()

        # tokenize
        words = [word for word in word_tokenize(lower_text)]

        # remove punctuation
        if self.remove_punctuations:
            words = [word for word in words if word not in list(string.punctuation)]

        # remove stopwords
        if self.remove_stopwords:
            stop_words = stopwords.words(self.language.value)
            words = [word for word in words if word not in stop_words]

        # remove numbers
        if self.remove_numbers:
            words = [word for word in words if (not isint(word[1])) and (not isfloat(word[1]))]

        # add tag
        words = nltk.pos_tag(words)

        # stemming
        if self.stemming:
            word_tokens = [Token(word, self.porter.stem(word), pos) for word, pos in words]
        else:
            word_tokens = [Token(word, "", pos) for word, pos in words]

        # filter
        if self.filter is not None:
            word_tokens = [token for token in word_tokens if self.filter(token)]

        words = [token.surface for token in word_tokens]
        chars = []
        for word in words:
            word = [(char, "", "") for char in list(word)]

            # pad words
            if self.max_word_len > 0:
                word = word + [(self.pad, "", "")] * self.max_word_len
                word = word[: self.max_word_len]

            chars.append(word)

        # pad sentence
        if self.max_sent_len > 0:
            chars = chars + [[(self.pad, "", "")] * self.max_word_len] * self.max_sent_len
            chars = chars[: self.max_sent_len]

        # Tuple -> Token
        return [[Token(char, "", "") for char in sent] for sent in chars]


class JapaneseCharTokenizer(CharTokenizer):
    def __init__(
        self,
        pad: str = "<pad>",
        max_sent_len: int = -1,
        max_word_len: int = -1,
        remove_stopwords: bool = False,
        remove_punctuations: bool = True,
        remove_numbers: bool = False,
        stemming: bool = False,
        filter: Optional[Callable] = None,
    ):
        self.language = Lang.JAPANESE
        self.pad = pad
        self.max_sent_len: int = max_sent_len
        self.max_word_len: int = max_word_len
        self.remove_stopwords: bool = remove_stopwords
        self.remove_punctuations: bool = remove_punctuations
        self.remove_numbers: bool = remove_numbers
        self.stemming: bool = stemming
        self.filter: Optional[Callable] = filter

    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
        """tokenize the input text at character granularity

        Args:
            text (str): input text
            mecab_user_dic (str): path to mecab user dictionary (user.dic).

        Returns:
            List[List[Token]]: list of tokens
        """
        # lowercase
        lower_text = text.lower()

        # add pos and base
        user_dic = f' -u {Path(kwargs["mecab_user_dic"]).absolute()}' if "mecab_user_dic" in kwargs else ""
        tagger = MeCab.Tagger(f"{ipadic.MECAB_ARGS}" + user_dic)
        result = tagger.parse(lower_text)
        word_tokens = []
        for line in result.split("\n"):
            if "\t" not in line:
                continue
            surface, _attrs = line.split("\t")
            attrs = _attrs.split(",")
            pos = attrs[0]
            base = attrs[6] if len(attrs) > 6 else surface

            if self.stemming:
                word = _attrs[6]
            else:
                word = surface
            word_tokens.append(Token(word, base, pos))

        # remove punctuations
        if self.remove_punctuations:
            word_tokens = [word for word in word_tokens if word.surface not in JA_PUNCTUATIONS]

        # remove stopwords
        if self.remove_stopwords:
            word_tokens = [word for word in word_tokens if word.surface not in JA_STOPWORDS]

        # remove numbers
        if self.remove_numbers:
            word_tokens = [word for word in word_tokens if (not isint(word.surface)) and (not isfloat(word.surface))]

        # apply filter
        if self.filter is not None:
            word_tokens = [word for word in word_tokens if self.filter(word)]

        words = [word.surface for word in word_tokens]
        chars = []
        for word in words:
            word = [(char, "", "") for char in list(word)]

            # pad words
            if self.max_word_len > 0:
                word = word + [(self.pad, "", "")] * self.max_word_len
                word = word[: self.max_word_len]

            chars.append(word)

        # pad sentence
        if self.max_sent_len > 0:
            chars = chars + [[(self.pad, "", "")] * self.max_word_len] * self.max_sent_len
            chars = chars[: self.max_sent_len]

        # Tuple -> Token
        return [[Token(char, "", "") for char in sent] for sent in chars]


class CharTokenizerFactory(object):
    """return CharTokenizer for each language"""

    @classmethod
    def get_tokenizer(
        cls,
        language=Lang.ENGLISH,
        pad="<pad>",
        max_sent_len=-1,
        max_word_len=-1,
        remove_stopwords=False,
        remove_punctuations=True,
        stemming=False,
        filter=None,
    ) -> CharTokenizer:
        if language == Lang.ENGLISH:
            return EnglishCharTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                max_word_len=max_word_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
                filter=filter,
            )
        elif language == Lang.JAPANESE:
            return JapaneseCharTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                max_word_len=max_word_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
                filter=filter,
            )
        else:
            raise NotImplementedError(f"Tokenizer for {language.value} is not implemented yet.")
