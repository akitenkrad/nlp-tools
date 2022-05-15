import string
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import MeCab
import nltk
import unidic
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from utils.data import Text, Token
from utils.utils import Lang

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

PAD = "<PAD>"

JA_PUNCTUATIONS = [c for c in string.punctuation]
JA_PUNCTUATIONS += ["．", "。", "，", "、"]
JA_PUNCTUATIONS += ["（", "）", "「", "」", "【", "】", "『", "』", "［", "］", "｛", "｝", "〈", "〉", "〔", "〕", "《", "》", "＜", "＞"]
JA_PUNCTUATIONS += ["！", "＠", "？", "：", "；", "”", "’", "ー", "〜", "-", "~", "−", "・", "＄", "％", "＾", "＆", "／", "＝", "＼"]


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: Text, **kwargs) -> List[Union[Token, List[Token]]]:
        pass


class EnglishWordTokenizer(Tokenizer):
    def __init__(self, pad=PAD, max_sent_len=-1, remove_stopwords=False, remove_punctuations=True, stemming=False, add_tag=False):
        self.language = Lang.ENGLISH
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.remove_punctuations: bool = remove_punctuations
        self.remove_stopwords: bool = remove_stopwords
        self.add_tag: bool = add_tag
        self.stemming: bool = stemming
        self.porter: PorterStemmer = PorterStemmer()

    def tokenize(self, text: Text, **kwargs) -> List[Token]:
        """tokenize a sentence into a list of words"""
        disable_max_len = kwargs.get("disable_max_len", False)

        # lowercase
        lower_text = text.text.lower()

        # tokenize
        words = [word for word in word_tokenize(lower_text)]

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
        else:
            words = [(word, "") for word in words]

        # stemming
        if self.stemming:
            words = [(word, self.porter.stem(word), pos) for word, pos in words]
        else:
            words = [(word, "", pos) for word, pos in words]

        # pad sentence
        if self.max_sent_len > 0 and not disable_max_len:
            pad = (self.pad, "", "")
            words = words + [pad] * self.max_sent_len
            words = words[: self.max_sent_len]

        # Tuple -> Token
        tokens = [Token(*word) for word in words]

        return tokens


class JapaneseWordTokenizer(Tokenizer):
    def __init__(self, pad=PAD, max_sent_len=-1, remove_stopwords=False, remove_punctuations=True):
        self.language = Lang.JAPANESE
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.remove_punctuations: bool = remove_punctuations
        self.remove_stopwords: bool = remove_stopwords

    def tokenize(self, text: Text, **kwargs):
        """tokenize a text into a list of words"""
        disable_max_len = kwargs.get("disable_max_len", False)

        # lowercase
        lower_text = text.text.lower()

        # add pos and base
        tagger = MeCab.Tagger(f"-d {unidic.DICDIR}")
        result = tagger.parse(lower_text)
        words = []
        for line in result.split("\n"):
            if "\t" not in line:
                continue
            surface, _attrs = line.split("\t")
            attrs = _attrs.split(",")
            pos = attrs[0]
            base = attrs[7]

            words.append((surface, base, pos))

        # remove stopwords
        # TODO how to determin whether a word is a stopword or not

        # remove punctuations
        if self.remove_punctuations:
            words = [word for word in words if word[0] not in JA_PUNCTUATIONS]

        # pad sentence
        if self.max_sent_len > 0 and not disable_max_len:
            pad = (self.pad, "", "")
            words = words + [pad] * self.max_sent_len
            words = words[: self.max_sent_len]

        # Tuple -> Token
        tokens = [Token(*word) for word in words]

        return tokens


class EnglishCharTokenizer(Tokenizer):
    def __init__(self, pad=PAD, max_sent_len=-1, max_word_len=-1, remove_stopwords=False, remove_punctuations=True, stemming=False):
        self.language: Lang = Lang.ENGLISH
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.max_word_len: int = max_word_len
        self.remove_stopwords: bool = remove_stopwords
        self.remove_punctuations: bool = remove_punctuations
        self.stemming: bool = stemming
        self.porter: PorterStemmer = PorterStemmer()

    def tokenize(self, text: Text, **kwargs) -> List[List[Token]]:
        """tokenize a sentence

        Return:
            [[c_1_1, c_1_2, ...], [c_2_1, c_2_2, ...], ...]
        """
        # lowercase
        lower_text = text.text.lower()

        # tokenize
        words = [word for word in word_tokenize(lower_text)]

        # remove punctuation
        if self.remove_punctuations:
            words = [word for word in words if word not in list(string.punctuation)]

        # remove stopwords
        if self.remove_stopwords:
            stop_words = stopwords.words(self.language.value)
            words = [word for word in words if word not in stop_words]

        # stemming
        if self.stemming:
            words = [self.porter.stem(word) for word in words]

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
        tokens = [[Token(char, "", "") for char in sent] for sent in chars]

        return tokens


class JapaneseCharTokenizer(Tokenizer):
    def __init__(self, pad="<pad>", max_sent_len=-1, max_word_len=-1, remove_stopwords=False, remove_punctuations=True, stemming=False):
        self.language = Lang.JAPANESE
        self.pad = pad
        self.max_sent_len: int = max_sent_len
        self.max_word_len: int = max_word_len
        self.remove_stopwords: bool = remove_stopwords
        self.remove_punctuations: bool = remove_punctuations
        self.stemming: bool = stemming

    def tokenize(self, text: Text, **kwargs) -> List[List[Token]]:
        # lowercase
        lower_text = text.text.lower()

        # add pos and base
        tagger = MeCab.Tagger(f"-d {unidic.DICDIR}")
        result = tagger.parse(lower_text)
        words = []
        for line in result.split("\n"):
            if "\t" not in line:
                continue
            surface, _attrs = line.split("\t")

            if self.stemming:
                word = _attrs[7]
            else:
                word = surface
            words.append(word)

        # remove stopwords
        # TODO how to determin whether a word is a stopword or not

        # remove punctuations
        if self.remove_punctuations:
            words = [word for word in words if word not in JA_PUNCTUATIONS]

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
        tokens = [[Token(char, "", "") for char in sent] for sent in chars]

        return tokens


class WordTokenizerFactory(object):
    """return WordTokenizer for each language"""

    @classmethod
    def get_tokenizer(
        cls,
        language=Lang.ENGLISH,
        pad="<pad>",
        max_sent_len=-1,
        remove_stopwords=False,
        remove_punctuations=True,
        stemming=False,
        add_tag=False,
    ) -> Tokenizer:
        if language == Lang.ENGLISH:
            return EnglishWordTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
                add_tag=add_tag,
            )
        elif language == Lang.JAPANESE:
            return JapaneseWordTokenizer(
                pad=pad, max_sent_len=max_sent_len, remove_stopwords=remove_stopwords, remove_punctuations=remove_punctuations
            )
        else:
            raise NotImplementedError(f"Tokenizer for {language.value} is not implemented yet.")


class CharTokenizerFactory(object):
    """return CharTokenizer for each language"""

    @classmethod
    def get_tokenizer(
        cls, language=Lang.ENGLISH, pad="<pad>", max_sent_len=-1, max_word_len=-1, remove_stopwords=False, remove_punctuations=True, stemming=False
    ) -> Tokenizer:
        if language == Lang.ENGLISH:
            return EnglishCharTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                max_word_len=max_word_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
            )
        elif language == Lang.JAPANESE:
            return JapaneseCharTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                max_word_len=max_word_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
            )
        else:
            raise NotImplementedError(f"Tokenizer for {language.value} is not implemented yet.")
