import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import ipadic
import MeCab
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer

from nlp_tools.utils.data import Token
from nlp_tools.utils.utils import Lang, isfloat, isint

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

PAD = "<PAD>"

JA_PUNCTUATIONS = [c for c in string.punctuation]
JA_PUNCTUATIONS += ["．", "。", "，", "、"]
JA_PUNCTUATIONS += ["（", "）", "「", "」", "【", "】", "『", "』", "［", "］", "｛", "｝", "〈", "〉", "〔", "〕", "《", "》", "＜", "＞"]
JA_PUNCTUATIONS += ["！", "＠", "？", "：", "；", "”", "’", "ー", "〜", "-", "~", "−", "・", "＄", "％", "＾", "＆", "／", "＝", "＼"]

JA_STOPWORDS = stopwords.words(Lang.ENGLISH.value)
JA_STOPWORDS += [
    ss.strip() for ss in open(Path(__file__).parent / "resources/texts/JA_STOPWORDS_SLOTHLIB.txt") if ss.strip() != ""
]


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[Token]:
        pass


class WordTokenizer(Tokenizer):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[Token]:
        pass


class CharTokenizer(Tokenizer):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[List[Token]]:
        pass


class EnglishWordTokenizer(WordTokenizer):
    def __init__(
        self,
        pad: str = PAD,
        max_sent_len: int = -1,
        remove_stopwords: bool = False,
        remove_punctuations: bool = True,
        remove_numbers: bool = False,
        stemming: bool = False,
        add_tag: bool = False,
        filter: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            pad (str): PAD token
            max_sent_len (int): max sentence length (> 0)
            remove_stopwords (bool): if True, remove stopwords
            remove_punctuations (bool): if True, remove punctuations
            stemming (bool): if True, stem words
            add_tag (bool): if True, add pos tag
            filter (function): filter tokens
                               ex. lambda tk: tk.pos_tag.startswith("NN") # take only nouns
        """
        self.language = Lang.ENGLISH
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.remove_punctuations: bool = remove_punctuations
        self.remove_stopwords: bool = remove_stopwords
        self.remove_numbers: bool = remove_numbers
        self.add_tag: bool = add_tag
        self.stemming: bool = stemming
        self.porter: PorterStemmer = PorterStemmer()
        self.filter: Optional[Callable] = filter

    def tokenize(self, text: str, **kwargs) -> List[Token]:
        """tokenize a sentence into a list of words"""
        disable_max_len = kwargs.get("disable_max_len", False)

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
            words = [word for word in words if (not isint(word)) and (not isfloat(word))]

        # add tag
        if self.add_tag or self.filter is not None:
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

        # apply filter
        if self.filter is not None:
            tokens = [token for token in tokens if self.filter(token)]

        return tokens


class JapaneseWordTokenizer(WordTokenizer):
    def __init__(
        self,
        pad: str = PAD,
        max_sent_len: int = -1,
        remove_stopwords: bool = False,
        remove_punctuations: bool = True,
        remove_numbers: bool = False,
        filter: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            pad (str): PAD token
            max_sent_len (int): max sentence length (> 0)
            remove_stopwords (bool): if True, remove stopwords
            remove_punctuations (bool): if True, remove punctuations
            filter (function): filter tokens
                               ex. lambda tk: tk.pos_tag.startswith("NN") # take only nouns
        """
        self.language: Lang = Lang.JAPANESE
        self.pad: str = pad
        self.max_sent_len: int = max_sent_len
        self.remove_punctuations: bool = remove_punctuations
        self.remove_stopwords: bool = remove_stopwords
        self.remove_numbers: bool = remove_numbers
        self.filter: Optional[Callable] = filter

    def tokenize(self, text: str, **kwargs) -> List[Token]:
        """tokenize a text into a list of words

        Args:
            text (str): input text
            mecab_user_dic (str): path to mecab user dictionary (user.dic).

        Returns:
            List[Token]: list of tokens
        """
        disable_max_len = kwargs.get("disable_max_len", False)

        # lowercase
        lower_text = text.lower()

        # add pos and base
        user_dic = f' -u {Path(kwargs["mecab_user_dic"]).absolute()}' if "mecab_user_dic" in kwargs else ""
        tagger = MeCab.Tagger(f"{ipadic.MECAB_ARGS}" + user_dic)
        result = tagger.parse(lower_text)
        words = []
        for line in result.split("\n"):
            if "\t" not in line:
                continue
            surface, _attrs = line.split("\t")
            attrs = _attrs.split(",")
            pos = attrs[0]
            base = attrs[6] if len(attrs) > 6 else surface

            words.append((surface, base, pos))

        # remove punctuations
        if self.remove_punctuations:
            words = [word for word in words if word[0] not in JA_PUNCTUATIONS]

        # remove stopwords
        if self.remove_stopwords:
            words = [word for word in words if word[1] not in JA_STOPWORDS]

        # remove numbers
        if self.remove_numbers:
            words = [word for word in words if (not isint(word[1])) and (not isfloat(word[1]))]

        # pad sentence
        if self.max_sent_len > 0 and not disable_max_len:
            pad = (self.pad, "", "")
            words = words + [pad] * self.max_sent_len
            words = words[: self.max_sent_len]

        # Tuple -> Token
        tokens = [Token(*word) for word in words]

        # apply filter
        if self.filter is not None:
            tokens = [token for token in tokens if self.filter(token)]

        return tokens


class TFTokenizer(WordTokenizer):
    def __init__(self, pretrained_model_name_or_path: str):
        self.__tf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, text: str, **kwargs) -> List[Token]:
        text_tokens = self.__tf_tokenizer.tokenize(text)
        return [Token(tok, "", "") for tok in text_tokens]


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

    def tokenize(self, text: str, **kwargs) -> List[List[Token]]:
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
        tokens = [[Token(char, "", "") for char in sent] for sent in chars]

        return tokens


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

    def tokenize(self, text: str, **kwargs) -> List[List[Token]]:
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
        filter=None,
        pretrained_model_name_or_path="",
        **kwargs,
    ) -> WordTokenizer:
        if pretrained_model_name_or_path != "":
            return TFTokenizer(pretrained_model_name_or_path)

        elif language == Lang.ENGLISH:
            return EnglishWordTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                stemming=stemming,
                add_tag=add_tag,
                filter=filter,
                **kwargs,
            )

        elif language == Lang.JAPANESE:
            return JapaneseWordTokenizer(
                pad=pad,
                max_sent_len=max_sent_len,
                remove_stopwords=remove_stopwords,
                remove_punctuations=remove_punctuations,
                filter=filter,
                **kwargs,
            )

        else:
            raise NotImplementedError(f"Tokenizer for {language.value} is not implemented yet.")


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
