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
from transformers import AutoTokenizer

from nlp_tools.tokenizers.base import JA_PUNCTUATIONS, JA_STOPWORDS, PAD, BaseTokenizer
from nlp_tools.utils.data import Token
from nlp_tools.utils.utils import Lang, isfloat, isint


class WordTokenizer(BaseTokenizer):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
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

    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
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
        tokens: list[Union[Token, list[Token]]] = [Token(*word) for word in words]

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

    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
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
        tokens: list[Union[Token, list[Token]]] = [Token(*word) for word in words]

        # apply filter
        if self.filter is not None:
            tokens = [token for token in tokens if self.filter(token)]

        return tokens


class TFTokenizer(WordTokenizer):
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        self.tf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

    def tokenize(self, text: str, **kwargs) -> list[Union[Token, list[Token]]]:
        text_tokens = self.tf_tokenizer(text, **kwargs)
        return [Token(tok, "", "") for tok in text_tokens]


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
