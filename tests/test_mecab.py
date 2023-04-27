import os
import shutil
from pathlib import Path

import pytest

from utils.mecab import MeCabItem, MeCabPos, MeCabUtil
from utils.tokenizers import WordTokenizerFactory
from utils.utils import Lang


def test_mecab_add_words_1():
    # 1. before adding words
    word_tokenizer = WordTokenizerFactory.get_tokenizer(language=Lang.JAPANESE, add_tag=True)

    tokens = word_tokenizer.tokenize("メロンパン")
    assert tokens[0].surface == "メロン"

    tokens = word_tokenizer.tokenize("いちごパン")
    assert tokens[0].surface == "いちご"

    tokens = word_tokenizer.tokenize("コーヒーパン")
    assert tokens[0].surface == "コーヒー"

    # after adding words
    user_dic = Path("/usr/local/lib/mecab/user.dic")
    if MeCabUtil.TEMP_DICT_CSV.exists():
        os.remove(MeCabUtil.TEMP_DICT_CSV)
    if user_dic.exists():
        os.remove(user_dic)

    items = [
        MeCabItem(word="メロンパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
        MeCabItem(word="いちごパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
        MeCabItem(word="コーヒーパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
    ]

    MeCabUtil.add_noun_to_dict(items, user_dic=user_dic)
    assert Path(user_dic).exists()

    tokens = word_tokenizer.tokenize("メロンパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "メロンパン"

    tokens = word_tokenizer.tokenize("いちごパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "いちごパン"

    tokens = word_tokenizer.tokenize("コーヒーパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "コーヒーパン"


def test_mecab_add_words_2():
    # 1. before adding words
    word_tokenizer = WordTokenizerFactory.get_tokenizer(language=Lang.JAPANESE, add_tag=True)

    tokens = word_tokenizer.tokenize("メロンパン")
    assert tokens[0].surface == "メロン"

    tokens = word_tokenizer.tokenize("いちごパン")
    assert tokens[0].surface == "いちご"

    tokens = word_tokenizer.tokenize("コーヒーパン")
    assert tokens[0].surface == "コーヒー"

    # after adding words
    user_dic = Path("/usr/local/lib/mecab/user.dic")
    if MeCabUtil.TEMP_DICT_CSV.exists():
        os.remove(MeCabUtil.TEMP_DICT_CSV)
    if user_dic.exists():
        os.remove(user_dic)

    items = [
        MeCabItem(word="メロンパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
        MeCabItem(word="いちごパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
        MeCabItem(word="コーヒーパン", tag=MeCabPos.NOUN, pos1=MeCabPos.NOUN_GENERAL),
    ]

    # 1st time
    MeCabUtil.add_noun_to_dict(items, user_dic=user_dic)
    assert Path(user_dic).exists()
    # 2nd time
    MeCabUtil.add_noun_to_dict(items, user_dic=user_dic)
    assert Path(user_dic).exists()
    # 3rd time
    MeCabUtil.add_noun_to_dict(items, user_dic=user_dic)
    assert Path(user_dic).exists()

    tokens = word_tokenizer.tokenize("メロンパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "メロンパン"

    tokens = word_tokenizer.tokenize("いちごパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "いちごパン"

    tokens = word_tokenizer.tokenize("コーヒーパン", mecab_user_dic=user_dic)
    assert tokens[0].surface == "コーヒーパン"
