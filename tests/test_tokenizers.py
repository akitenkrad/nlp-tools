from typing import Callable, List

import pytest

from utils.data import Token
from utils.tokenizers import PAD, WordTokenizerFactory
from utils.utils import Lang


@pytest.mark.parametrize(
    ("language", "text", "remove_punctuations", "remove_stopwords", "max_sent_len", "filter", "expected"),
    [
        (
            Lang.ENGLISH,
            "The price of greatness is responsibility.",
            True,
            True,
            5,
            None,
            [
                Token("price", "price", "NN"),
                Token("greatness", "great", "NN"),
                Token("responsibility", "respons", "NN"),
                Token(PAD, "", ""),
                Token(PAD, "", ""),
            ],
        ),
        (
            Lang.ENGLISH,
            "The price of greatness is responsibility.",
            False,
            False,
            -1,
            lambda tok: tok.pos_tag.startswith("NN"),
            [
                Token("price", "price", "NN"),
                Token("greatness", "great", "NN"),
                Token("responsibility", "respons", "NN"),
            ],
        ),
        (
            Lang.JAPANESE,
            "もっと自信を持ってよね！",
            True,
            False,
            5,
            None,
            [
                Token("もっと", "もっと", "副詞"),
                Token("自信", "自信", "名詞"),
                Token("を", "を", "助詞"),
                Token("持っ", "持つ", "動詞"),
                Token("て", "て", "助詞"),
            ],
        ),
        (
            Lang.JAPANESE,
            "もっと自信を持ってよね！",
            False,
            False,
            -1,
            lambda tk: tk.pos_tag == "名詞",
            [Token("自信", "自信", "名詞")],
        ),
    ],
)
def test_english_word_tokenizer(
    language: Lang,
    text: str,
    remove_punctuations: bool,
    remove_stopwords: bool,
    max_sent_len: int,
    filter: Callable,
    expected: List[Token],
):
    tokenizer = WordTokenizerFactory.get_tokenizer(
        language,
        pad=PAD,
        remove_punctuations=remove_punctuations,
        remove_stopwords=remove_stopwords,
        stemming=True,
        add_tag=True,
        max_sent_len=max_sent_len,
        filter=filter,
    )
    responses = tokenizer.tokenize(text)
    for res, ans in zip(responses, expected):
        assert res == ans
