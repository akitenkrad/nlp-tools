from typing import List

import pytest

from utils.data import Text, Token
from utils.tokenizers import PAD, WordTokenizerFactory
from utils.utils import Lang


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        (
            Lang.ENGLISH,
            Text("The price of greatness is responsibility."),
            [
                Token("price", "price", "NN"),
                Token("greatness", "great", "NN"),
                Token("responsibility", "respons", "NN"),
                Token(PAD, "", ""),
                Token(PAD, "", ""),
            ],
        ),
        (
            Lang.JAPANESE,
            Text("もっと自信を持ってよね！"),
            [Token("もっと", "もっと", "副詞"), Token("自信", "自信", "名詞"), Token("を", "を", "助詞"), Token("持っ", "持つ", "動詞"), Token("て", "て", "助詞")],
        ),
    ],
)
def test_english_word_tokenizer(language: Lang, text: Text, expected: List[Token]):
    tokenizer = WordTokenizerFactory.get_tokenizer(
        language, pad=PAD, remove_punctuations=True, remove_stopwords=True, stemming=True, add_tag=True, max_sent_len=5
    )
    responses = tokenizer.tokenize(text)
    for res, ans in zip(responses, expected):
        assert res == ans
