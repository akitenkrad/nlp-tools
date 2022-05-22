from typing import List

import numpy as np

from utils.utils import Lang


class Token(object):
    def __init__(self, surface: str, base: str, pos_tag: str):
        self.surface = surface
        self.base = base
        self.pos_tag = pos_tag

    def __str__(self) -> str:
        return f"<Token {self.surface} {self.base} {self.pos_tag}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Token):
            return (self.surface == __o.surface) and (self.base == __o.base) and (self.pos_tag == __o.pos_tag)
        else:
            return False


class Text(object):
    def __init__(self, text: str, language=Lang.ENGLISH):
        self.__text = text
        self.language = language

    def __str__(self) -> str:
        return f"<Text {self.text[:10]} ...>"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def text(self) -> str:
        return self.__text


class ConferenceText(Text):
    def __init__(
        self,
        title: str,
        summary: str,
        keywords: List[str],
        pdf_url: str,
        authors: List[str],
        language: Lang,
        **kwargs,
    ):
        self.title: str = title
        self.summary: str = summary
        self.keywords: List[str] = keywords
        self.pdf_url: str = pdf_url
        self.authors: List[str] = authors
        self.language = language
        self.topic = -99
        self.topic_prob = np.array([])
        for name, value in kwargs.items():
            if not hasattr(self, name):
                setattr(self, name, value)

        self.__text = f"{self.title} {self.summary}"

    def __str__(self):
        return f"<ConferenceText {self.title[:15]}... {self.language}>"

    def __repr__(self):
        return self.__str__()

    @property
    def text(self) -> str:
        return self.__text


class QAText(Text):
    def __init__(self, question: str, passages: List[str], answer: str, language=Lang.ENGLISH):
        self.question: str = question
        self.passages: List[str] = passages
        self.answer: str = answer
        self.language: Lang = language

        self.__text = f"{self.question}"

    def __str__(self):
        return f"<QAText {self.question[:15]}... {self.language}>"

    def __repr__(self):
        return self.__str__()

    @property
    def text(self) -> str:
        return self.__text
