from datetime import datetime
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return {"surface": self.surface, "base": self.base, "pos_tag": self.pos_tag}


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

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.__text, "language": self.language.value}


class ConferenceText(Text):
    def __init__(
        self,
        original_title: str,
        preprocessed_title: str = "",
        original_summary: str = "",
        preprocessed_summary: str = "",
        keywords: List[str] = [],
        pdf_url: str = "",
        authors: List[str] = [],
        language: Lang = Lang.ENGLISH,
        published_at: Optional[datetime] = None,
        **kwargs,
    ):
        self.original_title: str = original_title
        self.preprocessed_title: str = preprocessed_title
        self.original_summary: str = original_summary
        self.preprocessed_summary: str = preprocessed_summary
        self.keywords: List[str] = keywords
        self.pdf_url: str = pdf_url
        self.authors: List[str] = authors
        self.language: Lang = language
        self.published_at: datetime = published_at if published_at is not None else datetime(1800, 1, 1)
        self.topic = -99
        self.topic_prob = np.array([])

        self.attrs = []
        for name, value in kwargs.items():
            if not hasattr(self, name):
                setattr(self, name, value)
                self.attrs.append({"name": name, "value": value})

        self.__text = f"{self.preprocessed_title} {self.preprocessed_summary}"

    def __str__(self):
        return f"<ConferenceText {self.original_title[:15]}... (in {self.language})>"

    def __repr__(self):
        return self.__str__()

    @property
    def text(self) -> str:
        return self.__text

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "text": self.__text,
            "original_title": self.original_title,
            "preprocessed_title": self.preprocessed_title,
            "original_summary": self.original_summary,
            "preprocessed_summary": self.preprocessed_summary,
            "keywords": self.keywords,
            "pdf_url": self.pdf_url,
            "authors": self.authors,
            "language": self.language,
            "topic": self.topic,
            "topic_prob": self.topic_prob,
        }
        for attr in self.attrs:
            res[attr["name"]] = attr["value"]
        return res


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
