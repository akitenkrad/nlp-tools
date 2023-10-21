from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from nlp_tools.utils.utils import Lang


@dataclass
class Token(object):
    """Represent a token"""

    surface: str
    base: str
    pos_tag: str

    def __hash__(self) -> int:
        return hash(self.surface + self.base + self.pos_tag)

    def to_dict(self) -> Dict[str, str]:
        return {"surface": self.surface, "base": self.base, "pos_tag": self.pos_tag}


@dataclass
class Sentence(object):
    """Represent a sentence composed of a list of Tokens"""

    tokens: List[Token]
    language: Lang = Lang.ENGLISH

    def __hash__(self) -> int:
        return hash("".join([str(hash(token)) for token in self.tokens]))

    def to_dict(self) -> Dict:
        return {"tokens": [token.to_dict() for token in self.tokens]}

    @property
    def text(self) -> str:
        return " ".join([token.surface for token in self.tokens])


class ConferenceText(Sentence):
    def __init__(
        self,
        index: int,
        title: str,
        summary: str = "",
        preprocessed_title: str = "",
        preprocessed_summary: str = "",
        keywords: List[str] = [],
        pdf_url: str = "",
        authors: List[str] = [],
        language: Lang = Lang.ENGLISH,
        published_at: Optional[datetime] = None,
        **kwargs,
    ):
        self.index: int = index
        self.title: str = title
        self.summary: str = summary
        self.preprocessed_title: str = preprocessed_title
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
        return f"<ConferenceText {self.index:05d} {self.title[:15]}... (in {self.language})>"

    def __repr__(self):
        return self.__str__()

    @property
    def text(self) -> str:
        return self.__text

    def get_attr(self, key: str) -> str:
        for attr in self.attrs:
            if attr["name"] == key:
                return attr["value"]
        raise KeyError(f"Unexpected key: {key}")

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "index": self.index,
            "text": self.__text,
            "title": self.title,
            "summary": self.summary,
            "preprocessed_title": self.preprocessed_title,
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


class QAText(Sentence):
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
