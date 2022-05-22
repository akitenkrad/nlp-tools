import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from stats.stats import ConferenceText, Text
from utils.google_drive import GDriveObjects, download_from_google_drive
from utils.tokenizers import Tokenizer
from utils.utils import Lang


class Conference(ABC):
    @classmethod
    @abstractmethod
    def load(cls, preprocess_tokenizer: Optional[Tokenizer]) -> List[ConferenceText]:
        pass


class NeurIPS_2021(Conference):
    @classmethod
    def load(cls, preprocess_tokenizer: Optional[Tokenizer]) -> List[ConferenceText]:
        data_path = Path("data/conference/NeurIPS/neurips_2021.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:

            title = paper["title"]
            summary = paper["abstract"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(Text(title, language=preprocess_tokenizer.language))
                summary_tokens = preprocess_tokenizer.tokenize(Text(summary, language=preprocess_tokenizer.language))
                title = " ".join(token.surface for token in title_tokens)
                summary = " ".join(token.surface for token in summary_tokens)

            texts.append(
                ConferenceText(
                    title=title,
                    summary=summary,
                    keywords=[keyword.strip().lower() for keyword in paper["keywords"]],
                    pdf_url=paper["pdf_url"],
                    authors=paper["authors"],
                    language=Lang.ENGLISH,
                )
            )
        return texts


class ANLP_2022(Conference):
    @classmethod
    def load(self, preprocess_tokenizer: Optional[Tokenizer]) -> List[ConferenceText]:
        data_path = Path("data/conference/ANLP/ANLP-2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.ANLP_2022.value, str(data_path))

        data = json.load(open(data_path))
        keywords = data["categories"]
        papers = data["papers"]
        texts = []
        for paper in papers:

            title = paper["title"]
            summary = paper["abstract"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(Text(title, language=preprocess_tokenizer.languge))
                summary_tokens = preprocess_tokenizer.tokenize(Text(summary, language=preprocess_tokenizer.language))
                title = " ".join(token.surface for token in title_tokens)
                summary = " ".join(token.surface for token in summary_tokens)

            texts.append(
                ConferenceText(
                    title=title,
                    summary=summary,
                    keywords=[keywords[paper["id"].split("-")[0]].strip().lower()],
                    pdf_url="",
                    authors=[],
                    language=Lang.JAPANESE if paper["language"] == "japanese" else Lang.ENGLISH,
                )
            )
        return texts


class JSAI_2022(Conference):
    @classmethod
    def load(self, preprocess_tokenizer: Optional[Tokenizer]) -> List[ConferenceText]:
        data_path = Path("data/conference/JSAI/JSAI_2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.JSAI_2022.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:
            title = re.sub(r"^\[.+\]\s*", "", paper["title"])
            summary = paper["summary"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(Text(title, language=preprocess_tokenizer.language))
                summary_tokens = preprocess_tokenizer.tokenize(Text(summary, language=preprocess_tokenizer.language))
                title = " ".join(token.surface for token in title_tokens)
                summary = " ".join(token.surface for token in summary_tokens)

            if paper["language"] == "japanese":
                language = Lang.JAPANESE
            elif paper["language"] == "english":
                language = Lang.ENGLISH
            else:
                language = Lang.JAPANESE

            texts.append(
                ConferenceText(
                    title=title,
                    summary=summary,
                    keywords=[keyword.replace("Keywords:", "").strip().lower() for keyword in paper["keywords"]],
                    pdf_url="",
                    authors=paper["authors"],
                    language=language,
                )
            )
        return texts
