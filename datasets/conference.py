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
    def load(cls, preprocess_tokenizer: Optional[Tokenizer] = None) -> List[ConferenceText]:
        pass


class NeurIPS_2021(Conference):
    @classmethod
    def load(cls, preprocess_tokenizer: Optional[Tokenizer] = None) -> List[ConferenceText]:
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
    def load(self, preprocess_tokenizer: Optional[Tokenizer] = None) -> List[ConferenceText]:
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
    def load(self, preprocess_tokenizer: Optional[Tokenizer] = None) -> List[ConferenceText]:
        data_path = Path("data/conference/JSAI/JSAI_2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.JSAI_2022.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:
            original_title = paper["title"]
            original_summary = paper["summary"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(Text(original_title, language=preprocess_tokenizer.language))
                summary_tokens = preprocess_tokenizer.tokenize(Text(original_summary, language=preprocess_tokenizer.language))
                preprocessed_title = " ".join(token.surface for token in title_tokens)
                preprocessed_summary = " ".join(token.surface for token in summary_tokens)
            else:
                preprocessed_title = original_title
                preprocessed_summary = preprocessed_summary

            if paper["language"] == "japanese":
                language = Lang.JAPANESE
            elif paper["language"] == "english":
                language = Lang.ENGLISH
            else:
                language = Lang.JAPANESE

            if preprocessed_summary == "n/a":
                continue

            texts.append(
                ConferenceText(
                    original_title=original_title,
                    preprocessed_title=preprocessed_title,
                    original_summary=original_summary,
                    preprocessed_summary=preprocessed_summary,
                    keywords=paper["keywords"],
                    pdf_url=paper["url"],
                    authors=paper["authors"],
                    language=language,
                    session_schedule=paper["schedule"],
                    session_title=paper["session"],
                )
            )
        return texts
