import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from stats.stats import ConferenceText
from utils.google_drive import GDriveObjects, download_from_google_drive
from utils.utils import Lang


class Conference(ABC):
    @classmethod
    @abstractmethod
    def load(cls) -> List[ConferenceText]:
        pass


class NeurIPS_2021(Conference):
    @classmethod
    def load(self) -> List[ConferenceText]:
        data_path = Path("data/conference/NeurIPS/neurips_2021.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:
            texts.append(
                ConferenceText(
                    title=paper["title"],
                    summary=paper["abstract"],
                    keywords=[keyword.strip().lower() for keyword in paper["keywords"]],
                    pdf_url=paper["pdf_url"],
                    authors=paper["authors"],
                    language=Lang.ENGLISH,
                )
            )
        return texts


class ANLP_2022(Conference):
    @classmethod
    def load(self) -> List[ConferenceText]:
        data_path = Path("data/conference/ANLP/ANLP-2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.ANLP_2022.value, str(data_path))

        data = json.load(open(data_path))
        keywords = data["categories"]
        papers = data["papers"]
        texts = []
        for paper in papers:
            texts.append(
                ConferenceText(
                    title=paper["title"],
                    summary=paper["abstract"],
                    keywords=[keywords[paper["id"].split("-")[0]].strip().lower()],
                    pdf_url="",
                    authors=[],
                    language=Lang.JAPANESE if paper["language"] == "japanese" else Lang.ENGLISH,
                )
            )
        return texts


class JSAI_2022(Conference):
    @classmethod
    def load(self) -> List[ConferenceText]:
        data_path = Path("data/conference/JSAI/JSAI_2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.JSAI_2022.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:
            texts.append(
                ConferenceText(
                    title=re.sub(r"^\[.+\]\s*", "", paper["title"]),
                    summary=paper["summary"],
                    keywords=[keyword.replace("Keywords:", "").strip().lower() for keyword in paper["keywords"]],
                    pdf_url="",
                    authors=paper["authors"],
                    language=Lang.JAPANESE,
                )
            )
        return texts
