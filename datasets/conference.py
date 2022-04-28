import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from stats.stats import Text
from utils.google_drive import GDriveObjects, download_from_google_drive
from utils.utils import Lang


class Conference(ABC):
    @classmethod
    @abstractmethod
    def load(cls) -> List[Text]:
        pass


class NeurIPS_2021(Conference):
    @classmethod
    def load(self) -> List[Text]:
        data_path = Path("data/conference/NeurIPS/neurips_2021.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(data_path))

        data = json.load(open(data_path))
        texts = []
        for item in data:
            texts.append(
                Text(
                    title=item["title"],
                    summary=item["abstract"],
                    keywords=item["keywords"],
                    pdf_url=item["pdf_url"],
                    authors=item["authors"],
                    language=Lang.ENGLISH,
                )
            )
        return texts


class ANLP_2022(Conference):
    @classmethod
    def load(self) -> List[Text]:
        data_path = Path("data/conference/ANLP/ANLP-2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.ANLP_2022.value, str(data_path))

        data = json.load(open(data_path))
        keywords = data["categories"]
        papers = data["papers"]
        texts = []
        for paper in papers:
            keyword = keywords[paper["id"].split("-")[0]]
            title = paper["title"]
            summary = paper["abstract"]
            lang = Lang.JAPANESE if paper["language"] == "japanese" else Lang.ENGLISH
            texts.append(
                Text(
                    title=title,
                    summary=summary,
                    keywords=[keyword],
                    pdf_url="",
                    authors=[],
                    language=lang,
                )
            )
        return texts
