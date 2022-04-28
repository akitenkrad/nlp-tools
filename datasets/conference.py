import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from stats.stats import Text
from utils.google_drive import GDriveObjects, download_from_google_drive


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
                    item["title"],
                    item["abstract"],
                    item["keywords"],
                    item["pdf_url"],
                    item["authors"],
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
            texts.append(Text(title, summary, [keyword], "", []))
        return texts
