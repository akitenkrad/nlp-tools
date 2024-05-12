import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from dateutil.parser import parse as parse_date
from ml_tools.utils.utils import Config

from nlp_tools.utils.data import ConferenceText
from nlp_tools.utils.google_drive import GDriveObjects, download_from_google_drive
from nlp_tools.utils.tokenizers import WordTokenizer
from nlp_tools.utils.utils import Lang


class Conference(ABC):
    @classmethod
    @abstractmethod
    def load(cls, preprocess_tokenizer: Optional[WordTokenizer] = None) -> List[ConferenceText]:
        pass


class NeurIPS_2021(Conference):
    @classmethod
    def load(cls, preprocess_tokenizer: Optional[WordTokenizer] = None) -> List[ConferenceText]:
        data_path = Path("data/conference/NeurIPS/neurips_2021.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for index, paper in enumerate(papers):
            title = paper["title"]
            abstract = paper["abstract"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(title)
                abstract_tokens = preprocess_tokenizer.tokenize(abstract)
                preprocessed_title = " ".join(token.surface for token in title_tokens)
                preprocessed_abstract = " ".join(token.surface for token in abstract_tokens)
            else:
                preprocessed_title = ""
                preprocessed_abstract = ""

            texts.append(
                ConferenceText(
                    index=index,
                    title=title,
                    abstract=abstract,
                    preprocessed_title=preprocessed_title,
                    preprocessed_abstract=preprocessed_abstract,
                    keywords=[keyword.strip().lower() for keyword in paper["keywords"]],
                    pdf_url=paper["pdf_url"],
                    authors=paper["authors"],
                    language=Lang.ENGLISH,
                )
            )
        return texts


class ANLP_2022(Conference):
    @classmethod
    def load(self, preprocess_tokenizer: Optional[WordTokenizer] = None) -> List[ConferenceText]:
        data_path = Path("data/conference/ANLP/ANLP-2022.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(GDriveObjects.ANLP_2022.value, str(data_path))

        data = json.load(open(data_path))
        keywords = data["categories"]
        papers = data["papers"]
        texts = []
        for index, paper in enumerate(papers):
            title = paper["title"]
            abstract = paper["abstract"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(title)
                abstract_tokens = preprocess_tokenizer.tokenize(abstract)
                preprocessed_title = " ".join(token.surface for token in title_tokens)
                preprocessed_abstract = " ".join(token.surface for token in abstract_tokens)
            else:
                preprocessed_title = ""
                preprocessed_abstract = ""

            texts.append(
                ConferenceText(
                    index=index,
                    title=title,
                    abstract=abstract,
                    preprocessed_title=preprocessed_title,
                    preprocessed_abstract=preprocessed_abstract,
                    keywords=[keywords[paper["id"].split("-")[0]].strip().lower()],
                    pdf_url="",
                    authors=[],
                    language=Lang.JAPANESE if paper["language"] == "japanese" else Lang.ENGLISH,
                )
            )
        return texts


class JSAI_BASE(Conference):
    GDRIVE_OBJECT = GDriveObjects.JSAI_2022
    DATASET_PATH = Path("data/conference/JSAI/JSAI_2022.json")

    @classmethod
    def load(cls, preprocess_tokenizer: Optional[WordTokenizer] = None) -> List[ConferenceText]:
        data_path = cls.DATASET_PATH

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(cls.GDRIVE_OBJECT.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for index, paper in enumerate(papers):
            title = paper["title"]
            abstract = paper["summary"]

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(title)
                abstract_tokens = preprocess_tokenizer.tokenize(abstract)
                preprocessed_title = " ".join(token.surface for token in title_tokens)
                preprocessed_abstract = " ".join(token.surface for token in abstract_tokens)
            else:
                preprocessed_title = title
                preprocessed_abstract = preprocessed_abstract

            if paper["language"] == "japanese":
                language = Lang.JAPANESE
            elif paper["language"] == "english":
                language = Lang.ENGLISH
            else:
                language = Lang.JAPANESE

            if preprocessed_abstract == "n/a":
                continue

            texts.append(
                ConferenceText(
                    index=index,
                    title=title,
                    abstract=abstract,
                    preprocessed_title=preprocessed_title,
                    preprocessed_abstract=preprocessed_abstract,
                    keywords=paper["keywords"],
                    pdf_url=paper["url"],
                    authors=paper["authors"],
                    language=language,
                    session_schedule=paper["schedule"],
                    session_title=paper["session"],
                )
            )
        return texts


class JSAI_2023(JSAI_BASE):
    GDRIVE_OBJECT = GDriveObjects.JSAI_2023
    DATASET_PATH = Path("data/conference/JSAI/JSAI_2023.json")


class JSAI_2022(JSAI_BASE):
    GDRIVE_OBJECT = GDriveObjects.JSAI_2022
    DATASET_PATH = Path("data/conference/JSAI/JSAI_2022.json")


class ACL_Base(Conference):
    GDRIVE_OBJECT = GDriveObjects.ACL_2022

    @classmethod
    def load(self, preprocess_tokenizer: Optional[WordTokenizer] = None) -> List[ConferenceText]:
        data_path = Path(f"data/conference/ACL/{self.GDRIVE_OBJECT.name}.json")

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_from_google_drive(self.GDRIVE_OBJECT.value, str(data_path))

        papers = json.load(open(data_path))
        texts = []
        for paper in papers:
            title = paper.pop("title")
            abstract = paper.pop("abstract")

            if preprocess_tokenizer:
                title_tokens = preprocess_tokenizer.tokenize(title)
                abstract_tokens = preprocess_tokenizer.tokenize(abstract)
                preprocessed_title = " ".join(token.surface for token in title_tokens)
                preprocessed_abstract = " ".join(token.surface for token in abstract_tokens)
            else:
                preprocessed_title = ""
                preprocessed_abstract = ""

            venue = ""
            if "venue" in paper:
                venue = paper.pop("venue")
            elif "venues" in paper:
                venue = paper.pop("venues")

            texts.append(
                ConferenceText(
                    index=paper.pop("index"),
                    title=title,
                    abstract=abstract,
                    preprocessed_title=preprocessed_title,
                    preprocessed_abstract=preprocessed_abstract,
                    pdf_url=paper.pop("pdf_url") if "pdf_url" in paper else "",
                    authors=paper.pop("authors"),
                    language=Lang.ENGLISH,
                    published_at=parse_date(f"{paper['year']} {paper['month']}"),
                    venue=venue,
                    **paper,
                )
            )
        return texts


class ACL_2023(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2023


class ACL_2022(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2022


class ACL_2021(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2021


class ACL_2020(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2020


class ACL_2019(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2019


class ACL_2018(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2018


class ACL_2017(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2017


class ACL_2016(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2016


class ACL_2015(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2015


class ACL_2014(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2014


class ACL_2013(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2013


class ACL_2012(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2012


class ACL_2011(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2011


class ACL_2010(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2010


class ACL_2009(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2009


class ACL_2008(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2008


class ACL_2007(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2007


class ACL_2006(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2006


class ACL_2005(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2005


class ACL_2004(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2004


class ACL_2003(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2003


class ACL_2002(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2002


class ACL_2001(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2001


class ACL_2000(ACL_Base):
    GDRIVE_OBJECT = GDriveObjects.ACL_2000
