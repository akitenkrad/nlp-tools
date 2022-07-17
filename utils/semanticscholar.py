import json
import re
import socket
import string
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.error import HTTPError, URLError

from attrdict import AttrDict
from sumeval.metrics.rouge import RougeCalculator


class SemanticScholar(object):
    API: Dict[str, str] = {
        "search_by_title": "https://api.semanticscholar.org/graph/v1/paper/search?{QUERY}",
        "search_by_id": "https://api.semanticscholar.org/graph/v1/paper/{PAPER_ID}?{PARAMS}",
    }
    CACHE_PATH: Path = Path("__cache__/papers.pickle")

    def __init__(self, threshold: float = 0.95):
        self.__api = AttrDict(self.API)
        self.__rouge = RougeCalculator(stopwords=True, stemming=False, word_limit=-1, length_limit=-1, lang="en")
        self.__threshold = threshold

    @property
    def threshold(self) -> float:
        return self.__threshold

    def __retry_and_wait(self, msg: str, ex: Union[HTTPError, URLError, socket.timeout, Exception], retry: int) -> int:
        retry += 1
        if 5 < retry:
            raise ex
        if retry == 1:
            msg = "\n" + msg

        print(msg)

        if isinstance(ex, HTTPError) and ex.errno == -3:
            time.sleep(300.0)
        else:
            time.sleep(5.0)
        return retry

    def get_paper_id(self, title: str) -> str:

        # remove punctuation
        title = title
        for punc in string.punctuation:
            title = title.replace(punc, " ")
        title = re.sub(r"\s\s+", " ", title, count=1000)

        retry = 0
        while retry < 5:
            try:
                params = {
                    "query": title,
                    "fields": "title",
                    "offset": 0,
                    "limit": 100,
                }
                response = urllib.request.urlopen(self.__api.search_by_title.format(QUERY=urllib.parse.urlencode(params)), timeout=5.0)
                content = json.loads(response.read().decode("utf-8"))
                time.sleep(3.5)
                break

            except HTTPError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except URLError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except socket.timeout as ex:
                retry = self.__retry_and_wait(f"API Timeout -> Retry: {retry}", ex, retry)
            except Exception as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)

            if 5 <= retry:
                print(f"No paper-id found @ {title}")
                return ""

        for item in content["data"]:
            # remove punctuation
            ref_str = item["title"].lower()
            for punc in string.punctuation:
                ref_str = ref_str.replace(punc, " ")
            ref_str = re.sub(r"\s\s+", " ", ref_str, count=1000)

            score = self.__rouge.rouge_l(summary=title.lower(), references=ref_str)
            if score > self.threshold:
                return item["paperId"].strip()
        return ""

    def get_paper_detail(self, paper_id: str) -> Optional[Dict]:

        retry = 0
        while retry < 5:
            try:
                fields = [
                    "paperId",
                    "url",
                    "title",
                    "abstract",
                    "venue",
                    "year",
                    "referenceCount",
                    "citationCount",
                    "influentialCitationCount",
                    "isOpenAccess",
                    "fieldsOfStudy",
                    "authors",
                    "citations",
                    "references",
                ]
                params = f'fields={",".join(fields)}'
                response = urllib.request.urlopen(self.__api.search_by_id.format(PAPER_ID=paper_id, PARAMS=params), timeout=5.0)
                time.sleep(3.5)
                break

            except HTTPError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except URLError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except socket.timeout as ex:
                retry = self.__retry_and_wait(f"API Timeout -> Retry: {retry}", ex, retry)
            except Exception as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)

            if 5 <= retry:
                raise Exception(f"No paper found @ {paper_id}")

        content = json.loads(response.read().decode("utf-8"))

        dict_data = {}
        dict_data["paper_id"] = content["paperId"]
        dict_data["url"] = content["url"]
        dict_data["title"] = content["title"]
        dict_data["abstract"] = content["abstract"]
        dict_data["venue"] = content["venue"]
        dict_data["year"] = content["year"]
        dict_data["reference_count"] = content["referenceCount"]
        dict_data["citation_count"] = content["citationCount"]
        dict_data["influential_citation_count"] = content["influentialCitationCount"]
        dict_data["is_open_access"] = content["isOpenAccess"]
        dict_data["fields_of_study"] = content["fieldsOfStudy"]
        dict_data["authors"] = [{"author_id": item[0], "author_name": item[1]} for item in content["authors"]]
        dict_data["citations"] = [{"paper_id": item[0], "title": item[1]} for item in content["citations"]]
        dict_data["references"] = [{"paper_id": item[0], "title": item[1]} for item in content["references"]]

        return dict_data
