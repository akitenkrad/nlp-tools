import hashlib
import json
import os
import time
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import networkx as nx
import numpy as np
from dateutil.parser import ParserError
from dateutil.parser import parse as date_parse

from utils.semanticscholar import SemanticScholar
from utils.utils import is_notebook, now, timedelta2HMS

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

Author = namedtuple("Author", ("author_id", "author_name"))
RefPaper = namedtuple("RefPaper", ("paper_id", "title"))


class Paper(object):
    def __init__(self, dict_data: Dict[str, Any]):
        self.__dict_data: Dict[str, Any] = dict_data

    def __get(self, key: str, default: Any) -> Any:
        if key in self.__dict_data:
            return self.__dict_data[key]
        else:
            return default

    @property
    def hdf5_key(self):
        """key for hdf5"""
        return f"/papers/{self.paper_id[0]}/{self.paper_id[1]}/{self.paper_id[2]}/{self.paper_id}"

    @property
    def paper_id(self) -> str:
        """paper id from SemanticScholar"""
        return self.__get("paper_id", default="")

    @property
    def url(self) -> str:
        """url from SemanticScholar"""
        return self.__get("url", default="")

    @property
    def title(self) -> str:
        """title from SemanticScholar"""
        return self.__get("title", default="")

    @property
    def abstract(self) -> str:
        """abstract from SemanticScholar"""
        return self.__get("abstract", default="")

    @property
    def venue(self) -> str:
        """venue from SemanticScholar"""
        return self.__get("venue", default="")

    @property
    def year(self) -> int:
        """year from SemanticScholar"""
        return int(self.__get("year", default=-1))

    @property
    def reference_count(self) -> int:
        """reference count from SemanticScholar"""
        return int(self.__get("reference_count", default=0))

    @property
    def citation_count(self) -> int:
        """citation count from SemanticScholar"""
        return int(self.__get("citation_count", default=0))

    @property
    def influential_citation_count(self) -> int:
        """influential citation count from SemanticScholar"""
        return int(self.__get("influential_citation_count", default=0))

    @property
    def is_open_access(self) -> bool:
        """is open access from SemanticScholar"""
        return self.__get("is_open_access", default=False)

    @property
    def fields_of_study(self) -> List[str]:
        """fields of study from SemanticScholar"""
        return self.__get("fields_of_study", default=[])

    @property
    def authors(self) -> List[Author]:
        """authors from SemanticScholar"""
        author_list = self.__get("authors", default=[])
        return [Author(a["author_id"], a["author_name"]) for a in author_list]

    @property
    def citations(self) -> List[RefPaper]:
        """citations from SemanticScholar"""
        citation_list = self.__get("citations", default=[])
        return [RefPaper(p["paper_id"], p["title"]) for p in citation_list]

    @property
    def references(self) -> List[RefPaper]:
        """references from SemanticScholar"""
        reference_list = self.__get("references", default=[])
        return [RefPaper(p["paper_id"], p["title"]) for p in reference_list]

    @property
    def doi(self) -> str:
        """doi from arxiv"""
        return self.__get("doi", default="")

    @property
    def arxiv_primary_category(self) -> str:
        """primary_category from arxiv"""
        return self.__get("arxiv_primary_category", default="")

    @property
    def arxiv_categories(self) -> List[str]:
        """categories from arxiv"""
        return self.__get("arxiv_categories", default=[])

    @property
    def updated(self) -> Optional[str]:
        """updated from arxiv"""
        value = self.__get("updated", default="")
        return value if value else ""

    @property
    def published(self) -> Optional[str]:
        """published from arxiv"""
        value = self.__get("published", default="")
        return value if value else ""

    @property
    def arxiv_id(self) -> str:
        """id from arxiv"""
        return self.__get("arxiv_id", default="")

    @property
    def arxiv_title(self) -> str:
        """title from arxiv"""
        return self.__get("arxiv_title", default="")

    @property
    def arxiv_hash(self) -> str:
        """arxiv hash"""
        return hashlib.md5((self.arxiv_title + self.arxiv_id).encode("utf-8")).hexdigest()

    @property
    def has_arxiv_info(self) -> bool:
        return len(self.arxiv_id) > 0

    def __str__(self):
        return f"<Paper id:{self.paper_id} title:{self.title[:15]}... >"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict_data


class Papers(object):
    HDF5_STR = h5py.string_dtype(encoding="utf-8")

    @classmethod
    def to_key(cls, paper_id: str):
        return f"/papers/{paper_id[0]}/{paper_id[1]}/{paper_id[2]}/{paper_id}"

    def __init__(self, hdf5_path: PathLike):
        self.hdf5_path = Path(hdf5_path)
        self.ss = SemanticScholar()
        self.indices = self.load_index()

    def load_index(self):
        with h5py.File(self.hdf5_path, mode="a") as hdf5:
            indices = [item.decode("utf-8") for item in np.array(hdf5["papers/indices"], dtype=Papers.HDF5_STR)] if "papers/indices" in hdf5 else []
            indices = [item for item in indices if len(item) > 0]
            return indices

    def update_index(self, indices: List[str]):
        """update index if hdf5 database -> /papers/indices"""
        with h5py.File(self.hdf5_path, mode="a") as hdf5:

            from_indices = []
            for index in tqdm(indices, leave=False):
                path = f"papers/{index[0]}/{index[1]}/{index[2]}/{index}"
                if path not in hdf5:
                    print(f"WARNING: found unknown index: {index}")
                    continue
                from_indices.append(index)
            from_indices = list(set(from_indices))

            group = hdf5.require_group("papers")
            del hdf5["papers/indices"]
            to_indices = group.require_dataset(name="indices", shape=(len(from_indices),), dtype=Papers.HDF5_STR)
            to_indices[...] = np.array(sorted(from_indices), dtype=Papers.HDF5_STR)

    def is_exists(self, paper_id: str) -> bool:
        """check if the specified paper exists in hdf5"""
        with h5py.File(self.hdf5_path, mode="a") as hdf5:
            key = Papers.to_key(paper_id)
            return key in hdf5

    def delete_paper(self, paper_id: str):
        """delete the specified paper"""
        with h5py.File(self.hdf5_path, mode="a") as hdf5:
            key = Papers.to_key(paper_id)
            if key in hdf5:
                del hdf5[key]

    def str2datetime(self, date_str: str) -> Optional[datetime]:
        try:
            return date_parse(date_str)
        except ParserError:
            return None
        except OverflowError:
            return None

    def get_paper(self, paper_id: str, hdf5: Optional[h5py.File] = None) -> Paper:
        def _get_paper(paper_id: str, h5: h5py.File) -> Paper:
            key = Papers.to_key(paper_id)

            if key in h5:
                dict_data = {}
                dict_data["paper_id"] = h5[key]["paper_id"][0].decode("utf-8")
                dict_data["title"] = h5[key]["title"][0].decode("utf-8")
                dict_data["url"] = h5[key]["url"][0].decode("utf-8")
                dict_data["venue"] = h5[key]["venue"][0].decode("utf-8")
                dict_data["year"] = int(h5[key]["year"][0])
                dict_data["authors"] = [
                    {"author_id": item[0].decode("utf-8"), "author_name": item[1].decode("utf-8")}
                    for item in np.array(h5[key]["authors"], dtype=Papers.HDF5_STR)
                ]
                dict_data["abstract"] = h5[key]["abstract"][0].decode("utf-8")
                dict_data["reference_count"] = int(h5[key]["reference_count"][0])
                dict_data["citation_count"] = int(h5[key]["citation_count"][0])
                dict_data["references"] = [
                    {"paper_id": item[0].decode("utf-8"), "title": item[1].decode("utf-8")}
                    for item in np.array(h5[key]["references"], dtype=Papers.HDF5_STR)
                ]
                dict_data["citations"] = [
                    {"paper_id": item[0].decode("utf-8"), "title": item[1].decode("utf-8")}
                    for item in np.array(h5[key]["citations"], dtype=Papers.HDF5_STR)
                ]
                dict_data["fields_of_study"] = [item.decode("utf-8") for item in np.array(h5[key]["fields_of_study"], dtype=Papers.HDF5_STR)]
                dict_data["influencial_citation_count"] = int(h5[key]["influential_citation_count"][0])
                dict_data["is_open_access"] = bool(h5[key]["is_open_access"][0])
                dict_data["doi"] = h5[key]["doi"][0].decode("utf-8")
                dict_data["updated"] = self.str2datetime(h5[key]["updated"][0].decode("utf-8"))
                dict_data["published"] = self.str2datetime(h5[key]["published"][0].decode("utf-8"))
                dict_data["arxiv_hash"] = h5[key]["arxiv_hash"][0].decode("utf-8")
                dict_data["arxiv_id"] = h5[key]["arxiv_id"][0].decode("utf-8")
                dict_data["arxiv_title"] = h5[key]["arxiv_title"][0].decode("utf-8")
                dict_data["arxiv_primary_category"] = h5[key]["arxiv_primary_category"][0].decode("utf-8")
                dict_data["arxiv_categories"] = [cat.decode("utf-8") for cat in np.array(h5[key]["arxiv_categories"], dtype=Papers.HDF5_STR)]

                return Paper(dict_data)
            else:
                paper_data = self.ss.get_paper_detail(paper_id)
                if paper_data:
                    return Paper(paper_data)
                else:
                    raise RuntimeError(f"Paper doesn't found with ss api: {paper_id}")

        if hdf5 is not None:
            return _get_paper(paper_id, hdf5)
        else:
            with h5py.File(self.hdf5_path, mode="r") as h5:
                return _get_paper(paper_id, h5)

    def put_paper(self, paper: Paper):
        """save new paper in hdf5 file"""
        with h5py.File(self.hdf5_path, mode="a") as h5wf:

            if self.is_exists(paper.paper_id):
                return

            try:
                self.indices.append(paper.paper_id)
                # create group
                group = h5wf.require_group(paper.hdf5_key)

                # create dataset
                new_abstract = group.create_dataset(name="abstract", shape=(1,), dtype=Papers.HDF5_STR)
                new_authors = group.create_dataset(name="authors", shape=(len(paper.authors), 2), dtype=Papers.HDF5_STR)  # author_id, author_name
                new_citation_count = group.create_dataset(name="citation_count", shape=(1,), dtype=np.int32)
                new_citations = group.create_dataset(name="citations", shape=(len(paper.citations), 2), dtype=Papers.HDF5_STR)  # paper_id, title
                new_fields_of_study = group.create_dataset(name="fields_of_study", shape=(len(paper.fields_of_study),), dtype=Papers.HDF5_STR)
                new_influential_citation_count = group.create_dataset(name="influential_citation_count", shape=(1,), dtype=np.int32)
                new_is_open_access = group.create_dataset(name="is_open_access", shape=(1,), dtype=bool)
                new_paper_id = group.create_dataset(name="paper_id", shape=(1,), dtype=Papers.HDF5_STR)
                new_reference_count = group.create_dataset(name="reference_count", shape=(1,), dtype=np.int32)
                new_references = group.create_dataset(name="references", shape=(len(paper.references), 2), dtype=Papers.HDF5_STR)  # paper_id, title
                new_title = group.create_dataset(name="title", shape=(1,), dtype=Papers.HDF5_STR)
                new_url = group.create_dataset(name="url", shape=(1,), dtype=Papers.HDF5_STR)
                new_venue = group.create_dataset(name="venue", shape=(1,), dtype=Papers.HDF5_STR)
                new_year = group.create_dataset(name="year", shape=(1,), dtype=np.int32)
                new_doi = group.create_dataset(name="doi", shape=(1,), dtype=Papers.HDF5_STR)
                new_updated = group.create_dataset(name="updated", shape=(1,), dtype=Papers.HDF5_STR)
                new_published = group.create_dataset(name="published", shape=(1,), dtype=Papers.HDF5_STR)
                new_arxiv_hash = group.create_dataset(name="arxiv_hash", shape=(1,), dtype=Papers.HDF5_STR)
                new_arxiv_id = group.create_dataset(name="arxiv_id", shape=(1,), dtype=Papers.HDF5_STR)
                new_arxiv_title = group.create_dataset(name="arxiv_title", shape=(1,), dtype=Papers.HDF5_STR)
                new_arxiv_primary_category = group.create_dataset(name="arxiv_primary_category", shape=(1,), dtype=Papers.HDF5_STR)
                new_arxiv_categories = group.create_dataset(name="arxiv_categories", shape=(len(paper.arxiv_categories),), dtype=Papers.HDF5_STR)

                # store data
                new_abstract[0] = paper.abstract
                new_authors[...] = np.array([(author.author_id, author.author_name) for author in paper.authors], dtype=Papers.HDF5_STR)
                new_citation_count[0] = paper.citation_count
                new_citations[...] = np.array([(paper.paper_id, paper.title) for paper in paper.citations], dtype=Papers.HDF5_STR)
                new_fields_of_study[...] = np.array([field for field in paper.fields_of_study], dtype=Papers.HDF5_STR)
                new_influential_citation_count[0] = paper.influential_citation_count
                new_is_open_access[0] = paper.is_open_access
                new_paper_id[0] = paper.paper_id
                new_reference_count[0] = paper.reference_count
                new_references[...] = np.array([(paper.paper_id, paper.title) for paper in paper.references], dtype=Papers.HDF5_STR)
                new_title[0] = paper.title
                new_url[0] = paper.url
                new_venue[0] = paper.venue
                new_year[0] = paper.year
                new_published[0] = paper.published
                new_updated[0] = paper.updated
                new_doi[0] = paper.doi
                new_arxiv_id[0] = paper.arxiv_id
                new_arxiv_hash[0] = paper.arxiv_hash
                new_arxiv_title[0] = paper.arxiv_title
                new_arxiv_primary_category[0] = paper.arxiv_primary_category
                new_arxiv_categories[...] = np.array(paper.arxiv_categories, dtype=Papers.HDF5_STR)

            except Exception as ex:
                if paper.hdf5_key in h5wf:
                    del h5wf[paper.hdf5_key]
                raise ex

    def build_reference_graph(
        self,
        paper_id: str,
        min_influential_citation_count=1,
        max_depth=3,
        graph_dir="__cache__/graphs",
        export_interval=1000,
    ):
        """build a reference graph

        Args:
            paper_id (str): if of the root paper
            min_influential_citation_count (int): number of citation count. ignore papers with the citation count under the threshold
            max_depth (int): max depth
            cache_dir (StrOrPath): path to cache directory
            export_interval (int): export cache with the specified interval
        """
        TemporaryPaper = namedtuple(
            "TemporaryPaper",
            (
                "paper_id",
                "title",
                "year",
                "venue",
                "citations",
                "references",
                "reference_count",
                "citation_count",
                "influential_citation_count",
                "authors",
                "arxiv_primary_category",
            ),
        )

        def show_progress(
            paper_id: str,
            total: int,
            done: int,
            start: float,
            leave=True,
            export_papers=False,
            depth=1,
            paper: Optional[Paper] = None,
            ci_paper: Optional[Paper] = None,
        ):
            res = (
                f"{paper_id[:8]} -> {done:5d}/{total:5d} ({done / (total + 1e-10) * 100.0:5.2f}%) | "
                f"etime: {timedelta2HMS(int(time.time() - start))} @{now().strftime('%H:%M:%S')}"
            )

            if export_papers:
                res += f" | exported -> {len(self.indices):5d} papers"
            if paper is not None and ci_paper is not None:
                res += f" | papers: {len(self.indices):5d}"
                res += f" | {paper.paper_id[:5]} -> {ci_paper.paper_id[:5]}"
                res += f" @cc(icc): {ci_paper.citation_count:4d}({ci_paper.influential_citation_count:4d})"
                res += f' | {"=" * (depth // 100)}{"+" * ((depth % 100) // 10)}{"-" * (depth % 10)}???'

            if not leave:
                res = f"\r{res} | processing {done} papers...\r"

            if leave:
                print(res)
            else:
                print(res, end="")

        def add_edge(graph: nx.DiGraph, src: Paper, dst: Paper):
            graph.add_edge(src.paper_id, dst.paper_id)

            for paper in [src, dst]:
                if paper.paper_id is None:
                    continue
                graph.nodes[paper.paper_id]["name"] = paper.paper_id
                graph.nodes[paper.paper_id]["paper_id"] = paper.paper_id
                graph.nodes[paper.paper_id]["title"] = paper.title
                graph.nodes[paper.paper_id]["year"] = paper.year
                graph.nodes[paper.paper_id]["venue"] = paper.venue
                graph.nodes[paper.paper_id]["reference_count"] = paper.reference_count
                graph.nodes[paper.paper_id]["citation_count"] = paper.citation_count
                graph.nodes[paper.paper_id]["influential_citation_count"] = paper.influential_citation_count
                graph.nodes[paper.paper_id]["first_author_name"] = paper.authors[0].author_name if len(paper.authors) > 0 else ""
                graph.nodes[paper.paper_id]["first_author_id"] = paper.authors[0].author_id if len(paper.authors) > 0 else ""
                graph.nodes[paper.paper_id]["arxiv_primary_category"] = paper.arxiv_primary_category

        def export_graph(graph: nx.DiGraph, paper_id: str, out_dir="__graph__"):
            outfile: Path = Path(out_dir)
            outfile = outfile / paper_id[0] / paper_id[1] / paper_id[2] / f"{paper_id}.graphml"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml_lxml(graph, str(outfile.resolve().absolute()), encoding="utf-8", prettyprint=True, named_key_ids=True)

        G: nx.DiGraph = nx.DiGraph()
        stats: Dict[str, Any] = {
            "total": 0,
            "done": 0,
            "paper_queue": [],
            "new_papers": [],
            "finished_papers": [],
            "errors": [],
        }
        start = time.time()

        root_paper = self.get_paper(paper_id)
        stats["paper_queue"].insert(0, (root_paper, 0))
        stats["total"] += len(root_paper.citations)
        while 0 < len(stats["paper_queue"]):
            paper, depth = stats["paper_queue"].pop()
            if max_depth < depth:
                stats["paper_queue"] = []
                return

            for ci_ref_paper in paper.citations:
                if ci_ref_paper.paper_id is None:
                    stats["done"] += 1
                    continue

                # 1. show progress
                show_progress(paper_id, stats["total"], stats["done"], start, leave=False)

                if len(self.indices) > 0 and len(stats["new_papers"]) >= export_interval and len(self.indices) % export_interval == 0:
                    export_graph(G, paper_id, graph_dir)
                    self.update_index(self.indices)
                    stats["new_papers"] = []

                # 2. get paper detail
                try:

                    if ci_ref_paper.paper_id in stats["errors"]:
                        raise RuntimeError()

                    ci_paper: Paper = self.get_paper(ci_ref_paper.paper_id)
                    self.put_paper(ci_paper)
                    stats["new_papers"].append(ci_paper.paper_id)

                except Exception as ex:
                    print(f"Warning: {str(ex.__class__.__name__)}({str(ex)}) @{ci_ref_paper.paper_id}")
                    stats["done"] += 1
                    if len(ci_ref_paper.paper_id) > 0 and ci_ref_paper.paper_id not in stats["errors"]:
                        stats["errors"].append(ci_ref_paper.paper_id)
                    continue

                # 3. add the new paper into the list
                stats["done"] += 1
                if ci_paper.influential_citation_count >= min_influential_citation_count:
                    add_edge(G, paper, ci_paper)
                    show_progress(paper_id, stats["total"], stats["done"], start, depth=depth, paper=paper, ci_paper=ci_paper)

                    if ci_paper.paper_id not in stats["finished_papers"]:
                        stats["finished_papers"].append(ci_paper.paper_id)
                        temp_paper = TemporaryPaper(
                            ci_paper.paper_id,
                            ci_paper.title,
                            ci_paper.year,
                            ci_paper.venue,
                            ci_paper.citations,
                            ci_paper.references,
                            ci_paper.reference_count,
                            ci_paper.citation_count,
                            ci_paper.influential_citation_count,
                            ci_paper.authors,
                            ci_paper.arxiv_primary_category,
                        )
                        stats["paper_queue"].insert(0, (temp_paper, depth + 1))
                        stats["total"] += len(ci_paper.citations)

        # post process
        export_graph(G, paper_id, graph_dir)
        self.update_index(self.indices)

    def build_paper_categories_dataset(self, output_dir: PathLike):
        """build dataset for paper-category-inference

        input:
            {
                "paper_id": PAPER ID,
                "title": PAPER TITLE,
                "authors": [AUTHOR_NAME],
                "abstract": PAPER ABSTRACT
            }
        label:
            {
                "paper_id": PAPER ID,
                "categories": [PAPER CATEGORY]
            }

        Args:
            output_dir (PathLike): output directory
        """
        train_dir = Path(output_dir) / "paper_categories" / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir = Path(output_dir) / "paper_categories" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        train_label_path = train_dir / "labels.jsonl"
        train_papers_dir = train_dir / "papers"
        test_label_path = test_dir / "labels.jsonl"
        test_papers_dir = test_dir / "papers"

        with open(train_label_path, mode="w") as train_label_f, open(test_label_path, mode="w") as test_label_f:
            for paper_idx in tqdm(self.indices, desc="Building dataset"):
                paper = self.get_paper(paper_idx)

                paper_data = {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": [author.author_name for author in paper.authors],
                    "abstract": paper.abstract,
                    "year": paper.year,
                }

                if len(paper.fields_of_study) > 0:
                    # train data

                    # save label
                    label = {"paper_id": paper.paper_id, "categories": paper.fields_of_study}
                    train_label_f.write(json.dumps(label, ensure_ascii=False) + os.linesep)

                    # save text
                    paper_path = train_papers_dir / paper.paper_id[0] / paper.paper_id[1] / paper.paper_id[2] / f"{paper.paper_id}.json"
                    paper_path.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(paper_data, open(paper_path, mode="w"), ensure_ascii=False, indent=2)

                else:
                    # test data

                    # save label
                    label = {"paper_id": paper.paper_id, "categories": paper.fields_of_study}
                    test_label_f.write(json.dumps(label, ensure_ascii=False) + os.linesep)

                    # save text
                    paper_path = test_papers_dir / paper.paper_id[0] / paper.paper_id[1] / paper.paper_id[2] / f"{paper.paper_id}.json"
                    paper_path.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(paper_data, open(paper_path, mode="w"), ensure_ascii=False, indent=2)
