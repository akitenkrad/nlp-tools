import json
from collections import defaultdict
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from matplotlib.figure import Figure
from nltk import FreqDist

from utils.data import ConferenceText, Sentence
from utils.tokenizers import Tokenizer
from utils.utils import WordCloudMask, is_notebook, word_cloud

if is_notebook():
    from tqdm.notebook import tqdm
else:
    import tqdm


class DocStatTarget(Enum):
    BASIC_STATISTICS = "basic_statistics"
    TOPIC_MODEL = "topic_model"
    KEYWORD_STATISTICS = "keyword_statistics"


class ConferenceTopicModelStats(object):
    def __init__(self, embedding_model=None):
        self.topic_model: BERTopic = BERTopic(calculate_probabilities=True, embedding_model=embedding_model)
        self.topics: List[int] = []
        self.probs: np.ndarray = np.ndarray([])
        self.topics_per_class: pd.DataFrame = pd.DataFrame()
        self.texts: List[ConferenceText] = []
        self.__meta_data = {}

    @property
    def meta_data(self) -> dict:
        return self.__meta_data

    def __assert_out_path(self, out_path: PathLike):
        assert Path(out_path).suffix == ".html", f"out_path should be html format: {out_path}"

    def __save_fig(self, fig: Figure, out_path: PathLike):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, mode="wt", encoding="utf-8") as wf:
            wf.write(fig.to_html())

    def fit(self, texts: List[ConferenceText]):
        input_texts = [text.text for text in texts]
        topics, probs = self.topic_model.fit_transform(input_texts)
        self.topics = topics
        self.probs = probs

        classes = []
        for text in texts:
            if len(text.keywords) > 0:
                classes.append(text.keywords[0])
            else:
                classes.append("No Keywords")
        if len(classes) > 0:
            self.topics_per_class = self.topic_model.topics_per_class(input_texts, topics, classes=classes)
        else:
            self.topics_per_class = None

        for topic, text, prob in tqdm(
            zip(topics, texts, probs), total=len(texts), desc="analyzing topics...", leave=False
        ):
            text.topic = topic
            text.topic_prob = prob
        self.texts = texts
        self.__meta_data["topics"] = topics

    def get_topic(self, text: Sentence) -> Tuple[int, str, Dict[int, float]]:
        topics = {idx: 0.0 for idx in self.topic_model.topics.keys()}
        [_topic], _probs = self.topic_model.transform(text.text)
        for topic, prob in zip(_topics, _probs):
            topics[topic] = prob
        topic_info = self.topic_model.get_topic_info()
        primary_topic_name = topic_info[topic_info["Topic"] == topic]["Name"].values[0]
        return topic, primary_topic_name, topics

    def save_topic_info(self, out_path: PathLike):
        """save topic info as csv

        Args:
            out_path (PathLike): output file path (html)
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv(str(out_path), index=False, header=True)

    def save_intertopic_distance_map(self, out_path: PathLike):
        """save intertopic distance map as html

        Args:
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)
        fig = self.topic_model.visualize_topics()
        self.__save_fig(fig, out_path)

    def save_hierarchical_clustering(self, out_path: PathLike):
        """save hierarchical clustering as hitml

        Args:
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)
        fig = self.topic_model.visualize_hierarchy()
        self.__save_fig(fig, out_path)

    def save_bar_chart(self, out_path: PathLike, n_words=8, width=300):
        """save bar chart as html

        Args:
            out_path (PathLike): output file path (html)
            n_words (int): the argument for BERTopic.visualize_barchart()
            width (int): the argument for BERTopic.visualize_barchart()
        """
        self.__assert_out_path(out_path)
        fig = self.topic_model.visualize_barchart(
            top_n_topics=len(self.topic_model.topics), n_words=n_words, width=width
        )
        self.__save_fig(fig, out_path)

    def save_similarity_matrix(self, out_path: PathLike):
        """save similarity matrix as html

        Args:
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)

        fig = self.topic_model.visualize_heatmap()
        self.__save_fig(fig, out_path)

    def save_topics_per_class(self, out_path: PathLike, top_n_topics=50):
        """save topics per class as html

        Args:
            out_path (PathLike): output file path (html)
            top_n_topics (int): the argument for BERTopic.visualize_topics_per_class()
        """
        self.__assert_out_path(out_path)
        fig = self.topic_model.visualize_topics_per_class(self.topics_per_class, top_n_topics=top_n_topics)
        self.__save_fig(fig, out_path)

    def save_topic_prob_dist(self, out_path: PathLike, min_probability=1e-10):
        """save topic probability distribution per Text as html format

        Args:
            out_path (PathLike): output file path (html)
            nim_probability (float): the argument for BERTopic.visualize_distribution()
        """
        Path(out_path).mkdir(parents=True, exist_ok=True)
        self.__meta_data["topic_prob_dist"] = []
        for idx, text in enumerate(tqdm(self.texts, desc="Reporting Topic Prob Dist...", leave=False)):
            fig = self.topic_model.visualize_distribution(text.topic_prob, min_probability=min_probability)
            path = Path(out_path) / f"report_{idx:08d}.html"
            with open(path, mode="wt", encoding="utf-8") as wf:
                wf.write(fig.to_html())
            _, primary_topic_name, topics = self.get_topic(text)
            self.__meta_data["topic_prob_dist"].append(
                {
                    "title": text.original_title,
                    "htmlfile": path.name,
                    "topics": topics,
                    "primary_topic": primary_topic_name,
                }
            )


class KeywordStats(object):
    def __init__(self):
        self.keywords: Dict[str, List[Sentence]] = defaultdict(lambda: list())
        self.keyword_cnt = FreqDist()
        self.__meta_data = {}

    @property
    def meta_data(self):
        return self.__meta_data

    def __assert_out_path(self, out_path: PathLike):
        assert Path(out_path).suffix == ".html", f"out_path should be html format: {out_path}"

    def fit(self, texts: List[ConferenceText]):
        for text in tqdm(texts, desc="analyzing keywords...", leave=False):
            for keyword in text.keywords:
                self.keywords[keyword].append(text)
                self.keyword_cnt[keyword] += 1

    def save_keywords(self):
        """save keywords data into meta data"""
        self.__meta_data["keywords"] = []
        for keyword, texts in self.keywords.items():
            self.__meta_data["keywords"].append(
                {
                    "keyword": keyword,
                    "texts": [
                        {
                            "original_title": text.original_title,
                            "preprocessed_title": text.preprocessed_title,
                            "original_summary": text.original_summary,
                            "preprocessed_summary": text.preprocessed_summary,
                        }
                        for text in texts
                    ],
                }
            )

    def save_keyword_wordcloud(self, out_path: PathLike, top_n_keywords=25):
        """save WordCloud image for top-n keywords

        Args:
            out_path (PathLike): output directory
            top_n_keywords (int): keywords to save WordCloud
        """
        Path(out_path).mkdir(parents=True, exist_ok=True)
        self.__meta_data["top_n_keywords"] = []
        for idx, (keyword, cnt) in enumerate(
            tqdm(
                self.keyword_cnt.most_common(top_n_keywords),
                desc="Saving keyword word-cloud...",
                total=top_n_keywords,
                leave=False,
            )
        ):
            self.__meta_data["top_n_keywords"].append(
                {"keyword": keyword, "count": cnt, "word_cloud_file": f"{idx:03d}.png"}
            )
            texts = self.keywords[keyword]
            input_text = " ".join([text.text for text in texts])
            word_cloud(input_text, Path(out_path) / f"{keyword}.png", mask_type=WordCloudMask.CIRCLE)


class ConferenceStats(object):
    def __init__(self, conference_name: str, tokenizer: Tokenizer, embedding_for_topic_model=None):
        self.conference_name: str = conference_name
        self.tokenizer: Tokenizer = tokenizer
        self.topic_model_stats = ConferenceTopicModelStats(embedding_for_topic_model)
        self.keyword_stats = KeywordStats()

    def __update_progress(self, progress: tqdm, desc_text: str):
        progress.update(1)
        progress.set_description(desc_text)

    def analyze(self, texts: List[ConferenceText]):
        total = 3
        with tqdm(total=total, desc="analyzing...", leave=False) as progress:

            self.__update_progress(progress, "Basic Analysis...")
            # TODO: add basic stats

            self.__update_progress(progress, "Topic Modeling...")
            self.topic_model_stats.fit(texts)

            self.__update_progress(progress, "Keyword Statistics...")
            self.keyword_stats.fit(texts)

    def report(self, outdir: PathLike):
        total = 1
        total += 0  # DocStatTarget.BASIC_STATISTICS
        total += 7  # DocStatTarget.TOPIC_MODEL
        total += 1  # DocStatTarget.KEYWORD_STATISTICS

        with tqdm(total=total, desc="Reporting...", leave=True) as progress:

            meta_data: Dict[str, Any] = {
                "conference_name": self.conference_name,
            }

            # 1. prepare directory
            # ----------------------------------------------------------
            self.__update_progress(progress, "Reporting: Prepare Directory...")
            out_dir: Path = Path(outdir) / self.conference_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Topic Model
            # ----------------------------------------------------------
            topic_model_out_dir = out_dir / "topic_model"
            # 2. intertopic Distance Map
            self.__update_progress(progress, "Reporting: Topic Model - Intertopic Distance Map")
            self.topic_model_stats.save_intertopic_distance_map(
                topic_model_out_dir / "intertopic_distance_map.html",
            )

            # 3. Hierarchical Clustering
            self.__update_progress(progress, "Reporting: Topic Model - Hierarchical Clustering")
            self.topic_model_stats.save_hierarchical_clustering(
                topic_model_out_dir / "hierarchical_clustering.html",
            )

            # 4. BarChart
            self.__update_progress(progress, "Reporting: Topic Model - BarChart")
            self.topic_model_stats.save_bar_chart(
                topic_model_out_dir / "barchart.html",
                n_words=8,
            )

            # 5. Similarity Matrix
            self.__update_progress(progress, "Reporting: Topic Model - Similarity Matrix")
            self.topic_model_stats.save_similarity_matrix(
                topic_model_out_dir / "similarity_matrix.html",
            )

            # 6. Topics Per Cpass
            self.__update_progress(progress, "Reporting: Topic Model - Topics per Class")
            if self.topic_model_stats.topics_per_class is not None:
                self.topic_model_stats.save_topics_per_class(
                    topic_model_out_dir / "topics_per_class.html",
                    top_n_topics=50,
                )

            # 7. Topic Probability Distribution
            self.__update_progress(progress, "Reporting: Topic Model - Topic Probability Distribution")
            self.topic_model_stats.save_topic_prob_dist(
                topic_model_out_dir / "topic_model_prob_dist",
                min_probability=1e-10,
            )

            # 8. Save Topic Info
            self.__update_progress(progress, "Reporting: Topic Model - Topic Info")
            self.topic_model_stats.save_topic_info(
                topic_model_out_dir / "topic_info.csv",
            )

            # Keyword Statistics
            # ----------------------------------------------------------
            keyword_out_dir = out_dir / "keyword_stats"
            self.keyword_stats.save_keywords()
            # 9. Save Word Cloud
            self.__update_progress(progress, "Reporting: Keyword Statistics - Word Cloud")
            self.keyword_stats.save_keyword_wordcloud(keyword_out_dir / "word_cloud", top_n_keywords=50)

            # save meta data
            # ----------------------------------------------------------
            meta_data["topic_model"] = self.topic_model_stats.meta_data
            meta_data["keyword_stats"] = self.keyword_stats.meta_data
            json.dump(
                meta_data,
                open(out_dir / "meta.json", mode="wt", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
