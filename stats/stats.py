from collections import defaultdict
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MeCab
import numpy as np
import pandas as pd
import unidic
from bertopic import BERTopic
from matplotlib.figure import Figure
from nltk import FreqDist
from utils.data import Text
from utils.tokenizers import WordTokenizer
from utils.utils import Lang, is_notebook, word_cloud

if is_notebook():
    from tqdm.notebook import tqdm
else:
    import tqdm


class DocStatTarget(Enum):
    BASIC_STATISTICS = "basic_statistics"
    TOPIC_MODEL = "topic_model"
    KEYWORD_STATISTICS = "keyword_statistics"


class TopicModelStats(object):
    def __init__(self):
        self.topics: List[int] = []
        self.probs: np.ndarray = np.ndarray([])
        self.topics_per_class: pd.DataFrame = pd.DataFrame()
        self.texts: List[Text] = []
        self.__meta_data = {"topics": self.topics}

    @property
    def meta_data(self) -> dict:
        return self.__meta_data

    def __assert_out_path(self, out_path: PathLike):
        assert Path(out_path).suffix == ".html", f"out_path should be html format: {out_path}"

    def __save_fig(self, fig: Figure, out_path: PathLike):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, mode="wt", encoding="utf-8") as wf:
            wf.write(fig.to_html())

    def get_topic(self, topic_model: BERTopic, text: Text) -> Dict[int, float]:
        topics = {idx: 0.0 for idx in topic_model.topics.keys()}
        _topics, _probs = topic_model.find_topics(text.title + " " + text.summary)
        for topic, prob in zip(_topics, _probs):
            topics[topic] = prob
        return topics

    def save_intertopic_distance_map(self, topic_model: BERTopic, out_path: PathLike):
        """save intertopic distance map as html

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)

        fig = topic_model.visualize_topics()
        self.__save_fig(fig, out_path)
        self.__meta_data["intertopic_distance_map"] = {
            "width": fig.layout.width,
            "height": fig.layout.height,
        }

    def save_hierarchical_clustering(self, topic_model: BERTopic, out_path: PathLike):
        """save hierarchical clustering as hitml

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)

        fig = topic_model.visualize_hierarchy()
        self.__save_fig(fig, out_path)
        self.__meta_data["hierarchical_clustering"] = {
            "width": fig.layout.width,
            "height": fig.layout.height,
        }

    def save_bar_chart(self, topic_model: BERTopic, out_path: PathLike, n_words=8, width=300):
        """save bar chart as html

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
            n_words (int): the argument for BERTopic.visualize_barchart()
            width (int): the argument for BERTopic.visualize_barchart()
        """
        self.__assert_out_path(out_path)

        fig = topic_model.visualize_barchart(top_n_topics=len(topic_model.topics), n_words=n_words, width=width)
        self.__save_fig(fig, out_path)
        self.__meta_data["barchart"] = {
            "width": fig.layout.width,
            "height": fig.layout.height,
        }

    def save_similarity_matrix(self, topic_model: BERTopic, out_path: PathLike):
        """save similarity matrix as html

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
        """
        self.__assert_out_path(out_path)

        fig = topic_model.visualize_heatmap()
        self.__save_fig(fig, out_path)
        self.__meta_data["similarity_matrix"] = {
            "width": fig.layout.width,
            "height": fig.layout.height,
        }

    def save_topics_per_class(self, topic_model: BERTopic, out_path: PathLike, top_n_topics=50):
        """save topics per class as html

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
            top_n_topics (int): the argument for BERTopic.visualize_topics_per_class()
        """
        self.__assert_out_path(out_path)
        fig = topic_model.visualize_topics_per_class(self.topics_per_class, top_n_topics=top_n_topics)
        self.__save_fig(fig, out_path)
        self.__meta_data["topics_per_class"] = {
            "width": fig.layout.width,
            "height": fig.layout.height,
        }

    def save_topic_prob_dist(self, topic_model: BERTopic, out_path: PathLike, min_probability=0.001):
        """save topic probability distribution per Text as html format

        Args:
            topic_model (BERTopic): trained BERTopic Model
            out_path (PathLike): output file path (html)
            nim_probability (float): the argument for BERTopic.visualize_distribution()
        """
        Path(out_path).mkdir(parents=True, exist_ok=True)
        self.__meta_data["topic_prob_dist"] = []
        for idx, text in enumerate(tqdm(self.texts, desc="Reporting Topic Prob Dist...", leave=False)):
            fig = topic_model.visualize_distribution(text.prob, min_probability=min_probability)
            path = Path(out_path) / f"report_{idx:08d}.html"
            with open(path, mode="wt", encoding="utf-8") as wf:
                wf.write(fig.to_html())
            topics = self.get_topic(topic_model, text)
            self.__meta_data["topic_prob_dist"].append(
                {
                    "title": text.title,
                    "htmlfile": path.name,
                    "topics": topics,
                    "width": fig.layout.width,
                    "height": fig.layout.height,
                }
            )


class KeywordStats(object):
    def __init__(self):
        self.keywords: Dict[str, List[Text]] = defaultdict(lambda: list())
        self.keyword_cnt = FreqDist()
        self.__meta_data = {}

    @property
    def meta_data(self):
        return self.__meta_data

    def __assert_out_path(self, out_path: PathLike):
        assert Path(out_path).suffix == ".html", f"out_path should be html format: {out_path}"

    def save_keywords(self):
        """save keywords data into meta data"""
        self.__meta_data["keywords"] = []
        for keyword, texts in self.keywords.items():
            self.__meta_data["keywords"].append({"keyword": keyword, "texts": [{"title": text.title, "summary": text.summary} for text in texts]})

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
            self.__meta_data["top_n_keywords"].append({"keyword": keyword, "count": cnt, "word_cloud_file": f"{idx:03d}.png"})
            texts = self.keywords[keyword]
            input_text = " ".join([text.preprocess() for text in texts])
            word_cloud(input_text, Path(out_path) / f"{idx:03d}.png")


class DocStat(object):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.topic_model: BERTopic = BERTopic(calculate_probabilities=True)
        self.tokenizer = WordTokenizer(
            language=Lang.ENGLISH,
            remove_stopwords=True,
            remove_punctuations=True,
            stemming=True,
            add_tag=True,
        )
        self.topic_model_stats = TopicModelStats()
        self.keyword_stats = KeywordStats()

    def analyze(self, texts: List[Text], targets: List[DocStatTarget]):
        total = len(targets)
        with tqdm(total=total, desc="analyzing...", leave=False) as progress:

            def update_progress(desc_text: str):
                progress.update(1)
                progress.set_description(desc_text)

            # Basic analysis
            # ----------------------------------------------------------
            if DocStatTarget.BASIC_STATISTICS in targets:
                update_progress("Basic Analysis...")

            # Topic Modeling
            # ----------------------------------------------------------
            if DocStatTarget.TOPIC_MODEL in targets:
                update_progress("Topic Modeling...")
                texts_for_tp = [text.preprocess() for text in texts]
                topics, probs = self.topic_model.fit_transform(texts_for_tp)
                self.topic_model_stats.topics = topics
                self.topic_model_stats.probs = probs

                # Topics per Class
                classes = [text.keywords[0] for text in texts]
                topics_per_class = self.topic_model.topics_per_class(texts_for_tp, topics, classes=classes)
                self.topic_model_stats.topics_per_class = topics_per_class

                for topic, text, prob in zip(topics, texts, probs):
                    text.topic = topic
                    text.prob = prob
                self.topic_model_stats.texts = texts

            # Keyword Statistics
            # ----------------------------------------------------------
            if DocStatTarget.KEYWORD_STATISTICS in targets:
                update_progress("Keyword Statistics...")
                for text in texts:
                    for keyword in text.keywords:
                        self.keyword_stats.keywords[keyword].append(text)
                        self.keyword_stats.keyword_cnt[keyword] += 1
