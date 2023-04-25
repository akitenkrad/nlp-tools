import json
from os import PathLike
from pathlib import Path
from typing import List

from stats.stats import DocStat, DocStatTarget, Sentence
from utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Report(object):
    def __init__(self, dataset_name: str, texts: List[Sentence], targets: List[DocStatTarget]):
        self.stats: DocStat = DocStat(dataset_name)
        self.stats.analyze(texts, targets)
        self.targets = targets

    def report(self, output_dir: PathLike):
        total = 1
        if DocStatTarget.BASIC_STATISTICS in self.targets:
            total += 0
        if DocStatTarget.TOPIC_MODEL in self.targets:
            total += 6
        if DocStatTarget.KEYWORD_STATISTICS in self.targets:
            total += 1

        with tqdm(total=total, desc="Reporting...", leave=True) as progress:

            def update_progress(text: str):
                progress.update(1)
                progress.set_description(text)

            meta_data = {
                "dataset_name": self.stats.dataset_name,
            }

            # 1. prepare directory
            # ----------------------------------------------------------
            update_progress("Reporting: Prepare Directory...")
            out_dir: Path = Path(output_dir) / self.stats.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Topic Model
            # ----------------------------------------------------------
            if DocStatTarget.TOPIC_MODEL in self.targets:
                topic_model_out_dir = out_dir / "topic_model"
                # 2. intertopic Distance Map
                update_progress("Reporting: Topic Model - Intertopic Distance Map")
                self.stats.topic_model_stats.save_intertopic_distance_map(
                    self.stats.topic_model,
                    topic_model_out_dir / "intertopic_distance_map" / "report.html",
                )

                # 3. Hierarchical Clustering
                update_progress("Reporting: Topic Model - Hierarchical Clustering")
                self.stats.topic_model_stats.save_hierarchical_clustering(
                    self.stats.topic_model,
                    topic_model_out_dir / "hierarchical_clustering" / "report.html",
                )

                # 4. BarChart
                update_progress("Reporting: Topic Model - BarChart")
                self.stats.topic_model_stats.save_bar_chart(
                    self.stats.topic_model,
                    topic_model_out_dir / "barchart" / "report.html",
                    n_words=8,
                )

                # 5. Similarity Matrix
                update_progress("Reporting: Topic Model - Similarity Matrix")
                self.stats.topic_model_stats.save_similarity_matrix(
                    self.stats.topic_model,
                    topic_model_out_dir / "similarity_matrix" / "report.html",
                )

                # 6. Topics Per Cpass
                update_progress("Reporting: Topic Model - Topics per Class")
                self.stats.topic_model_stats.save_topics_per_class(
                    self.stats.topic_model,
                    topic_model_out_dir / "topics_per_class" / "report.html",
                    top_n_topics=50,
                )

                # 7. Topic Probability Distribution
                update_progress("Reporting: Topic Model - Topic Probability Distribution")
                self.stats.topic_model_stats.save_topic_prob_dist(
                    self.stats.topic_model,
                    topic_model_out_dir / "topic_model_prob_dist",
                    min_probability=0.001,
                )

            # Keyword Statistics
            # ----------------------------------------------------------
            if DocStatTarget.KEYWORD_STATISTICS in self.targets:
                keyword_out_dir = out_dir / "keyword_stats"
                self.stats.keyword_stats.save_keywords()
                # 8. Save Word Cloud
                update_progress("Reporting: Keyword Statistics - Word Cloud")
                self.stats.keyword_stats.save_keyword_wordcloud(keyword_out_dir / "word_cloud", top_n_keywords=50)

            # save meta data
            # ----------------------------------------------------------
            if DocStatTarget.TOPIC_MODEL in self.targets:
                meta_data["topic_model"] = self.stats.topic_model_stats.meta_data
            if DocStatTarget.KEYWORD_STATISTICS in self.targets:
                meta_data["keyword_stats"] = self.stats.keyword_stats.meta_data
            json.dump(
                meta_data,
                open(out_dir / "meta.json", mode="wt", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
