import json
from os import PathLike
from pathlib import Path
from typing import List

from stats.stats import DocStat, Text

from utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Report(object):
    def __init__(self, dataset_name: str, texts: List[Text]):
        self.stats: DocStat = DocStat(dataset_name)
        self.stats.analyze(texts)

    def report(self, output_dir: PathLike):
        total = 7
        with tqdm(total=total, desc='Reporting...', leave=True) as progress:

            def update_progress(text: str):
                progress.update(1)
                progress.set_description(text)

            meta_json = {
                'dataset_name': self.stats.dataset_name,
                'topic_model': {
                    'topic_prob_dist': [],
                }
            }

            # 1. prepare directory
            update_progress('Reporting: Prepare Directory...')
            out_dir: Path = Path(output_dir) / self.stats.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Topic Model
            topic_model_out_dir = out_dir / 'topic_model'
            # 2. intertopic Distance Map
            update_progress('Reporting: Topic Model - Intertopic Distance Map')
            fig = self.stats.topic_model.visualize_topics()
            html_dir = topic_model_out_dir / 'intertopic_distance_map'
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / 'report.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            meta_json['topic_model']['intertopic_distance_map'] = {'width': fig.layout.width, 'height': fig.layout.height}

            # 3. Hierarchical Clustering
            update_progress('Reporting: Topic Model - Hierarchical Clustering')
            fig = self.stats.topic_model.visualize_hierarchy()
            html_dir = topic_model_out_dir / 'hierarchical_clustering'
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / 'report.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            meta_json['topic_model']['hierarchical_clustering'] = {'width': fig.layout.width, 'height': fig.layout.height}

            # 4. BarChart
            update_progress('Reporting: Topic Model - BarChart')
            fig = self.stats.topic_model.visualize_barchart(
                top_n_topics=len(self.stats.topic_model.topics), n_words=8, width=300
            )
            html_dir = topic_model_out_dir / 'barchart'
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / 'report.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            meta_json['topic_model']['barchart'] = {'width': fig.layout.width, 'height': fig.layout.height}

            # 5. Similarity Matrix
            update_progress('Reporting: Topic Model - Similarity Matrix')
            fig = self.stats.topic_model.visualize_heatmap()
            html_dir = topic_model_out_dir / 'similarity_matrix'
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / 'report.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            meta_json['topic_model']['similarity_matrix'] = {'width': fig.layout.width, 'height': fig.layout.height}

            # 6. Topics Per Cpass
            update_progress('Reporting: Topic Model - Topics per Class')
            fig = self.stats.topic_model.visualize_topics_per_class(
                self.stats.topic_model_attrs['topics_per_class'], top_n_topics=50
            )
            html_dir = topic_model_out_dir / 'topics_per_class'
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / 'report.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            meta_json['topic_model']['topics_per_class'] = {'width': fig.layout.width, 'height': fig.layout.height}

            # 7. Topic Probability Distribution
            update_progress('Reporting: Topic Model - Topic Probability Distribution')
            prob_dist_dir = topic_model_out_dir / 'topic_model_prob_dist'
            prob_dist_dir.mkdir(parents=True, exist_ok=True)
            for idx, text in enumerate(tqdm(self.stats.topic_model_attrs['texts'], desc='Reporting Topic Prob Dist...', leave=False)):
                fig = self.stats.topic_model.visualize_distribution(text.prob, min_probability=0.001)
                path = prob_dist_dir / f'report_{idx:08d}.html'
                with open(path, mode='wt', encoding='utf-8') as wf:
                    wf.write(fig.to_html())
                topics = self.stats.get_topic(text)
                meta_json['topic_model']['topic_prob_dist'].append({
                    'title': text.title,
                    'htmlfile': path.name,
                    'topics': topics,
                    'width': fig.layout.width,
                    'height': fig.layout.height
                })

            # save meta data
            json.dump(meta_json, open(out_dir / 'meta.json', mode='wt', encoding='utf-8'), ensure_ascii=False, indent=2)
