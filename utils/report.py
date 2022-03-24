from typing import List
from os import PathLike
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag

from utils.utils import is_notebook
from stats.stats import DocStat, Text

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class HtmlBuilder(object):
    def __init__(self, title:str):
        self.__build_base_html()
        self.__add_title(title)

    def __build_base_html(self):
        self.__soup = BeautifulSoup('<HTML></HTML>', 'html.parser')
        self.__head = self.__soup.new_tag('head')
        self.__soup.html.insert(0, self.__head)
        self.__body = self.__soup.new_tag('body')
        self.__soup.html.insert(1, self.__body)

        metatag = self.__soup.new_tag('meta', attrs={'charset': 'UTF-8'})
        self.head.insert(0, metatag)

    @property
    def html(self):
        return self.__soup.html
    @property
    def head(self):
        return self.__head
    @property
    def body(self):
        return self.__body

    def new_tag(self, name, **kwargs) -> Tag:
        return self.__soup.new_tag(name, **kwargs)

    def __add_title(self, title:str):
        title_tag = self.__soup.new_tag('title')
        title_tag.string = title
        idx = len(list(self.head.children))
        self.head.insert(idx, title_tag)

    def add_html_section(self, title:str, html_path:str, text:str=''):
        idx = len(list(self.body.children))
        h2 = self.new_tag('h2')
        h2.string = title
        iframe = self.new_tag('iframe', attrs={'src': html_path, 'target':'_blank'})

        self.body.insert(idx, h2)
        self.body.insert(idx + 1, iframe)

        if text != '':
            div = self.new_tag('div')
            div.string = text
            self.body.insert(idx + 1, div)

    def add_image_section(self, title:str, image_path:str, text:str=''):
        idx = len(list(self.body.children))
        h2 = self.new_tag('h2')
        h2.string = title
        img = self.new_tag('img', attrs={'src': image_path, 'alt': image_path})

        self.body.insert(idx, h2)
        self.body.insert(idx + 1, img)

        if text != '':
            div = self.new_tag('div')
            div.string = text
            self.body.insert(idx + 1, div)

    def add_detail_section(self, title:str, description:str='', contents:List[Tag]=[]):
        idx = len(list(self.body.children))
        details = self.new_tag('details')
        summary = self.new_tag('summary')
        summary.string = title
        details.insert(0, summary)

        if description != '':
            details.string = description
        if 0 < len(contents):
            for idx, content in enumerate(contents):
                details.insert(idx+1, content)
        
        self.body.insert(idx, details)

    def save(self, outpath:PathLike):
        html_text = self.__soup.prettify(encoding='utf-8')
        with open(outpath, mode='wt', encoding='utf-8') as wf:
            wf.write(html_text)

class Report(object):
    def __init__(self, report_title:str, texts:List[Text]):
        self.stats:DocStat = DocStat()
        self.builder:HtmlBuilder = HtmlBuilder(report_title)

        self.stats.analyze(texts)

    def report(self, out_dir:PathLike):

        with tqdm(total=10, desc='Reporting...', leave=True) as progress:
            def update_progress(text:str):
                progress.update(1)
                progress.set_description(text)

            # prepare directory
            update_progress('Reporting: Prepare Directory...')
            out_dir:Path = Path(out_dir)
            img_dir = out_dir / 'images'
            html_dir = out_dir / 'html'
            img_dir.mkdir(parents=True, exist_ok=True)
            html_dir.mkdir(parents=True, exist_ok=True)

            # Topic Model
            ## intertopic Distance Map
            update_progress('Reporting: Topic Model - Intertopic Distance Map')
            fig = self.stats.topic_model.visualize_topics(width=800, height=800)
            with open(html_dir / 'intertopic_distance_map.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_html_section('Intertopic Distance Map', str(html_dir / 'intertopic_distance_map.html'))

            ## Hierarchical Clustering
            update_progress('Reporting: Topic Model - Hierarchical Clustering')
            fig = self.stats.topic_model.visualize_hierarchy()
            with open(html_dir / 'hierarchical_clustering.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_html_section('Hierarchical Clustering', str(html_dir / 'hierarchical_clustering.html'))

            ## BarChart
            update_progress('Reporting: Topic Model - BarChart')
            fig = self.stats.topic_model.visualize_barchart(top_n_topics=len(topic_model.topics), n_words=8, width=300)
            with open(html_dir / 'barchart.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_html_section('Topic Word Score', str(html_dir / 'barchart.html'))

            ## Similarity Matrix
            update_progress('Reporting: Topic Model - Similarity Matrix')
            fig = self.stats.topic_model.visualize_heatmap()
            with open(html_dir / 'similarity_matrix.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_html_section('Similarity Matrix', str(html_dir / 'similarity_matrix.html'))

            ## Topics Per Cpass
            update_progress('Reporting: Topic Model - Topics per Class')
            fig = self.stats.topic_model.visualize_topics_per_class(self.stats.topic_model_attrs['topics_per_class'])

            ## Topic Probability Distribution
            # for topic_idx, topic_info in self.stats.topic_model_attrs['topic_docs'].items():

        # save index.html
        self.builder.save(out_dir / 'index.html')