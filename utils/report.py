from typing import List
from os import PathLike
from pathlib import Path
import hashlib
import shutil
from collections import namedtuple
from bs4 import BeautifulSoup
from bs4.element import Tag

from utils.utils import is_notebook
from stats.stats import DocStat, Text

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

OptionObject = namedtuple('OptionOject', {'path', 'option'})

class HtmlBuilder(object):
    def __init__(self, title:str):
        self.__build_base_html()
        self.__add_title(title)
        self.__javascript = []

    def __build_base_html(self):
        self.__soup = BeautifulSoup('<HTML></HTML>', 'html.parser')
        self.__head = self.__soup.new_tag('head')
        self.__soup.html.insert(0, self.__head)
        self.__body = self.__soup.new_tag('body')
        self.__soup.html.insert(1, self.__body)

        metatag = self.__soup.new_tag('meta', attrs={'charset': 'UTF-8'})
        self.head.insert(0, metatag)

        css = self.__soup.new_tag('link', attrs={'rel': 'stylesheet', 'href': 'main.css'})
        self.head.insert(1, css)

        self.__js_filename = 'main.js'
        js_script = self.__soup.new_tag('script', attrs={'src': self.__js_filename})
        self.head.insert(2, js_script)

        self.__classes = {
            'select': 'select_class'
        }

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

    def new_iframe(self, html_path:str, width:int, height:int):
        return self.new_tag('iframe', attrs={'src': html_path, 'target':'_blank', 'style': f'width:{width}px; height:{height}px;'})
    
    def new_image(self, image_path:str):
        return self.new_tag('img', attrs={'src': image_path, 'alt': image_path})
    
    def __add_title(self, title:str):
        title_tag = self.__soup.new_tag('title')
        title_tag.string = title
        idx = len(list(self.head.children))
        self.head.insert(idx, title_tag)

    def add_iframe_section(self, title:str, description:str='', html_path:str='', width:int=800, height:int=800):
        idx = len(list(self.body.children))
        h2 = self.new_tag('h2')
        h2.string = title
        iframe = self.new_iframe(html_path, width, height)

        self.body.insert(idx, h2)
        self.body.insert(idx + 1, iframe)

        if description != '':
            div = self.new_tag('div')
            div.string = description
            self.body.insert(idx + 1, div)

    def add_image_section(self, title:str, image_path:str, description:str=''):
        idx = len(list(self.body.children))
        h2 = self.new_tag('h2')
        h2.string = title
        img = self.new_tag('img', attrs={'src': image_path, 'alt': image_path})

        self.body.insert(idx, h2)
        self.body.insert(idx + 1, img)

        if description != '':
            div = self.new_tag('div')
            div.string = description
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

    def add_select_section(self, title:str, options:List[OptionObject], description:str=''):
        idx = len(list(self.body.children))
        h2 = self.new_tag('h2')
        h2.string = title
        select_id = hashlib.md5((title+description).encode('utf-8')).hexdigest()
        div = self.new_tag('div', attrs={'id': f'{select_id[:5]}_div', 'class': 'select_div'})
        js_func = f'{select_id[:5]}_onChange'
        
        # select
        select = self.new_tag('select', attrs={'id': select_id, 'class':self.__classes['select'], 'onchange':f'"{js_func}();"'})
        div.insert(0, select)
        option_tags = []
        for idx, option in enumerate(options):
            opt_id = f'{select_id[:5]}_option_{idx}_id'
            opt_val = f'{select_id[:5]}_option_{idx}_val'
            opt_tag = self.new_tag('option', attrs={'id': opt_id, 'value': opt_val})
            opt_tag.string = option.option
            select.insert(idx, opt_tag)
            option_tags.append({'id': opt_id, 'tag': opt_tag, 'switch_case':
            f'''
                        case {opt_val}:
                            document.getElementById('{opt_id}').style.display = "";
            '''})

        display_all_none = []
        for option in option_tags:
            display_all_none.append(f'''
                    document.getElementById('{option["id"]}').style.display = "none";
            ''')
        display_all_none = ''.join(display_all_none)

        switch_case = ''.join(option['switch_case'] for option in options)

        js = f'''
            function {js_func}(){{
                if(document.getElementById('{select_id}')){{
                    {display_all_none}
                    opt_val = document.getElementById('{select_id}').value;
                    switch(opt_val) {{
                    {switch_case}
                }}
            window.onload = {js_func};
            }}
        '''
        self.__javascript.append(js)

        self.body.insert(idx, h2)
        self.body.insert(idx+1, div)

    def save(self, out_dir:PathLike):
        # save index.html
        html_text = self.__soup.prettify()
        with open(out_dir / 'index.html', mode='wt', encoding='utf-8') as wf:
            wf.write(html_text)

        # save javascript
        js_text = '\n'.join(self.__javascript)
        with open(out_dir / self.__js_filename, mode='wt', encoding='utf-8') as wf:
            wf.write(js_text)

        # copy css
        css_from = Path(__file__).parent.parent / 'resources' / 'html_rsc' / 'main.css'
        css_to = out_dir / 'main.css'
        shutil.copy(str(css_from), str(css_to))

class Report(object):
    def __init__(self, report_title:str, texts:List[Text]):
        self.stats:DocStat = DocStat()
        self.builder:HtmlBuilder = HtmlBuilder(report_title)

        self.stats.analyze(texts)

    def report(self, out_dir:PathLike):
        total = 7
        with tqdm(total=total, desc='Reporting...', leave=True) as progress:
            def update_progress(text:str):
                progress.update(1)
                progress.set_description(text)

            # 1. prepare directory
            update_progress('Reporting: Prepare Directory...')
            out_dir:Path = Path(out_dir)
            _img_dir = Path('images')
            _html_dir = Path('html')
            img_dir = out_dir / _img_dir 
            html_dir = out_dir / _html_dir
            img_dir.mkdir(parents=True, exist_ok=True)
            html_dir.mkdir(parents=True, exist_ok=True)

            # Topic Model
            # 2. intertopic Distance Map
            update_progress('Reporting: Topic Model - Intertopic Distance Map')
            fig = self.stats.topic_model.visualize_topics(width=800, height=800)
            with open(html_dir / 'intertopic_distance_map.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_iframe_section(
                title='Intertopic Distance Map',
                description='',
                html_path=str(_html_dir / 'intertopic_distance_map.html'),
                width=fig.layout.width+50, height=fig.layout.height+50
            )

            # 3. Hierarchical Clustering
            update_progress('Reporting: Topic Model - Hierarchical Clustering')
            fig = self.stats.topic_model.visualize_hierarchy()
            with open(html_dir / 'hierarchical_clustering.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_iframe_section(
                title='Hierarchical Clustering',
                description='',
                html_path=str(_html_dir / 'hierarchical_clustering.html'),
                width=fig.layout.width+50, height=fig.layout.height+50
            )

            # 4. BarChart
            update_progress('Reporting: Topic Model - BarChart')
            fig = self.stats.topic_model.visualize_barchart(top_n_topics=len(self.stats.topic_model.topics), n_words=8, width=300)
            with open(html_dir / 'barchart.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_iframe_section(
                title='Topic Word Score',
                description='',
                html_path=str(_html_dir / 'barchart.html'),
                width=fig.layout.width+50, height=fig.layout.height+50
            )

            # 5. Similarity Matrix
            update_progress('Reporting: Topic Model - Similarity Matrix')
            fig = self.stats.topic_model.visualize_heatmap()
            with open(html_dir / 'similarity_matrix.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_iframe_section(
                title='Similarity Matrix',
                description='',
                html_path=str(_html_dir / 'similarity_matrix.html'),
                width=fig.layout.width+50, height=fig.layout.height+50
            )

            # 6. Topics Per Cpass
            update_progress('Reporting: Topic Model - Topics per Class')
            fig = self.stats.topic_model.visualize_topics_per_class(self.stats.topic_model_attrs['topics_per_class'])
            with open(html_dir / 'topics_per_class.html', mode='wt', encoding='utf-8') as wf:
                wf.write(fig.to_html())
            self.builder.add_iframe_section(
                title='Topics per Class',
                description='',
                html_path=str(_html_dir / 'topics_per_class.html'),
                width=fig.layout.width+50, height=fig.layout.height+50
            )

            # 7. Topic Probability Distribution
            update_progress('Reporting: Topic Model - Topic Probability Distribution')
            prob_dist_dir = img_dir / 'topic_model_prob_dist'
            prob_dist_dir.mkdir(parents=True, exist_ok=True)
            options = []
            for idx, text in enumerate(tqdm(self.stats.topic_model_attrs['texts'], desc='Reporting Topic Prob Dist...', leave=False)):
                fig = self.stats.topic_model.visualize_distribution(text.prob, min_probability=0.001)
                path = prob_dist_dir / '{idx:08d}.html'
                with open(path, mode='wt', encoding='utf-8') as wf:
                    wf.write(fig.to_html())
                options.append(OptionObject(path, text.title))

            self.builder.add_select_section(
                title='Topic Probability Distribution',
                description='',
                options=options,
            )

        # save index.html
        self.builder.save(out_dir)