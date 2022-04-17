from typing import List, Dict, Tuple

import numpy as np
from bertopic import BERTopic
from utils.tokenizers import WordTokenizer
from utils.utils import Lang, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    import tqdm


class Text(object):
    def __init__(
        self,
        title: str,
        summary: str,
        keywords: List[str],
        pdf_url: str,
        authors: List[str],
        **kwargs,
    ):
        self.title: str = title
        self.summary: str = summary
        self.keywords: List[str] = keywords
        self.pdf_url: str = pdf_url
        self.authors: List[str] = authors
        self.topic = -99
        self.prob = np.array([])
        for name, value in kwargs.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def __str__(self):
        return f'<Text {self.title[:10]}...>'

    def __repr__(self):
        return self.__str__()


class DocStat(object):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.topic_model: BERTopic = BERTopic(calculate_probabilities=True)
        self.topic_model_attrs: dict = {}
        self.tokenizer = WordTokenizer(
            language=Lang.ENGLISH,
            remove_stopwords=True,
            remove_punctuations=True,
            stemming=True,
            add_tag=True,
        )

    def __topic_model_preprocess(self, texts: List[str]) -> List[str]:
        '''extract only Noun words'''
        words = [self.tokenizer.tokenize(text) for text in tqdm(texts)]
        texts = [
            ' '.join([word[0] for word in word_list if word[1].startswith('N')])
            for word_list in words
        ]
        return texts

    @property
    def topics(self) -> dict:
        names: Dict[int, str] = self.topic_model.topic_names
        sizes: Dict[int, int] = self.topic_model.topic_sizes
        all_topics: Dict[int, List[Tuple[str, float]]] = self.topic_model.get_topics()
        res = {index: {'name': name, 'size': size, 'top_n_words': n_words} for (index, name), size, n_words in zip(names.items(), sizes.values(), all_topics.values())}
        return res

    def analyze(self, texts: List[Text]):
        with tqdm(total=2, desc='analyzing...', leave=False) as progress:

            def update_progress(desc_text: str):
                progress.update(1)
                progress.set_description(desc_text)

            # Basic analysis
            update_progress('Basic Analysis...')

            # Topic Modeling
            update_progress('Topic Modeling...')
            texts_for_tp = [text.title + '\n' + text.summary for text in texts]
            texts_for_tp = self.__topic_model_preprocess(texts_for_tp)
            topics, probs = self.topic_model.fit_transform(texts_for_tp)
            self.topic_model_attrs['topics'] = topics
            self.topic_model_attrs['probs'] = probs

            # Topics per Class
            keywords = [text.keywords[0] for text in texts]
            topics_per_class = self.topic_model.topics_per_class(
                texts_for_tp, topics, classes=keywords
            )
            self.topic_model_attrs['topics_per_class'] = topics_per_class

            for topic, text, prob in zip(topics, texts, probs):
                text.topic = topic
                text.prob = prob
            self.topic_model_attrs['texts'] = texts

    def get_topic(self, text: Text) -> Dict[int, float]:
        topics = {idx: 0.0 for idx in self.topics.keys()}
        _topics, _probs = self.topic_model.find_topics(text.title + '\n' + text.summary)
        for topic, prob in zip(_topics, _probs):
            topics[topic] = prob
        return topics
