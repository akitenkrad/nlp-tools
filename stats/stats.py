from typing import List
from collections import OrderedDict
from bertopic import BERTopic

from utils.utils import Lang, is_notebook
from utils.tokenizers import WordTokenizer

if is_notebook():
    from tqdm.notebook import tqdm
else:
    import tqdm

class Text(object):
    def __init__(self, title:str, summary:str, **kwargs):
        self.title = title
        self.summary = summary
        for name, value in kwargs.items():
            setattr(self, name, value)

class DocStat(object):
    def __init__(self):
        self.topic_model = BERTopic(calculate_probabilities=True)
        self.topic_model_attrs:dict = {}
        self.tokenizer = WordTokenizer(language=Lang.ENGLISH, remove_stopwords=True, remove_punctuations=True, stemming=True, add_tag=True)

    def __topic_model_preprocess(self, texts:List[str]) -> List[str]:
        '''extract only Noun words'''
        words = [tokenizer.tokenize(text) for text in tqdm(texts)]
        texts = [' '.join([word[0] for word in word_list if word[1].startswith('N')]) for word_list in words]
        return texts

    def analyze(self, texts:List[Text], keywords:List[str]):
        with tqdm(total=2, desc='analyzing...', leave=False) as progress:
            def update_progress(desc_text:str):
                progress.update(1)
                progress.set_description(desc_text)

            # Basic analysis
            update_progress('Basic Analysis...')

            # Topic Modeling
            update_progress('Topic Modeling...')
            texts_for_tp = [text.title + '\n' + text.summary for text in texts]
            texts_for_tp = self.__topic_model_preprocess(texts)
            topics, probs = self.topic_model.fit_transform(texts_for_tp)
            self.topic_model_attrs['topics'] = topics
            self.topic_model_attrs['probs'] = probs

            # Topics per Class
            keywords = [p['keywords'][0] for p in data]
            topics_per_class = topic_model.topics_per_class(texts, topics, classes=keywords)
            self.topic_model_attrs['topics_per_class'] = topics_per_class

            self.topic_model_attrs['topic_docs'] = OrderedDict()
            for idx, name in self.topic_model.topic_names.items():
                self.topic_model_attrs['topic_docs'][idx] = {'name': name}
            for topic, text, prob in zip(topics, texts, probs):
                self.topic_model_attrs['topic_docs'][topic]['title'] = text.title
                self.topic_model_attrs['topic_docs'][topic]['summary'] = text.summary
                self.topic_model_attrs['topic_docs'][topic]['prob'] = prob

