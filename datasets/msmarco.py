import json
import tarfile
from collections import namedtuple
from enum import Enum
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from embeddings.base import Embedding
from sklearn.model_selection import train_test_split
from utils.google_drive import GDriveObjects, download_from_google_drive
from utils.utils import Config, Phase, download, is_notebook

from datasets.base import BaseDataset

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

Passage = namedtuple('Passage', ('is_selected', 'passage_text'))
RougeL = namedtuple('RougeL', ('start_pos', 'end_pos', 'passage_span', 'answer', 'rouge_l'))

MsmarcoItemX = namedtuple('MsmarcoItemX', ('query_id', 'query_type', 'query_word_tokens', 'query_char_tokens',
                                           'passage_word_tokens', 'passage_char_tokens', 'passage_is_selected', 'answer_word_tokens', 'answer_char_tokens'))
MsmarcoItemPos = namedtuple('MsmarcoItemPos', ('start_pos', 'end_pos'))


class MsmarcoDatasetType(Enum):
    TRAIN = 'train'
    DEV = 'dev'


class MsmarcoRecord(object):

    def __init__(self, idx: int, item: dict, rouge_l: dict):
        '''
        Args:
            idx: int
            item:
                {
                    'answers': [ANSWER],
                    'passages': [{'is_selected': IS_SELECTED, 'passage_text': PASSAGE}],
                    'query': QUERY,
                    'query_id': QUERY_ID,
                    'query_type': QUERY_TYPE,
                    'wellFormedAnswers': WELL_FORMED_ANSWERS
                }
            rouge_l:
                {
                    'start_pos': START POS,
                    'end_pos': END POS,
                    'passage_span': PASSAGE SPAN,
                    'answer': ANSWER,
                    'rouge_l': ROUGE-L
                }
        '''
        self.index = idx
        self.query_id: int = int(item['query_id'])
        self.query_type: str = item['query_type']
        self.query: str = item['query']
        self.passages: List[Passage] = [Passage(p['is_selected'], p['passage_text']) for p in item['passages']]
        self.answers: List[str] = [answer for answer in item['answers']]
        self.rouge_l = RougeL(int(rouge_l['start_pos']), int(rouge_l['end_pos']), str(rouge_l['passage_span']), str(rouge_l['answer']), float(rouge_l['rouge_l']))

    def __str__(self):
        return f'<Record query_id:{self.query_id} query:{self.query[:10]}...>'

    def __repr__(self):
        return self.__str__()

    def query2tokens(self, embedding: Embedding) -> Tuple[np.ndarray, List[np.ndarray]]:
        word_embed = embedding.word_embed(self.query)
        char_embed = embedding.char_embed(self.query)
        return word_embed, char_embed

    def answers2tokens(self, embedding: Embedding) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        word_tokens = []
        char_tokens = []
        for answer in self.answers:
            # word tokens
            word_embed = embedding.word_embed(answer)
            word_tokens.append(word_embed)

            # char tokens
            char_embed = embedding.char_embed(answer)
            char_tokens.append(char_embed)

        return word_tokens, char_tokens

    def passages2tokens(self, embedding: Embedding) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[bool]]:
        word_tokens = []
        char_tokens = []
        is_selected = []
        for passage in self.passages:
            # word tokens
            word_embed = embedding.word_embed(passage.passage_text)
            word_tokens.append(word_embed)

            # char tokens
            char_embed = embedding.char_embed(passage.passage_text)
            char_tokens.append(char_embed)

            # is_selected
            is_selected.append(passage.is_selected)

        return word_tokens, char_tokens, is_selected

    def to_data(self, embedding: Embedding) -> Tuple[MsmarcoItemX, MsmarcoItemPos]:
        qwt, qct = self.query2tokens(embedding)
        pwt, pct, psl = self.passages2tokens(embedding)
        awt, act = self.answers2tokens(embedding)
        x = MsmarcoItemX(torch.LongTensor(self.query_id),                                                       # query_id
                         self.query_type,                                                                       # query_type
                         torch.tensor(qwt, dtype=torch.float32),                                                # query_word_tokens
                         [torch.tensor(chars, dtype=torch.float32) for chars in qct],                           # query_char_tokens
                         [torch.tensor(passage, dtype=torch.float32) for passage in pwt],                       # passage_word_tokens
                         [[torch.tensor(chars, dtype=torch.float32) for chars in passage] for passage in pct],  # passage_char_tokens
                         torch.tensor(psl, dtype=torch.float32),                                                # passage_is_selected
                         [torch.tensor(answer, dtype=torch.float32) for answer in awt],                         # answer_word_tokens
                         [[torch.tensor(chars, dtype=torch.float32) for chars in answer] for answer in act])    # answer_char_tokens
        y = MsmarcoItemPos(torch.LongTensor([self.rouge_l.start_pos])[0],                                       # start_pos
                           torch.LongTensor([self.rouge_l.end_pos])[0])                                         # end_pos
        return x, y


MsmarcoDs = Dict[int, MsmarcoRecord]


class MsmarcoDataset(BaseDataset):
    def __init__(self, config: Config, ds_type: MsmarcoDatasetType, embedding: Embedding, rouge_threshold=0.7):
        super().__init__(config, Phase.TRAIN)
        self.embedding: Embedding = embedding
        self.ds_type: MsmarcoDatasetType = ds_type
        self.rouge_threshold = rouge_threshold

        self.train_data, self.valid_data, self.test_data = self.__load_data__(self.dataset_path)
        # self.valid_data = self.train_data
        # self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path: Path) -> Tuple[MsmarcoDs, MsmarcoDs, MsmarcoDs]:
        '''load imdb corpus dataset
        download tar.gz file if dataset does not exist

        Args:
            dataset_path (Path): path to dataset

        Returns:
            train_data, valid_data, test_data: dict[idx, ImdbItem]
        '''

        # download dataset if it does not exist
        (dataset_path / 'msmarco').mkdir(parents=True, exist_ok=True)
        if not (dataset_path / 'msmarco' / 'train_v2.1.json').exists():
            self.config.log.dataset_log.info('download MSMARCO train_v2.1.json from Google Drive')
            download_from_google_drive(GDriveObjects.MSMARCO_TRAIN_DATA.value, str(dataset_path / 'msmarco' / 'train_v2.1.json'))
        if not (dataset_path / 'msmarco' / 'train_v2.1.rouge.json').exists():
            self.config.log.dataset_log.info('download MSMARCO train_v2.1.rouge.json from Google Drive')
            download_from_google_drive(GDriveObjects.MSMARCO_TRAIN_ROUGE.value, str(dataset_path / 'msmarco' / 'train_v2.1.rouge.json'))
        if not (dataset_path / 'msmarco' / 'dev_v2.1.json').exists():
            self.config.log.dataset_log.info('download MSMARCO dev_v2.1.json from Google Drive')
            download_from_google_drive(GDriveObjects.MSMARCO_DEV_DATA.value, str(dataset_path / 'msmarco' / 'dev_v2.1.json'))
        if not (dataset_path / 'msmarco' / 'dev_v2.1.rouge.json').exists():
            self.config.log.dataset_log.info('download MSMARCO dev_v2.1.rouge.json from Google Drive')
            download_from_google_drive(GDriveObjects.MSMARCO_DEV_ROUGE.value, str(dataset_path / 'msmarco' / 'dev_v2.1.rouge.json'))

        if self.ds_type == MsmarcoDatasetType.TRAIN:
            json_data = json.load(open(dataset_path / 'msmarco' / 'train_v2.1.json'))
            rouge_data = {int(key): value for key, value in json.load(open(dataset_path / 'msmarco' / 'train_v2.1.rouge.json')).items()}
        elif self.ds_type == MsmarcoDatasetType.DEV:
            json_data = json.load(open(dataset_path / 'msmarco' / 'dev_v2.1.json'))
            rouge_data = {int(key): value for key, value in json.load(open(dataset_path / 'msmarco' / 'dev_v2.1.rouge.json')).items()}

        keys = list(json_data['query_id'].keys())

        # load rouge_l value
        target_qids = set([r['query_id'] for r in rouge_data.values() if r['rouge_l'] >= self.rouge_threshold])
        self.config.log.dataset_log.info(f'filter with rouge_l={self.rouge_threshold} -> total: {len(rouge_data)} target: {len(target_qids)}')

        # load msmarco
        data_all: list[MsmarcoRecord] = []
        for idx, key in tqdm(enumerate(keys), desc='loading data...', total=len(keys)):

            loaded_data = {
                'answers': json_data['answers'][key],
                'passages': json_data['passages'][key],
                'query': json_data['query'][key],
                'query_id': json_data['query_id'][key],
                'query_type': json_data['query_type'][key],
                'wellFormedAnswers': json_data['wellFormedAnswers'][key]
            }

            rouge_l = rouge_data[int(json_data['query_id'][key])]
            record = MsmarcoRecord(idx, loaded_data, rouge_l)

            if int(json_data['query_id'][key]) in target_qids:
                data_all.append(record)

        # split data
        train_data, test_data = train_test_split(data_all, test_size=self.test_size)
        train_data, valid_data = train_test_split(train_data, test_size=self.valid_size)

        # list -> dict
        train_dict = {idx: d for idx, d in enumerate(train_data)}
        valid_dict = {idx: d for idx, d in enumerate(valid_data)}
        test_dict = {idx: d for idx, d in enumerate(test_data)}

        return train_dict, valid_dict, test_dict

    def __load_text__(self, path: PathLike) -> str:
        text = open(path, 'rt').read().strip()
        return text

    def __getitem__(self, index):
        if self.phase == Phase.TRAIN or self.phase == Phase.DEV:
            record: MsmarcoRecord = self.train_data[index]
        elif self.phase == Phase.VALID:
            record: MsmarcoRecord = self.valid_data[index]
        elif self.phase == Phase.TEST or self.phase == Phase.SUBMISSION:
            record: MsmarcoRecord = self.test_data[index]
        else:
            raise RuntimeError('Unknown Phase')

        return record.to_data(self.embedding)

    @staticmethod
    def collate_fn(batch):
        import pdb
        pdb.set_trace()
