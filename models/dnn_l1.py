from typing import Tuple, Callable
import torch
import torch.nn as nn

from utils.utils import Config
from models.base import BaseModel

class DnnL1(BaseModel):
    '''DNN with 1 layer
    
    Input:
        (batch_size, sentence_length, embedding_dim)
    
    Output:
        (batch_size, n_class)
    '''
    def __init__(self, config:Config, embedding_dim:int, n_class:int, name:str='dnn-l1'):
        super().__init__(config, name)
        self.embedding_dim = embedding_dim
        self.n_class = n_class 
        self.build()

    def build(self):
        self.linear_1 = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim // 2)
        self.linear_2 = nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim // 4)
        self.output = nn.Linear(self.embedding_dim // 4, self.n_class)
        self.dropout = nn.Dropout(0.2)

    def step(self, x: torch.Tensor, y: torch.Tensor, loss_func: Callable) -> Tuple[float, torch.Tensor]:
        x = x.to(self.__device)
        y = y.to(self.__device)
        out = self(x)
        loss = loss_func(out)
        return loss, out

    def forward(self, x):
        out = self.linear_1(x)
        out = self.layer_norm_1(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.layer_norm_2(out)
        out = self.dropout(out)
        out = self.output(out)
        return out
