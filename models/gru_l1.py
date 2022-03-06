from typing import Tuple, Callable
import torch
import torch.nn as nn

from utils.utils import Config
from models.base import BaseModel

class GRU_L1(BaseModel):
    '''GRU with 1 layer
    
    Input:
        (batch_size, sentence_length, embedding_dim)
    
    Output:
        (batch_size, n_class)
    '''
    def __init__(self, config:Config, embedding_dim:int, hidden_dim:int, n_class:int, name:str='dnn-l1'):
        super().__init__(config, name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class 
        self.build()

    def build(self):
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1)
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.output = nn.Linear(self.hidden_dim // 2, self.n_class)
        self.dropout = nn.Dropout(0.2)

    def step(self, x: torch.Tensor, y: torch.Tensor, loss_func: Callable) -> Tuple[float, torch.Tensor]:
        x = x.type(torch.float32).to(self.config.train.device)
        y = y.type(torch.long).to(self.config.train.device)
        out = self(x).squeeze()
        loss = loss_func(out, y)
        return loss, out

    def step_wo_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.float32).to(self.config.train.device)
        y = y.type(torch.long).to(self.config.train.device)
        out = self(x).squeeze()
        return out

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = hidden.reshape(-1, self.hidden_dim)
        out = self.linear_1(hidden)
        out = self.batch_norm_1(out)
        out = self.dropout(out)
        out = self.output(out)
        if self.n_class < 2:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=-1)
        return out
