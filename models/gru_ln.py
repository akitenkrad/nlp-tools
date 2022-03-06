from typing import Tuple, Callable
import torch
import torch.nn as nn

from utils.utils import Config
from models.base import BaseModel

class GRU_Ln(BaseModel):
    '''GRU with n layer
    
    Input:
        (batch_size, sentence_length, embedding_dim)
    
    Output:
        (batch_size, n_class)
    '''
    def __init__(self, config:Config, embedding_dim:int, hidden_dim:int, n:int=1, n_class:int=1, name:str='dnn-l1'):
        super().__init__(config, name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n = n
        self.n_class = n_class 
        self.build()

    def build(self):
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True)

        self.linear_0 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch_norm_0 = nn.BatchNorm1d(self.hidden_dim // 2)

        for i in range(1, self.n + 1):
            setattr(self, f'linear_{i}', nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2))
            setattr(self, f'batch_norm_{i}', nn.BatchNorm1d(self.hidden_dim // 2))

        self.output = nn.Linear(self.hidden_dim // 2, self.n_class)
        self.dropout = nn.Dropout(0.2)

    def step(self, x: torch.Tensor, y: torch.Tensor, loss_func: Callable) -> Tuple[float, torch.Tensor]:
        x = x.type(torch.float32).to(self.config.train.device)
        y = y.type(torch.float32).to(self.config.train.device)
        out = self(x).squeeze()
        loss = loss_func(out, y)
        return loss, out

    def step_wo_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.float32).to(self.config.train.device)
        y = y.type(torch.float32).to(self.config.train.device)
        out = self(x).squeeze()
        return out

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = hidden.reshape(-1, self.hidden_dim)
        out = self.linear_0(hidden)
        out = self.batch_norm_0(out)

        for i in range(1, self.n + 1):
            l_layer = getattr(self, f'linear_{i}')
            b_layer = getattr(self, f'batch_norm_{i}')
            out = l_layer(out)
            out = b_layer(out)
            out = self.dropout(out)

        out = self.output(out)
        if self.n_class < 2:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=-1)
        return out
