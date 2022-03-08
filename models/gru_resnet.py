from typing import Tuple, Callable
import torch
import torch.nn as nn

from utils.utils import Config
from models.base import BaseModel

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.batch_norm_1 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.linear_1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_layers)])
        self.batch_norm_2 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.linear_2 = nn.ModuleList([nn.Linear(hidden_dim, input_dim) for _ in range(n_layers)])

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in range(self.n_layers):
            x = self.batch_norm_1[layer](x)
            x = self.activation(x)
            x = self.linear_1[layer](x)
            x = self.batch_norm_2[layer](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_2[layer](x)
        return x

class GRU_Resnet(BaseModel):
    '''GRU with n Resnet layer
    
    Input:
        (batch_size, sentence_length, embedding_dim)
    
    Output:
        (batch_size, n_class)

    Args:
        config (Config): instance of Config
        embedding_dim (int): embedding dim
        hidden_dim (int): hidden dim of the GRU
        n_resnet_layers (int): number of ResNet layers following GRU layer
        n_class (int): number of output class
        name (str): name of the model
    '''
    def __init__(self, config:Config, embedding_dim:int, hidden_dim:int, n_resnet_layers:int=1, n_class:int=1, name:str='dnn-l1'):
        super().__init__(config, name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_resnet_layers = n_resnet_layers
        self.n_class = n_class 
        self.build()

    def build(self):
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True)

        self.linear_0 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch_norm_0 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.resblocks = ResidualBlock(self.hidden_dim // 2, self.hidden_dim // 4, n_layers=self.n_resnet_layers, dropout=0.1)
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
        out = self.resblocks(out)
        out = self.output(out)
        if self.n_class < 2:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=-1)
        return out
