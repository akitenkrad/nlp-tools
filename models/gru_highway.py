from typing import Tuple, Callable
import torch
import torch.nn as nn

from utils.utils import Config
from models.base import BaseModel

class HighwayBlock(nn.Module):
    def __init__(self, input_dim:int, n_layers:int, activation=nn.LeakyReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.batch_norm_1 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.batch_norm_2 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.2)
        self.activation = activation

        for layer in range(n_layers):
            gate = self.gate[layer]
            nn.init.constant_(gate.bias, -1.0)

    def forward(self, x):
        for layer in range(self.n_layers):
            x = self.batch_norm_1[layer](x)
            x = self.activation(x)

            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = self.activation(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear

            x = self.batch_norm_2[layer](x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

class GRU_Highway(BaseModel):
    '''GRU with n Highway layer
    
    Input:
        (batch_size, sentence_length, embedding_dim)
    
    Output:
        (batch_size, n_class)

    Args:
        config (Config): instance of Config
        embedding_dim (int): embedding dim
        hidden_dim (int): hidden dim of the GRU
        n (int): number of Highway layers following GRU layer
        n_class (int): number of output class
        name (str): name of the model
    '''
    def __init__(self, config:Config, embedding_dim:int, hidden_dim:int, n_highway_layers:int=1, n_class:int=1, name:str='dnn-l1'):
        super().__init__(config, name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_highway_layers = n_highway_layers
        self.n_class = n_class 
        self.build()

    def build(self):
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True)

        self.linear_0 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch_norm_0 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.highway = HighwayBlock(self.hidden_dim // 2, self.n_highway_layers, nn.LeakyReLU())
        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.relu = nn.LeakyReLU()
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
        out = self.highway(out)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.output(out)
        out = self.dropout(out)
        if self.n_class < 2:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=-1)
        return out
