import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Generative Adversarial Network Discriminator"""

    def __init__(self, sent_len: int, embedding_dim: int, dropout=0.1):
        super().__init__()
        self.sent_len = sent_len
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.discriminator_conv_0 = nn.Conv2d(1, 64, 5, 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.02)
        self.dropout = nn.Dropout(dropout)
        self.discriminator_conv_1 = nn.Conv2d(64, 64, 5, 2)
        self.discriminator_conv_2 = nn.Conv2d(64, 128, 5, 2)
        self.discriminator_conv_3 = nn.Conv2d(128, 128, 5, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1, 1)  # TODO: Calculate in-feature, out-feature
