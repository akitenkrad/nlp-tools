import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_channels: int) -> None:
        """the Encoder of simple Autoencoder architecture

        Input:
            (batch_size, 1, width, height)
        Output:
            (batch_size, 2)
        """
        super().__init__()

        self.conv_0 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.batch_norm_0 = nn.BatchNorm2d(32)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(3136, 2)
        self.relu = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv_0.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        out = self.relu(self.batch_norm_0(self.conv_0(x)))
        out = self.relu(self.batch_norm_1(self.conv_1(out)))
        out = self.relu(self.batch_norm_1(self.conv_2(out)))
        out = self.relu(self.batch_norm_1(self.conv_3(out)))
        out = out.reshape(-1, 3136)
        out = self.linear(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_channels: int) -> None:
        """

        Input:
            (batch_size, 2)
        Output:
            (batch_size, 1, 28, 28)
        """
        super().__init__()

        self.linear = nn.Linear(2, 3136)
        self.conv_t_0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.conv_t_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv_t_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv_t_3 = nn.ConvTranspose2d(
            32, n_channels, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0
        )
        self.batch_norm_0 = nn.BatchNorm2d(64)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_t_0.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_t_1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_t_2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv_t_3.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(-1, 64, 7, 7)
        out = self.relu(self.batch_norm_0(self.conv_t_0(out)))
        out = self.relu(self.batch_norm_0(self.conv_t_1(out)))
        out = self.relu(self.batch_norm_1(self.conv_t_2(out)))
        out = torch.sigmoid(self.conv_t_3(out))
        return out


class Autoencoder(nn.Module):
    def __init__(self, n_channels: int) -> None:
        """
        Input:
            (batch_size, 1, 28, 28)
        Output:
            (batch_size, 1, 28, 28)
        """
        super().__init__()
        self.encoder = Encoder(n_channels=n_channels)
        self.decoder = Decoder(n_channels=n_channels)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
