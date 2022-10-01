import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv_1 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.dense = nn.Linear(2048, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, 1, 28, 28)
        """
        out = self.dropout(self.relu(self.conv_0(x)))
        out = self.dropout(self.relu(self.conv_1(out)))
        out = self.dropout(self.relu(self.conv_2(out)))
        out = self.dropout(self.relu(self.conv_3(out)))
        out = out.reshape(-1, 2048)
        out = self.dense(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(100, 3136)
        self.batch_norm_0 = nn.BatchNorm1d(3136)
        self.batch_norm_1 = nn.BatchNorm2d(128)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.upsample_0 = nn.Upsample(size=(14, 14))
        self.upsample_1 = nn.Upsample(size=(28, 28))
        self.upsample_2 = nn.Upsample(size=(32, 32))
        self.conv_0 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2, dilation=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, 100)
        """
        out = self.relu(self.batch_norm_0(self.dense(x)))
        out = out.reshape(-1, 64, 7, 7)
        out = self.upsample_0(out)
        out = self.relu(self.batch_norm_1(self.conv_0(out)))
        out = self.upsample_1(out)
        out = self.relu(self.batch_norm_1(self.conv_1(out)))
        out = self.upsample_2(out)
        out = self.relu(self.batch_norm_2(self.conv_2(out)))
        out = self.relu(self.batch_norm_2(self.conv_3(out)))
        out = self.relu(self.conv_4(out))
        return out
