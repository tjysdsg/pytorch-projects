import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TDNN']


class AvgPoolStd(nn.Module):
    def __init__(self):
        super(AvgPoolStd, self).__init__()

    def forward(self, x):
        x_mean = x.mean(dim=2)
        x_std = x.std(dim=2)
        out = torch.cat([x_mean, x_std], dim=1)
        return out


class TDNN(nn.Module):
    def __init__(self, in_planes=64, n_classes=2, embedding_size=512):
        super(TDNN, self).__init__()
        # Extractor
        self.conv1 = nn.Conv1d(in_planes, 512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 1536, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(1536)
        self.conv3 = nn.Conv1d(1536, 512, kernel_size=3, dilation=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(1500)
        # Encoder
        self.pool = AvgPoolStd()
        # Embedding
        self.fc1 = nn.Linear(3000, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        embd = self.fc1(x)
        out = self.fc2(embd)
        out = self.fc3(out)

        return out, embd


if __name__ == '__main__':
    net = TDNN(64, n_classes=2)
    a = torch.rand(128, 1, 99, 64)
    b, c = net(a)
    print(b.shape)
