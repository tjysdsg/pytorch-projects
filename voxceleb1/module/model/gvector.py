import sys
import torch
import torch.nn as nn
from voxceleb1.module.model.resnet import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class Gvector(nn.Module):
    def __init__(self, channels=16, block='BasicBlock', num_blocks=[2, 2, 2, 2],
                 embd_dim=128, drop=0.5, n_class=1211, pooling='mean,std'):
        super(Gvector, self).__init__()
        block = str_to_class(block)
        self.resnet = ResNet(channels, block, num_blocks)
        self.pooling = pooling.split(',')
        self.fc1 = nn.Linear(channels * 8 * 2, embd_dim)
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(embd_dim, n_class)

    def extractor(self, x):
        # B * T * D
        x = x.unsqueeze(1)
        # B * 1 * T * D
        x = self.resnet(x)
        # B * C * H * W
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x_pooling = []
        if 'mean' in self.pooling:
            # x.mean(dim=2): B * C
            x_pooling.append(x.mean(dim=2))
        if 'std' in self.pooling:
            # x.std(dim=2): B * C
            x_pooling.append(x.std(dim=2))
        x = torch.cat(x_pooling, dim=1)
        # B * 2C
        x = self.fc1(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.extractor(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    import time

    # x.shape = [B, D, T]
    x = torch.zeros(128, 24, 100).cuda()
    model = Gvector(embd_dim=512, n_class=1211).cuda()
    tic = time.time()
    y = model(x)
    toc = time.time()
    print(y.shape, toc - tic)
