import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet40(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb


class ConvNet40DDC(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40DDC, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 3 * 3)
        x1 = F.relu(self.fc1(x1))
        x1 = self.dropout(x1)
        emb1 = x1

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = x2.view(-1, 64 * 3 * 3)
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        emb2 = x2

        out = self.fc2(x1)
        return out, emb1, emb2


class ConvNet40DDCV2(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40DDCV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 3 * 3)
        x1 = F.relu(self.fc1(x1))
        x1 = self.dropout(x1)
        emb1 = x1

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = x2.view(-1, 64 * 3 * 3)
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        emb2 = x2

        out1 = self.fc2(x1)
        out2 = self.fc2(x2)
        return out1, emb1, out2, emb2


class ConvNet40Double(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40Double, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 3 * 3)
        emb1_1 = x1
        x1 = F.relu(self.fc1(x1))
        x1 = self.dropout(x1)
        emb1_2 = x1

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = x2.view(-1, 64 * 3 * 3)
        emb2_1 = x2
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        emb2_2 = x2

        out = self.fc2(x1)
        return out, emb1_1, emb1_2, emb2_1, emb2_2


class ConvNet50(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb


class ConvNet32(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet32, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 2 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb


class ConvTradNet(nn.Module):
    def __init__(self, classes=3):
        super(ConvTradNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (20, 8))
        self.pool = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(64, 64, (10, 4))
        self.fc1 = nn.Linear(64 * 4 * 8, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = x.view(-1, 64 * 4 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb


class ConvNetLarge(nn.Module):
    def __init__(self, classes):
        super(ConvNetLarge, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear_input = 128 * 3 * 3
        self.fc1 = nn.Linear(self.linear_input, 256)
        self.fc2 = nn.Linear(256, classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.linear_input)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, x


def test():
    data = torch.rand(128, 1, 40, 40)
    net = ConvNet40(4)
    result, _ = net(data)
    print(result.shape)
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    print("param:", count)


if __name__ == "__main__":
    test()
