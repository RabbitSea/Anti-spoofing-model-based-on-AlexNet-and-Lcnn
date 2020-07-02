import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.all_layer = nn.Sequential(
            # 1
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 3), padding=2),  # input，output, kernel   # 32
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.BatchNorm2d(16, affine=True),  # 16channels
            nn.Dropout2d(p=0.5),
            # 2
            nn.Conv2d(16, 32, kernel_size=1),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.Conv2d(16, 48, kernel_size=3),   # 48
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 24
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(24, affine=True),
            nn.Dropout2d(p=0.5),
            # 3
            nn.Conv2d(24, 48, kernel_size=1),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.Conv2d(24, 64, kernel_size=3),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 32
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # 32
            nn.BatchNorm2d(32, affine=True),
            nn.Dropout2d(p=0.5),
            # 4
            nn.Conv2d(32, 64, kernel_size=1),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.BatchNorm2d(16, affine=True),
            # 5
            nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # 16
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(8, affine=True)
        )

        nn.init.xavier_normal_(self.all_layer[0].weight)

        nn.init.xavier_normal_(self.all_layer[5].weight)
        nn.init.xavier_normal_(self.all_layer[7].weight)

        nn.init.xavier_normal_(self.all_layer[12].weight)
        nn.init.xavier_normal_(self.all_layer[14].weight)

        nn.init.xavier_normal_(self.all_layer[19].weight)
        nn.init.xavier_normal_(self.all_layer[21].weight)

        nn.init.xavier_normal_(self.all_layer[25].weight)
        nn.init.xavier_normal_(self.all_layer[27].weight)

        self.fc1 = nn.Linear(96, 2)

    def forward(self, x):

        x = x.to(torch.float32)  # 只针对cqt数据集
        x = self.all_layer(x)
        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        # X = F.dropout(x, training=self.training, p=0.7)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.mfm = torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=(1, 3))  # input，output, kernel   # 32
        nn.init.xavier_normal_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(16, affine=True)  # 16channels
        # 2
        self.conv2a = nn.Conv2d(16, 32, kernel_size=1)
        nn.init.xavier_normal_(self.conv2a.weight)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn2 = nn.BatchNorm2d(24, affine=True)
        # 3
        self.conv3a = nn.Conv2d(24, 48, kernel_size=1)
        nn.init.xavier_normal_(self.conv3a.weight)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3)
        nn.init.xavier_normal_(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 2))  # 32
        self.bn3 = nn.BatchNorm2d(32, affine=True)
        # 4
        self.conv4a = nn.Conv2d(32, 64, kernel_size=1)
        nn.init.xavier_normal_(self.conv4a.weight)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        nn.init.xavier_normal_(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn4 = nn.BatchNorm2d(16, affine=True)
        # 5
        self.conv5a = nn.Conv2d(16, 32, kernel_size=3)
        nn.init.xavier_normal_(self.conv5a.weight)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)
        nn.init.xavier_normal_(self.conv5.weight)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn5 = nn.BatchNorm2d(8, affine=True)

        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.bn1(self.pool1(self.mfm(self.conv1(x))))   # 64 * 67
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.mfm(self.conv2a(x))   # 64*67
        x = self.bn2(self.pool2(self.mfm(self.conv2(x)))) # 31*32
        x = F.dropout(x, p=0.5, training=self.training) # 31*32

        x = self.mfm(self.conv3a(x))  #　31*32
        x = self.bn3(self.pool3(self.mfm(self.conv3(x))))  # １４*　１５
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.mfm(self.conv4a(x))     # 14*15
        x = self.bn4(self.pool4(self.mfm(self.conv4(x))))  # 11*12
        # x = F.dropout(x, p=0.7, training=self.training)

        x = self.mfm(self.conv5a(x))  # 9*10
        x = self.bn5(self.pool5(self.mfm(self.conv5(x)))) # 3*4

        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        # X = F.dropout(x, training=self.training, p=0.7)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class Net3(nn.Module):  # fft的模型
    def __init__(self):
        super(Net3, self).__init__()
        self.all_layer = nn.Sequential(  # input  128*400
        nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 3)),  # 124*132
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 122*65
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 2
        nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1)),  # 122*65
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1)),  # 120*63
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 60*31
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 3
        nn.Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1)),  # 60*31
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1)),  # 58*29
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 56*14
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 4
        nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),  # 56*14
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),  # 54*12
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 27*6
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # 5
        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)),  # 25*4
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1)),  # 23 * 2
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 11 * 1
        nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        nn.init.xavier_normal_(self.all_layer[0].weight)

        nn.init.xavier_normal_(self.all_layer[5].weight)
        nn.init.xavier_normal_(self.all_layer[7].weight)

        nn.init.xavier_normal_(self.all_layer[12].weight)
        nn.init.xavier_normal_(self.all_layer[14].weight)

        nn.init.xavier_normal_(self.all_layer[19].weight)
        nn.init.xavier_normal_(self.all_layer[21].weight)

        nn.init.xavier_normal_(self.all_layer[25].weight)
        nn.init.xavier_normal_(self.all_layer[27].weight)

        self.fc1 = nn.Linear(88, 2)

    def forward(self, x):
        x = self.all_layer(x)
        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        # X = F.dropout(x, training=self.training, p=0.7)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class Net4(nn.Module):   # 比Net3多加了一个全连接隐藏层
    def __init__(self):
        super(Net4, self).__init__()
        self.all_layer = nn.Sequential(  # input  128*400
        nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 3)),  # 124*132
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 122*65
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 2
        nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1)),  # 122*65
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1)),  # 120*63
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 60*31
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 3
        nn.Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1)),  # 60*31
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1)),  # 58*29
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 56*14
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout2d(p=0.5, inplace=False),
        # 4
        nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),  # 56*14
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),  # 54*12
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 27*6
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # 5
        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)),  # 25*4
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1)),  # 23 * 2
        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 11 * 1
        nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        nn.init.xavier_normal_(self.all_layer[0].weight)

        nn.init.xavier_normal_(self.all_layer[5].weight)
        nn.init.xavier_normal_(self.all_layer[7].weight)

        nn.init.xavier_normal_(self.all_layer[12].weight)
        nn.init.xavier_normal_(self.all_layer[14].weight)

        nn.init.xavier_normal_(self.all_layer[19].weight)
        nn.init.xavier_normal_(self.all_layer[21].weight)

        nn.init.xavier_normal_(self.all_layer[25].weight)
        nn.init.xavier_normal_(self.all_layer[27].weight)

        self.fc1 = nn.Linear(88, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.to(torch.float32)  # 只针对cqt数据集
        x = self.all_layer(x)
        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        # X = F.dropout(x, training=self.training, p=0.7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class Net_cqt(nn.Module):
    def __init__(self):
        super(Net_cqt, self).__init__()
        self.all_layer = nn.Sequential(  # input  128*400
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 3)),  # 124*132
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 122*65
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.5, inplace=False),
            # 2
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1)),  # 122*65
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1)),  # 120*63
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 60*31
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.5, inplace=False),
            # 3
            nn.Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1)),  # 60*31
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1)),  # 58*29
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0, dilation=1, ceil_mode=False),  # 56*14
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.5, inplace=False),
            # 4
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),  # 56*14
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),  # 54*12
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 27*6
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # 5
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)),  # 25*4
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1)),  # 23 * 2
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False),  # 11 * 1
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        nn.init.xavier_normal_(self.all_layer[0].weight)

        nn.init.xavier_normal_(self.all_layer[5].weight)
        nn.init.xavier_normal_(self.all_layer[7].weight)

        nn.init.xavier_normal_(self.all_layer[12].weight)
        nn.init.xavier_normal_(self.all_layer[14].weight)

        nn.init.xavier_normal_(self.all_layer[19].weight)
        nn.init.xavier_normal_(self.all_layer[21].weight)

        nn.init.xavier_normal_(self.all_layer[25].weight)
        nn.init.xavier_normal_(self.all_layer[27].weight)

        self.fc1 = nn.Linear(88, 2)

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.double)
        x = x.to(torch.float32)  # 只针对cqt数据集
        x = self.all_layer(x)
        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        # X = F.dropout(x, training=self.training, p=0.7)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
