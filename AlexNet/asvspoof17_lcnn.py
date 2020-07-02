from __future__ import print_function
import argparse  # 命令行解析模块
import os
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import shutil
import asv17_util

from AlexNet.asvspoof17_My_Dataset import ASVspoofTestData, ASVspoofTrainData, ASVspoofDevData

# Training settings
parser = argparse.ArgumentParser(description='ConvNet reduction')  # 创建命令行解析的对象
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=24, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output-dir', type=str, default='./', metavar='N',    # 设置输出的目录
                    help='Output directory where models will be stored')
parser.add_argument('--train', type=bool, default=True,
                    help='Set/unset to toggle training')
parser.add_argument('--eval', type=bool, default=True,
                    help='Set/unset to toggle evaluation')

args = parser.parse_args()  # 参数实例，Namespace类型
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_output_dir = args.output_dir + '/'
try:
    if not os.path.isdir(model_output_dir):
        os.mkdir(model_output_dir)
except Exception as e:
    print("Error creating directory")
    print(e)

print('Parameters: ', args.__dict__)   # 输出args里面的所有参数


start_time = time.time()
print('Start time: {}'.format(start_time))

torch.manual_seed(args.seed)  # 为CPU生成种子用于生产随机数
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 如果gpu可用则创建字典

train_dataset = ASVspoofTrainData()
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
# batch_size=args.batch_size, shuffle=True, **kwargs)  # **kwargs传入不定长参数，传入的是变量kwargs

test_dataset = ASVspoofTestData()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
# batch_size=args.test_batch_size, shuffle=True, **kwargs)

dev_dataset = ASVspoofDevData()
devdata_loader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # input227*227*3
        nn.init.xavier_normal_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.mfm1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2a = nn.Conv2d(16, 32, kernel_size=1)
        nn.init.xavier_normal_(self.conv2a.weight)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3a = nn.Conv2d(24, 48, kernel_size=1)
        nn.init.xavier_normal_(self.conv3a.weight)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv4a = nn.Conv2d(32, 64, kernel_size=1)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv4a.weight)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv5a = nn.Conv2d(16, 32, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv5a.weight)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # 3x3 - 32
        nn.init.xavier_normal_(self.conv5.weight)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(128, 64)

        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):  # 实现模型前向传播的矩阵运算  backwrd 实现后向传播的自动梯度计算
        # try pooling after relu!
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.mfm1(self.conv2a(x))
        x = self.pool2(self.mfm1(self.conv2(x)))
        x = self.mfm1(self.conv3a(x))
        x = self.pool3(self.mfm1(self.conv3(x)))
        x = self.mfm1(self.conv4a(x))
        x = self.pool4(self.mfm1(self.conv4(x)))
        x = self.mfm1(self.conv5a(x))
        x = self.pool5(self.mfm1(self.conv5(x)))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv1(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv2a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv2(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv3a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv3(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv4a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv4(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv5a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv5(x), training=self.training)))

        x = x.view(x.size(0), -1)  # 扁平化处理，拉成一行

        x = F.dropout(x, p=0.7, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


model = Net()
print(model)
if args.cuda:
    model.cuda()


def get_optimizer(lr):
    return optim.Adam(model.parameters(), lr=lr)


optimizer = get_optimizer(args.lr)


# count the amount of parameters in the network
def measure_size():
    print('Conv1:', model.conv1.weight.size())
    print('Conv2:', model.conv2.weight.size())
    print('Conv3:', model.conv3.weight.size())
    print('Conv4:', model.conv3.weight.size())
    print('Conv5:', model.conv3.weight.size())
    print('Fc1:', model.fc1.weight.size())
    print('Fc2:', model.fc2.weight.size())
    return sum(p.numel() for p in model.parameters())


########################################################################################################################


prev_loss = 10.
curr_lr = args.lr


def train(epoch):
    model.train()
    total_loss = 0.0
    global optimizer
    global prev_loss
    for batch_idx, (data, target) in enumerate(train_loader):  # 迭代到train_loader里面取数据,随机取
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 梯度归零
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            process_rate = batch_idx * args.batch_size / len(train_loader)
            print('batch_idx = {}'.format(batch_idx))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                process_rate, loss.item()))

    print("Total loss = %.6f" % (total_loss / len(train_loader.dataset)))

    dev_correct = 0
    dev_loss = 0.
    model.eval()
    f = open('dev-eer', 'w')
    for data, devtarget in devdata_loader:
        # data = data.unsqueeze_(1)  # 在第1维（下标从0开始）增加一个维度
        with torch.no_grad():
            data, target = Variable(data), Variable(devtarget)
        output = model(data)
        output_arr = output.data.numpy().reshape((-1, 2))  # tensor to array, reshape(-1, 2)转换为两列
        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = output_arr[i, 0] - output_arr[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))  # genuine
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))  # spoof
        dev_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        dev_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    f.close()
    dev_loss /= len(devdata_loader.dataset)
    print("Dev loss is %.6f" % dev_loss)
    print('Dev set: Accuracy: {}/{} ({:.0f}%)\n'.format(dev_correct, len(devdata_loader.dataset),
                                                            100. * dev_correct / len(devdata_loader.dataset)))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    f = open('lcnn-eer-file', 'w')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data = data.unsqueeze_(1)
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        output_arr = output.data.numpy().reshape((-1, 2))
        tgt = target.data.numpy()
        for i in range(tgt.shape[0]):
            itgt = int(tgt[i])  # int_target
            scr = output_arr[i, 0] - output_arr[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))  # genuine
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))  # spoof
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    f.close()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


########################################################################################################################
# the training procedure
initial_size = measure_size()
print('initial_size: {}\n'.format(initial_size))

if args.train:
    for epoch in range(1, args.epochs+1):
        train(epoch)
        torch.save(model, model_output_dir + 'lcnn_iter' + str(epoch) + '.mdl')
        shutil.copy('dev-eer', model_output_dir + '/lcnn-dev-eer-' + str(epoch))

    torch.save(model, model_output_dir + 'lcnn_final.mdl')

if args.eval:
    test()

end_time = time.time()
process_time = end_time-start_time
print('End time: {}'.format(end_time))
print('Time cost: {}'.format(process_time))

print('EER of dev:')
cm_eer_dev = asv17_util.compute_eer17_from_file('dev-eer')
print('EER of eval:')
cm_eer_eval = asv17_util.compute_eer17_from_file('lcnn-eer-file')

