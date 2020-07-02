from __future__ import print_function
import os
import torch
import torch.nn as nn
import argparse  # 命令行解析模块
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim import lr_scheduler

import shutil
import time
'''迁移alexnet, 参数调整'''

from AlexNet.asvspoof17_My_Dataset import ASVspoofTestData, ASVspoofTrainData, ASVspoofDevData
from AlexNet import asv17_util

# Training settings
parser = argparse.ArgumentParser(description='ConvNet reduction')  # 创建命令行解析的对象
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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

torch.manual_seed(args.seed)  # 为CPU生成种子用于生产随机数
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 如果gpu可用则创建字典

train_dataset = ASVspoofTrainData()
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_dataset = ASVspoofTestData()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

dev_dataset = ASVspoofDevData()
devdata_loader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

start_time = time.time()
print('Start time: {}'.format(start_time))

model = models.alexnet(pretrained=True)
for parma in model.parameters():
    # parma.requires_grad = False
    parma.requires_grad = True

# 6*6*256
model.classifier = torch.nn.Sequential(

    torch.nn.Linear(256 * 6 * 6, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),  # 当dropout=0.3时过拟合较严重

    torch.nn.Linear(1024, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),

    torch.nn.Linear(128, 2)
)

print(model)
if args.cuda:
    model.cuda()


def get_optimizer(lr):
    return optim.Adam(model.parameters(), lr=lr)


optimizer = get_optimizer(args.lr)
# scheduler = lr_scheduler.StepLR(step_size=10, optimizer=optimizer, gamma=0.5)


# count the amount of parameters in the network
def measure_size():
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
        # batch_index批量下标0, 1, 2, 3,...30
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 梯度归零
        output = model(data)
        # output = F.softmax(output, dim=1)  # Alexnet 网络输出做softmax,100x2,行和为1
        # output_loss = torch.log(output)
        output = F.log_softmax(output, dim=1)
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

    print("Train set average loss = %.6f" % (total_loss / len(train_loader.dataset)))
    dev_loss = 0.
    dev_correct = 0
    model.eval()
    f = open('dev-eer', 'w')
    f2 = open('dev-loss-acc', 'a')
    for data, devtarget in devdata_loader:
        # data = data.unsqueeze_(1)  # 在第1维（下标从0开始）增加一个维度
        with torch.no_grad():
            data, target = Variable(data), Variable(devtarget)
        output = model(data)
        # output = F.softmax(output, dim=1)    # 做softmax
        # output_loss = torch.log(output)      # 再取对数
        output = F.log_softmax(output, dim=1)
        output_arr = output.data.numpy().reshape((-1, 2))  # tensor to array, reshape(-1, 2)转换为两列
        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = output_arr[i, 0] - output_arr[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))
        dev_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        dev_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    f.close()
    dev_loss /= len(devdata_loader.dataset)
    print("Dev set: Average loss is %.6f" % dev_loss)
    print('Dev set: Accuracy: {}/{} ({:.0f}%)\n'.format(dev_correct, len(devdata_loader.dataset),
                                                        100. * dev_correct / len(devdata_loader.dataset)))
    f2.write('%.6f %.6f %d/%d(%.0f)%%\n' % (total_loss/ len(train_loader.dataset), dev_loss, dev_correct, len(devdata_loader.dataset),
                                       100. * dev_correct / len(devdata_loader.dataset)))
    f2.close()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    f = open('eer-file', 'w')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data = data.unsqueeze_(1)
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # output = F.softmax(output, dim=1)
        # output_loss = torch.log(output)
        output = F.log_softmax(output, dim=1)
        output_arr = output.data.numpy().reshape((-1, 2))
        tgt = target.data.numpy()
        for i in range(tgt.shape[0]):
            itgt = int(tgt[i])  # int_target
            scr = output_arr[i, 0] - output_arr[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    f.close()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


########################################################################################################################
# # the training procedure
initial_size = measure_size()
print('initial_size: {}'.format(initial_size))

if args.train:
    for epoch in range(1, args.epochs+1):
        train(epoch)
        # scheduler.step()
        torch.save(model, model_output_dir + 'iter' + str(epoch) + '.mdl')
        shutil.copy('dev-eer', model_output_dir + '/dev-eer-' + str(epoch))

    torch.save(model, model_output_dir + 'alexnet_1_new.mdl')

if args.eval:
    test()

end_time = time.time()
process_time = end_time-start_time
print('End time: {}'.format(end_time))
print('Time cost: {}'.format(process_time))

print('EER of dev:')
cm_eer_dev = asv17_util.compute_eer17_from_file('dev-eer')
print('EER of eval:')
cm_eer_eval = asv17_util.compute_eer17_from_file('eer-file')

#
# model = torch.load('iter18.mdl')
# test()
# end_time = time.time()
# process_time = end_time-start_time
# print('Time cost: {}'.format(process_time))
# print('EER of eval:')
# cm_eer_eval = asv17_util.compute_eer17_from_file('eer-file')




