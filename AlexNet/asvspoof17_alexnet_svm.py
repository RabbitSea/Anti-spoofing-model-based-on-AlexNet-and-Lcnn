from __future__ import print_function

import argparse  # 命令行解析模块
import os
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from sklearn import svm
from sklearn.preprocessing import scale
from torch.autograd import Variable

'''仅在训练集上训练的alexnet_1.mdl,用svm做分类'''

from AlexNet.asvspoof17_My_Dataset import ASVspoofTestData, ASVspoofTrainData, ASVspoofDevData
from ASVspoof2017_util import asv17_util

warnings.filterwarnings('ignore')
# Training settings
parser = argparse.ArgumentParser(description='ConvNet reduction')  # 创建命令行解析的对象
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output-dir', type=str, default='./', metavar='N',  # 设置输出的目录
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

print('Parameters: ', args.__dict__)  # 输出args里面的所有参数

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
model = torch.load('AlexNet1_iter18.mdl')
for parma in model.parameters():
    parma.requires_grad = False

# 6*6*256
model.classifier = torch.nn.Sequential(
    # torch.nn.ReLU(),
    # torch.nn.Linear(256 * 6 * 6, 1000),
)

print(model)
if args.cuda:
    model.cuda()


########################################################################################################################
#

def train2():  # 输入数据做标准化
    model.train()
    Input_svm = []
    Input_target = []
    for batch_idx, (data, target) in enumerate(train_loader):  # 迭代到train_loader里面取数据,随机取
        # batch_index批量下标0, 1, 2, 3,...30
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output_x = model(data)  # size:batch_size*9216,tensor
        output_x = np.array(output_x)  # 转成array,size:batch_size*9216

        for i in range(len(data)):
            output_x[i] = scale(output_x[i], axis=0)  # 数据标准化
            Input_svm.append(output_x[i])
            Input_target.append(target[i])

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(Input_svm, Input_target)

    now_time_train = time.time()
    train_time = now_time_train - start_time
    print("spend {}s for training".format(train_time))

    f = open('dev-eer-svm', 'w')
    for data, devtarget in devdata_loader:
        # data = data.unsqueeze_(1)  # 在第1维（下标从0开始）增加一个维度
        with torch.no_grad():
            data, target = Variable(data), Variable(devtarget)
        output_x = model(data)
        output_x = np.array(output_x)  # 转成array,size:batch_size*9216

        for i in range(len(data)):
            output_x[i] = scale(output_x[i], axis=0)  # 数据标准化

        # dev_pre_class = clf.predict(output_x)  # 输出分类
        dev_pre = clf.predict_log_proba(output_x)  # 输出类型为 ndarray
        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = dev_pre[i, 0] - dev_pre[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))
    f.close()
    now_time_dev = time.time()
    dev_time = now_time_dev - now_time_train
    print("spend {}s for dev test".format(dev_time))

    model.eval()
    Input_svm = []
    f2 = open('eer-file-svm', 'w')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output_x = model(data)
        output_x = np.array(output_x)  # 转成array,size:batch_size*9216

        for i in range(len(data)):
            output_x[i] = scale(output_x[i], axis=0)  # 数据标准化

        # test_pre_class = clf.predict(output_x)  # 输出分类
        test_pre = clf.predict_log_proba(output_x)  # 输出类型为 ndarray
        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = test_pre[i, 0] - test_pre[i, 1]
            if itgt == 0:
                f2.write('%f %s\n' % (scr, 'target'))
            else:
                f2.write('%f %s\n' % (scr, 'nontarget'))
    f2.close()
    now_time_test = time.time()
    test_time = now_time_test - now_time_dev
    print("spend {}s for eval test".format(test_time))


def train():  # 输入数据不做标准化
    model.train()
    Input_svm = []
    Input_target = []
    for batch_idx, (data, target) in enumerate(train_loader):  # 迭代到train_loader里面取数据,随机取
        # batch_index批量下标0, 1, 2, 3,...30
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output_x = model(data)  # size:batch_size*9216,tensor
        output_x = np.array(output_x)  # 转成array,size:batch_size*9216

        for i in range(len(data)):
            Input_svm.append(output_x[i])
            Input_target.append(target[i])

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(Input_svm, Input_target)

    now_time_train = time.time()
    train_time = now_time_train - start_time
    print("spend {}s for train set".format(train_time))

    f = open('dev-eer-svm', 'w')
    Input_svm = []
    dev_correct = 0
    for data, devtarget in devdata_loader:
        # data = data.unsqueeze_(1)  # 在第1维（下标从0开始）增加一个维度
        with torch.no_grad():
            data, target = Variable(data), Variable(devtarget)
        output_x = model(data)

        dev_pre_class = clf.predict(output_x)  # 输出分类
        dev_pre = clf.predict_log_proba(output_x)  # 输出类型为 ndarray
        dev_preb = torch.Tensor(dev_pre_class)

        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = dev_pre[i, 0] - dev_pre[i, 1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'target'))
            else:
                f.write('%f %s\n' % (scr, 'nontarget'))
        dev_correct += dev_preb.eq(target.data.view_as(dev_preb)).cpu().sum()  # target.data的类型为tensor
    f.close()
    print("Dev set: Acc is %.6f" % dev_correct)
    now_time_dev = time.time()
    dev_time = now_time_dev - now_time_train
    print("spend {}s for dev set".format(dev_time))

    model.eval()
    test_correct = 0
    f2 = open('eer-file-svm', 'w')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output_x = model(data)
        test_pre_class = clf.predict(output_x)  # 输出分类
        test_pre = clf.predict_log_proba(output_x)  # 输出类型为 ndarray
        test_preb = torch.Tensor(test_pre_class)
        tgt = target.data.numpy()  # tensor to array
        for i in range(tgt.shape[0]):  # shape[0] = number of the rows = 100
            itgt = int(tgt[i])
            scr = test_pre[i, 0] - test_pre[i, 1]
            if itgt == 0:
                f2.write('%f %s\n' % (scr, 'target'))
            else:
                f2.write('%f %s\n' % (scr, 'nontarget'))
        test_correct += test_preb.eq(target.data.view_as(test_preb)).cpu().sum()  # target.data的类型为tensor
    f2.close()
    print("Test set: Acc is %.6f" % test_correct)
    now_time_test = time.time()
    test_time = now_time_test - now_time_dev
    print("spend {}s for eval test".format(test_time))


########################################################################################################################
# the training procedure


if args.train:
    train2()

# if args.eval:
#     test()

end_time = time.time()
process_time = end_time - start_time
print('End time: {}'.format(end_time))
print('Time cost: {}'.format(process_time))

print('EER of dev:')
cm_eer_dev = asv17_util.compute_eer17_from_file('dev-eer-svm')
print('EER of eval:')
cm_eer_eval = asv17_util.compute_eer17_from_file('eer-file-svm')

# model = torch.load('final.mdl')
# test()
# print('EER of eval:')
# cm_eer_eval = asv17_util.compute_eer17_from_file('eer-file')
