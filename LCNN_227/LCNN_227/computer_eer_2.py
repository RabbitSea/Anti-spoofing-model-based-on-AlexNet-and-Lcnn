from __future__ import print_function

import argparse  # 命令行解析模块
import os
import time

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import warnings

'''计算每个模型的dev-eer和test-eer'''

from LCNN_227.asv2017_Dataset import ASVspoofTestData
from ASVspoof2017_util import asv17_util

warnings.filterwarnings('ignore')
# Training settings
parser = argparse.ArgumentParser(description='ConvNet reduction')  # 创建命令行解析的对象
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

torch.manual_seed(args.seed)  # 为CPU生成种子用于生产随机数
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 如果gpu可用则创建字典


test_dataset = ASVspoofTestData()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


start_time = time.time()
print('Start time: {}'.format(start_time))


# count the amount of parameters in the network
def measure_size():
    return sum(p.numel() for p in model.parameters())


########################################################################################################################


prev_loss = 10.
curr_lr = args.lr


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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct, test_loss


########################################################################################################################
# the training procedure
# initial_size = measure_size()
# print('initial_size: {}'.format(initial_size))
#
# print('EER of dev:')
# cm_eer_dev = asv17_util.compute_eer17_from_file('dev-eer-9')
# print('EER of eval:')
# cm_eer_eval = asv17_util.compute_eer17_from_file('eer-file')
#
# f = open('a_each_eer-file.txt', 'a')
# for n in range(45, 46):
#     model_path = 'lcnn_iter{}.mdl'.format(n)
#     dev_file = 'lcnn-dev-eer-{}'.format(n)
#     model = torch.load(model_path)
#     print('\n start test')
#     correct, test_loss = test()
#     correct = int(correct)
#     correct_rate = int(100 * correct / 13306)
#     print('EER of eval:')
#     cm_eer_eval = 100 * asv17_util.compute_eer17_from_file('eer-file')
#     dev_eer = 100 * asv17_util.compute_eer17_from_file(dev_file)
#     f.write('%.6f%% %f %.6f%% %s %d/13306(%d%%) \n' % (dev_eer, test_loss, cm_eer_eval, model_path, correct, correct_rate))
# f.close()

f = open('a_each_dev_eer', 'a')
for n in range(34, 71):
    dev_file = 'lcnn-dev-eer-{}'.format(n)
    dev_eer = 100 * asv17_util.compute_eer17_from_file(dev_file)
    f.write('%f\n' % (dev_eer))
f.close()
end_time = time.time()
process_time = end_time-start_time
print('Time cost: {}'.format(process_time))


