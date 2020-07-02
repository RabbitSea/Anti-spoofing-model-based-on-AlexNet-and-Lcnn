import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # 可方便指定各种transformer，直接传入DataLoader

#
# allfile_root = '../ASVspoof2017_small_dataset/Spectrum/'
# train_root = allfile_root + 'train_features/'
# train_record_path = allfile_root + 'protocol/train_jpg.txt'
# dev_root = allfile_root + 'dev_features/'
# dev_record_path = allfile_root + 'protocol/dev_jpg.txt'
# eval_root =  allfile_root + 'eval_features/'
# eval_record_path = allfile_root + 'protocol/eval_jpg.txt'


# 全局归一化数据库
# allfile_root = '../ASVspoof2017/'
# train_root = '../ASVspoof2017_feature/picture_227/train_features/'
# train_record_path = allfile_root + 'train_jpg.txt'
# dev_root = '../ASVspoof2017_feature/picture_227/dev_features/'
# dev_record_path = allfile_root + 'dev_jpg.txt'
# eval_root =  '../ASVspoof2017_feature/picture_227/eval_features/'
# eval_record_path = allfile_root + 'eval_jpg.txt'

# 按帧归一化数据库
allfile_root = '../ASVspoof2017/'
train_root = 'train_features/'
train_record_path = allfile_root + 'train_jpg.txt'
dev_root = 'dev_features/'
dev_record_path = allfile_root + 'dev_jpg.txt'
eval_root = 'eval_features/'
eval_record_path = allfile_root + 'eval_jpg.txt'

# 按行一化数据库
# allfile_root = '../ASVspoof2017/'
# train_root = '../ASVspoof2017_feature/picture_227_row/train_features/'
# train_record_path = allfile_root + 'train_jpg.txt'
# dev_root = '../ASVspoof2017_feature/picture_227_row/dev_features/'
# dev_record_path = allfile_root + 'dev_jpg.txt'
# eval_root = '../ASVspoof2017_feature/picture_227_row/eval_features/'
# eval_record_path = allfile_root + 'eval_jpg.txt'

# allfile_root = 'D:/b 所有课程/1-Speech_Recognition_code/ASVspoof2017/'
# train_root = 'D:/b 所有课程/1-Speech_Recognition_code/LCNN_FFT_picture/train_features/'
# train_record_path = allfile_root + 'train_jpg.txt'
# dev_root = 'D:/b 所有课程/1-Speech_Recognition_code/LCNN_FFT_picture/dev_features/'
# dev_record_path = allfile_root + 'dev_jpg.txt'
# eval_root = 'D:/b 所有课程/1-Speech_Recognition_code/LCNN_FFT_picture/eval_features/'
# eval_record_path = allfile_root + 'eval_jpg.txt'


class ASVspoofTrainData(Dataset):
    def __init__(self, is_train=True):
        # record_path:记录图片路径及对应label的文件
        self.root = train_root
        self.record_path = train_record_path
        self.is_train = is_train
        self.pic_file = []
        self.labels = []
        with open(self.record_path) as fp:
            for line in fp.readlines():
                line = line.strip()
                listFromLine = line.split(' ')
                # listFromLine[0]:某图片的图片名字，listFromLine[1]:该图片对应的label
                self.pic_file.append(listFromLine[0])  # ‘name’
                self.labels.append(int(listFromLine[1]))
        # 定义transform，将数据封装为Tensor
        self.classes = torch.from_numpy(np.array(self.labels)).long()
        self.num_obj = len(self.classes)
        self.transformations = transforms.Compose([transforms.ToTensor()])  # 没有做mean和normalization, 在傅立叶变换时就做了

    # 数据集长度
    def __len__(self):
        return self.num_obj

    # 获取单条数据
    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        # print('Train: index: ' + str(idx))
        img = self.transformations(Image.open(self.root + self.pic_file[idx]))
        # print(img.shape)
        label = int(self.labels[idx])
        return img, label


class ASVspoofDevData(Dataset):

    def __init__(self):
        self.root = dev_root
        self.record_path = dev_record_path
        self.pic_file = []
        self.labels = []
        with open(self.record_path) as fp:
            for line in fp.readlines():
                line = line.strip()
                listFromLine = line.split(' ')
                # listFromLine[0]:某图片的图片名字，listFromLine[1]:该图片对应的label
                self.pic_file.append(listFromLine[0])  # ‘name’
                self.labels.append(int(listFromLine[1]))
        # 定义transform，将数据封装为Tensor
        self.classes = torch.from_numpy(np.array(self.labels)).long()
        self.num_obj = len(self.classes)
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()
        # print('DEv: index: ' + str(idx))
        img = self.transformations(Image.open(self.root + self.pic_file[idx]))
        label = int(self.labels[idx])
        return img, label


class ASVspoofTestData(Dataset):
    def __init__(self):
        self.root = eval_root
        self.record_path = eval_record_path
        self.pic_file = []
        self.labels = []
        with open(self.record_path) as fp:
            for line in fp.readlines():
                line = line.strip()
                listFromLine = line.split(' ')
                # listFromLine[0]:某图片的图片名字，listFromLine[1]:该图片对应的label
                self.pic_file.append(listFromLine[0])  # ‘name’
                self.labels.append(int(listFromLine[1]))
        # 定义transform，将数据封装为Tensor
        self.classes = torch.from_numpy(np.array(self.labels)).long()
        self.num_obj = len(self.classes)
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()
        img = self.transformations(Image.open(self.root + self.pic_file[idx]))
        label = int(self.labels[idx])
        return img, label


if __name__ == '__main__':  # 文件作为脚本直接执行，而 import 到其他脚本中是不会被执行的。
    d = ASVspoofTestData()
    dl = DataLoader(d, batch_size=100, shuffle=True)
    for x, l in dl:  # x是由图片转成的tensor, batch x 3 x 227 x 227, l是torch.Size([100])
        print("New batch")
        # 打印该批次的标签和图片
        img = torchvision.utils.make_grid(x)
        img = img.numpy().transpose([1, 2, 0])  # 图片打印的位置
        plt.imshow(img)
        plt.show()
