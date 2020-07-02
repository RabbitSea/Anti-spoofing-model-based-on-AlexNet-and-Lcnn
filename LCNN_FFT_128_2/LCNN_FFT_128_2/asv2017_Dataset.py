import math
import os

import h5py
import numpy as np
import torch
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ASVspoof2017_util import asv17_util

# 处理自己的数据集, 继承Dataset类，重写_getitem_()

label_encoder = {'genuine': 0, 'spoof': 1}
feature_type = 'CQT'  # fft  cqt  cqcc
feat_dim = 128
time_dim = 400

'''1 读取protocol'''
# protocol_path = '../ASVspoof2017_small_dataset_2/protocol/'
# feat_path = '../ASVspoof2017_small_dataset_2/feature_cnn_128/'
# wave_path_pattern = '../ASVspoof2017_small_dataset_2/wav/{}'
# train_protocol_file = '../ASVspoof2017_small_dataset_2/protocol/ASVspoof2017_V2_train.trn.txt'
# dev_protocol_file = '../ASVspoof2017_small_dataset_2/protocol/ASVspoof2017_V2_dev.trl.txt'
# eval_protocol_file = '../ASVspoof2017_small_dataset_2/protocol/ASVspoof2017_V2_eval.trl.txt'

# protocol_path = '../../ASVspoof2017_small_dataset/protocol/'
# feat_path = '../../ASVspoof2017_small_dataset/feature_cnn_128/'
# wave_path_pattern = '../../ASVspoof2017_small_dataset/wav/{}'
# train_protocol_file = '../../ASVspoof2017_small_dataset/protocol/ASVspoof2017_V2_train.trn.txt'
# dev_protocol_file = '../../ASVspoof2017_small_dataset/protocol/ASVspoof2017_V2_dev.trl.txt'
# eval_protocol_file = '../../ASVspoof2017_small_dataset/protocol/ASVspoof2017_V2_eval.trl.txt'

# protocol_path = '../../ASVspoof2017/protocol/'
# feat_path = '../../ASVspoof2017_feature/feature_cnn_128/'
# wave_path_pattern = '../../ASVspoof2017/wav/{}'
# train_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_train.trn.txt'
# dev_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_dev.trl.txt'
# eval_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_eval.trl.txt'


protocol_path = '../../ASVspoof2017/protocol/'
feat_path = '../../ASVspoof2017_feature/feature_cqt_128/'
wave_path_pattern = '../../ASVspoof2017/wav/{}'
train_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_train.trn.txt'
dev_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_dev.trl.txt'
eval_protocol_file = '../../ASVspoof2017/protocol/ASVspoof2017_V2_eval.trl.txt'


train_feat_file = os.path.join(feat_path, 'spoof2017_{}_train_featureCell.h5'.format(feature_type))  # feature文件名
dev_feat_file = os.path.join(feat_path, 'spoof2017_{}_dev_featureCell.h5'.format(feature_type))
eval_feat_file = os.path.join(feat_path, 'spoof2017_{}_eval_featureCell.h5'.format(feature_type))


class ASVspoofTrainData(Dataset):
    def __init__(self, is_train=True):
        train_file, train_type, train_speaker, train_phrase, train_environment, train_playback, train_recording = asv17_util.read_protocol(
            train_protocol_file)
        self.train_file = train_file
        # self.feats = load_feat_cnn(train_feat_file, train_file, 768, 400)
        self.num_obj = len(train_file)
        self.classes = torch.from_numpy(np.array([0 if not train_type[ind] == "spoof" else 1 for ind in range(self.num_obj)])).long()
        self.transformations = transforms.Compose([transforms.ToTensor()])

    # 数据集长度
    def __len__(self):
        return self.num_obj

    # 获取单条数据
    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        f = h5py.File(train_feat_file, 'r')
        wav_name = self.train_file[idx]
        # print('Train: index: {}， wav_name: {}'.format(idx, wav_name))
        # wav_name = wav_name.replace('.wav', '')
        feature = f[wav_name][:]
        feature = precess_feat_cnn(feature, nFeatures=feat_dim, nTime=time_dim)

        f.close()

        # feat =torch.tensor(feature, dtype=torch.float32)
        feat = self.transformations(feature)

        label = int(self.classes[idx])
        return feat, label


class ASVspoofDevData(Dataset):

    def __init__(self):
        dev_file, dev_type, dev_speaker, dev_phrase, dev_environment, dev_playback, dev_recording = asv17_util.read_protocol(
            dev_protocol_file)

        self.dev_file = dev_file
        self.num_obj = len(dev_file)
        self.classes = torch.from_numpy(np.array([0 if not dev_type[ind] == "spoof" else 1 for ind in range(self.num_obj)])).long()
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        f = h5py.File(dev_feat_file, 'r')
        wav_name = self.dev_file[idx]
        # print('dev: index: {}， wav_name: {}'.format(idx, wav_name))
        # wav_name = wav_name.replace('.wav', '')
        feature = f[wav_name][:]
        feature = precess_feat_cnn(feature, nFeatures=feat_dim, nTime=time_dim)

        f.close()

        # feat =torch.tensor(feature, dtype=torch.float32)
        feat = self.transformations(feature)
        label = int(self.classes[idx])

        return feat, label


class ASVspoofTestData(Dataset):
    def __init__(self):
        eval_file, eval_type, eval_speaker, eval_phrase, eval_environment, eval_playback, eval_recording = asv17_util.read_protocol(
            eval_protocol_file)

        self.eval_file = eval_file
        self.num_obj = len(eval_file)
        self.classes = torch.from_numpy(
            np.array([0 if not eval_type[ind] == "spoof" else 1 for ind in range(self.num_obj)])).long()
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()
        # print('Eval: index: ' + str(idx))
        f = h5py.File(eval_feat_file, 'r')
        wav_name = self.eval_file[idx]
        feature = f[wav_name][:]
        feature = precess_feat_cnn(feature, nFeatures=feat_dim, nTime=time_dim)

        f.close()

        feat = self.transformations(feature)
        label = int(self.classes[idx])
        return feat, label


'''处理特征输入到CNN'''


def precess_feat_cnn(feat_in, nFeatures, nTime):
    feat = feat_in

    if feat.shape[1] < nTime / 2:
        rr = int(math.ceil(float(nTime) / feat.shape[1]))
        tttt = np.tile(feat, rr)
        feat = tttt[:, 0:nTime]

    if feat.shape[1] < nTime:
        temp = feat.reshape(nFeatures, -1)
        temo = feat.reshape(nFeatures, -1)[:, 0:nTime - feat.shape[1]]
        feat = np.hstack([temp, temo])

    if feat.shape[1] > nTime:
        feat = feat[:, 0:nTime]

    # feat = feat - np.mean(feat, axis=1, keepdims=True)
    # feat = np.divide(feat, np.std(feat, axis=1, keepdims=True))
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min

    '''数据标准化'''

    feat = scale(feat, axis=1)  # 每行进行标准化, 按帧归一化
    # feat = scale(feat, axis=0)  # 每列进行标准化
    return feat


if __name__ == '__main__':
    d = ASVspoofDevData()
    dl = DataLoader(d, batch_size=100, shuffle=True)
    batch = 0
    for x, l in dl:
        batch += 1
        print("New batch:", batch)

