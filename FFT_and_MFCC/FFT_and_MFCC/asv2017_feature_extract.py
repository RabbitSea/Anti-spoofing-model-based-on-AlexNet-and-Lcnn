import os

import h5py
import librosa
import numpy as np
import pywt
import torch
from ASVspoof2017_util import asv17_util
from scipy import signal
from tqdm import tqdm

''' parameters'''
max_prob_value = 0.99
min_prob_value = 0.0000001

max_log_prob_value = 0
min_log_prob_value = -100000

sample_rate = 16000
# n_fft = int(25 * sample_rate / 1000)
# hop_length = 512
n_fft = 254  # mfcc 和 stft一起的时候，n-fft 和 hop_length应该一致
hop_length = 128

n_imfcc = 13
n_mfcc = 13
n_cqt = 19  # 13
# n_fft = 864 * 2
f_max = sample_rate / 2
f_min = sample_rate / (2 ** 10)


def use_gpu():
    # return False
    return torch.cuda.is_available()


def extract_feat(wav_path, feature_type):
    audio, sr = librosa.load(wav_path, sr=16000)

    if feature_type == "IMFCC":
        return extract_imfcc(audio, sr)
    elif feature_type == "MFCC":
        return extract_mfcc(audio, sr)
    elif feature_type == "MFCC2":
        return extract_sli_mfcc(audio, sr)
    elif feature_type == "CQT":
        return extract_cqt(audio, sr)
    elif feature_type == "LCQT":
        return extract_log_cqt(audio, sr)
    elif feature_type == "SPEC":
        return extract_spect(audio, sr)
    elif feature_type == "DB4":
        return extract_db4(audio, sr)
    elif feature_type == "DB8":
        return extract_db8(audio, sr)
    elif feature_type == "FFT":
        return extract_fft(audio, sr)
    elif feature_type == "RAW":
        return extract_raw(audio, sr)
    else:
        raise ValueError('feature_type error:' + feature_type)


def trim_silence(audio, threshold=0.1, frame_length=254):  # 2048，过滤scilence,将它剪掉
    if audio.size < frame_length:
        frame_length = audio.size
    # energy = librosa.feature.rmse(audio, frame_length=frame_length)
    energy = librosa.feature.rms(audio, frame_length=frame_length)  # 按帧计算能量
    frames = np.nonzero(energy > threshold)  # 返回energy中大于threshold的下标
    indices = librosa.core.frames_to_samples(frames)[1]  # 帧下标转换为样本下标，若hop_length=512,那么将[0, 1, 2]转换成[0, 512, 1024]
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]  # 返回能量大的样本信号


# def save_silence(audio, frame_length=8192):  # 2048，保留scilence,512ms
#     m_hop_length = 1024
#     if audio.size < frame_length:
#         frame_length = audio.size
#     n_frame = int((audio.size-(frame_length-m_hop_length))/m_hop_length)  # 向下取整，不希望填充
#     audio = audio[0:(frame_length + (n_frame-1) * m_hop_length)]  # 裁剪
#     # print('n_frame.shape{}'.format(n_frame))
#     # print('audio.shape{}'.format(audio.shape))
#
#     energy = librosa.feature.rms(audio, frame_length=frame_length, hop_length=m_hop_length, pad_mode='wrap')  # 按帧计算能量，类型为ndarray
#     frame_min_dx = np.argmin(energy)  # 返回第一个最小值的位置
#     indice1 = librosa.core.frames_to_samples(frame_min_dx)  # 帧下标转换为样本下标，若hop_length=512,那么将[0, 1, 2]转换成[0, 512, 1024]
#     energy[0, frame_min_dx] += 0.5  # 修改最小值
#     audio_min1 = audio[indice1:(indice1+frame_length)]
#
#     frame_min_dx = np.argmin(energy)  # 返回第二个最小值的位置
#     indice2 = librosa.core.frames_to_samples(frame_min_dx)
#     audio_min2 = audio[indice2:(indice2+frame_length)]
#     A_min = np.hstack((audio_min1, audio_min2))
#
#     return A_min  # 返回能量低的样本信号5


def save_silence(audio, frame_length=8192):  # 2048，保留scilence,512ms
        m_hop_length = 2048
        if audio.size < frame_length:
            frame_length = audio.size

        energy = librosa.feature.rms(audio, frame_length=frame_length, hop_length=m_hop_length,
                                     pad_mode='wrap')  # 按帧计算能量，类型为ndarray
        frame_min_dx = np.argmin(energy)  # 返回第一个最小值的位置
        indice = librosa.core.frames_to_samples(frame_min_dx)  # 帧下标转换为样本下标，若hop_length=512,那么将[0, 1, 2]转换成[0, 512, 1024]
        energy[0, frame_min_dx] += 0.5  # 修改最小值

        frame_min_dx = np.argmin(energy)  # 返回第二个最小值的位置
        indice = librosa.core.frames_to_samples(frame_min_dx)
        return audio[indice:(indice+frame_length)]  # 返回能量低的样本信号


def extract_imfcc(audio, sr):
    S = np.abs(librosa.core.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2.0
    mel_basis = librosa.filters.mel(sr, n_fft)
    mel_basis = np.linalg.pinv(mel_basis).T
    mel = np.dot(mel_basis, S)
    S = librosa.power_to_db(mel)
    imfcc = np.dot(librosa.filters.dct(n_imfcc, S.shape[0]), S)
    imfcc_delta = librosa.feature.delta(imfcc)
    imfcc_delta_delta = librosa.feature.delta(imfcc)
    feature = np.concatenate((imfcc, imfcc_delta, imfcc_delta_delta), axis=0)
    return feature


def extract_mfcc(audio, sr):  # 提取整段mfcc
    y = preemphasis(audio)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
    return feature


def extract_sli_mfcc(audio, sr):  # 提取沉默信号mfcc
    y = save_silence(audio)
    if y.size == 0:
        y = audio
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')  # mode表示填充模式
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
    return feature


def extract_cqt(audio, sr):
    y = trim_silence(audio)
    if y.size == 0:
        y = audio
    cqt = librosa.feature.chroma_cqt(y, sr, hop_length=hop_length, fmin=f_min, n_chroma=n_cqt, n_octaves=5)
    return cqt


def extract_log_cqt(audio, sr):
    cqt = librosa.feature.chroma_cqt(audio, sr, hop_length=hop_length, fmin=f_min, n_chroma=84, n_octaves=7)
    return librosa.amplitude_to_db(cqt)


def extract_spect(audio, sr):
    # audio = trim_silence(audio, 0.01)
    S, _ = librosa.core.spectrum._spectrogram(audio, hop_length=hop_length, n_fft=n_fft, power=2)
    result = librosa.power_to_db(S)
    result = result[0:-1, :]
    return result


def extract_fft(audio, sr):
    min_level_db = -100
    num_freq = 1025
    ref_level_db = 20
    frame_length_ms = 20
    frame_shift_ms = 10

    def _normalize(S):
        return np.clip((S - min_level_db) / -min_level_db, 0, 1)

    def _stft(y):
        n_fft, hop_length, win_length = _stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)

    def _stft_parameters():
        # n_fft = (num_freq - 1) * 2
        n_fft = 254  # 1534  # 254 # 1800
        hop_length = 128   # 150  # 128  # 150
        # hop_length = int(frame_shift_ms / 1000 * sample_rate)
        # win_length = int(frame_length_ms / 1000 * sample_rate)
        win_length = 254  # 1534  # 254  # 1500
        return n_fft, hop_length, win_length

    def _amp_to_db(x):
        return 20 * np.log10(np.maximum(1e-5, x))

    # y = librosa.core.load(wav_path, sr=sample_rate)[0]
    y = audio
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


def preemphasis(x):
    p_preemphasis = 0.97
    return signal.lfilter([1, -p_preemphasis], [1], x)


def extract_db4(audio, sr):
    S, _ = librosa.core.spectrum._spectrogram(audio, hop_length=150, n_fft=1500, power=2)
    S = librosa.power_to_db(S)
    cA, cD = pywt.dwt(S, 'db4')
    return cA


def extract_db8(audio, sr):
    S, _ = librosa.core.spectrum._spectrogram(audio, hop_length=150, n_fft=1500, power=2)
    S = librosa.power_to_db(S)
    cA, cD = pywt.dwt(S, 'db8')
    return cA


def extract_raw(audio, sr):
    # y = trim_silence(audio, threshold=0.05)
    # if y.size == 0:
    #     y = audio
    return audio


def _amp_to_db(x):  # amplitude to dB
    ref_level_db = 20
    return 20 * np.log10(np.maximum(1e-5, x)) - ref_level_db


def _normalize(S):
    min_level_db = -100
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


'''1 读取protocol'''
# protocol_path = '../../ASVspoof2017/protocol/'
# feat_path = '../../ASVspoof2017_feature/new_feature_mfcc_39/'  # 全段mfcc
# wave_path_pattern = '../../ASVspoof2017/wav/{}'

# protocol_path = '../../ASVspoof2017/protocol/'
# feat_path = '../../ASVspoof2017_feature/scilen_mfcc_39/'  # 沉默版的mfcc3
# wave_path_pattern = '../../ASVspoof2017/wav/{}'

protocol_path = '../../ASVspoof2017/protocol/'
feat_path = '../../ASVspoof2017_feature/scilence_mfcc_39/'  # 沉默版的mfcc2
wave_path_pattern = '../../ASVspoof2017/wav/{}'


def extract_feat_protocol_file(feature_type):
    # print('Processing: ASVspoof2017_V2_train ......')
    # train_protocol_file = os.path.join(protocol_path, 'ASVspoof2017_V2_train.trn.txt')
    # train_file, train_type, train_speaker, train_phrase, train_environment, train_playback, train_recording = asv17_util.read_protocol(
    #     train_protocol_file)  # 读取protocol，读成列向量
    # train_feat_file = os.path.join(feat_path, 'spoof2017_{}_train_featureCell.h5'.format(feature_type))  # feature文件名
    #
    # train_all_feat = extract_file_list(feature_type, train_file, wave_path_pattern.format('train'))  # 特征提取
    # save_all_feat(train_all_feat, train_feat_file, train_file)  # 特征保存
    #
    # print('Processing: ASVspoof2017_V2_dev ......')
    # dev_protocol_file = os.path.join(protocol_path, 'ASVspoof2017_V2_dev.trl.txt')
    # dev_file, dev_type, dev_speaker, dev_phrase, dev_environment, dev_playback, dev_recording = asv17_util.read_protocol(
    #     dev_protocol_file)
    # dev_feat_file = os.path.join(feat_path, 'spoof2017_{}_dev_featureCell.h5'.format(feature_type))
    #
    # dev_all_feat = extract_file_list(feature_type, dev_file, wave_path_pattern.format('dev'))
    # save_all_feat(dev_all_feat, dev_feat_file, dev_file)

    print('Processing: ASVspoof2017_V2_eval ......')
    eval_protocol_file = os.path.join(protocol_path, 'ASVspoof2017_V2_eval.trl.txt')
    eval_file, eval_type, eval_speaker, eval_phrase, eval_environment, eval_playback, eval_recording = asv17_util.read_protocol(
        eval_protocol_file)
    eval_feat_file = os.path.join(feat_path, 'spoof2017_{}_eval_featureCell.h5'.format(feature_type))

    eval_all_feat = extract_file_list(feature_type, eval_file, wave_path_pattern.format('eval'))
    save_all_feat(eval_all_feat, eval_feat_file, eval_file)


'''2 提取特征'''


def extract_file_list(feature_type, utterance_list, wave_path):
    # 　调用特征提取函数
    all_feat = []
    for idx, utterance in enumerate(tqdm(utterance_list)):
        wave_filename = os.path.join(wave_path, utterance + '.wav')
        feat = extract_feat(wave_filename, feature_type)
        all_feat.append(feat)
    return all_feat


'''3 保存特征'''


def save_all_feat(all_feat, filename, utterance_list):
    f = h5py.File(filename, 'w')
    for ind, utterance in enumerate(utterance_list):
        # utterance = os.path.splitext(utterance)[0]
        # utterance = utterance.replace('.wav', '')  # 已经删除‘.wav’
        f[utterance] = all_feat[ind]
        # f[utterance_list[ind]] = all_feat[ind]
    f.close()


if __name__ == '__main__':
    # as_util.show_time()
    # extract_feat_protocol_file('MFCC')
    extract_feat_protocol_file('MFCC2')
    # extract_feat_protocol_file('FFT')
    #
    # as_util.show_time()
    # if (os.name == 'posix'): os.system("shutdown 0")
