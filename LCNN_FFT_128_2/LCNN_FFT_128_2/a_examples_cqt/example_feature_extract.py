import librosa.display
import matplotlib.pyplot as plt


from LCNN_FFT_128_2.asv2017_feature_extract import extract_cqt, extract_mfcc, extract_fft

# filepath = '../../ASVspoof2017_small_dataset/wav/train/'
filepath = 'D:/b 所有课程/1-Speech_Recognition_code/ASVspoof2017/wav/train/'
filename1 = 'T_1000001.wav'
filename2 = 'T_1000002.wav'

y1, sr1 = librosa.load(filepath+filename1)
y2, sr2 = librosa.load(filepath+filename2)

cqt1 = extract_cqt(y1, sr1)
cqt2 = extract_cqt(y2, sr2)

mfcc1 = extract_mfcc(y1, sr1)
mfcc2 = extract_mfcc(y2, sr2)

stft1 = extract_fft(y1, sr1)
stft2 = extract_fft(y2, sr2)


print('sr: {}'.format(sr1))
print("Shape of CQT1: ", cqt1.shape)
print("Shape of CQT2: ", cqt2.shape)
print("Shape of MFCC1：", mfcc1.shape)
print("Shape of MFCC2：", mfcc2.shape)
print("Shape of STFT1：", stft1.shape)
print("Shape of STFT2：", stft2.shape)

plt.figure()
librosa.display.specshow(cqt1, y_axis='chroma', x_axis='time')
plt.title('chroma_cqt')
plt.show()

# print(stft2)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# y1, sr1 = librosa.load(filepath+filename1)
# # plt.subplot(211)
# D = np.abs(librosa.stft(y1))**2
# S = librosa.feature.melspectrogram(S=D, sr=sr1)
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, x_axis='time',
#                          y_axis='mel', sr=sr1,
#                          fmax=8000)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()


