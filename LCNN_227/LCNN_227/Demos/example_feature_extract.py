from LCNN_227 import asv2017_feature_extract as fe
import soundfile
import librosa


filepath = '../../../ASVspoof2017_small_dataset/wav/train/'
filename1 = 'T_1000001.wav'
filename2 = 'T_1000072.wav'

y1, sr1 = soundfile.read(filepath+filename1)
y2, sr2 = soundfile.read(filepath+filename2)

cqt1 = fe.extract_cqt(y1, sr1)
cqt2 = fe.extract_cqt(y2, sr2)

mfcc1 = fe.extract_mfcc(y1, sr1)
mfcc2 = fe.extract_mfcc(y2, sr2)

stft1 = fe.extract_fft(y1, sr1)
stft2 = fe.extract_fft(y2, sr2)


print("Shape of CQT1: ", cqt1.shape)
print("Shape of CQT2: ", cqt2.shape)
print("Shape of MFCC1：", mfcc1.shape)
print("Shape of MFCC2：", mfcc2.shape)
print("Shape of STFT1：", stft1.shape)
print("Shape of STFT2：", stft2.shape)
# print(stft2)



