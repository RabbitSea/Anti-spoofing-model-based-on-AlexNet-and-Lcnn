addpath(genpath('ASVspoof2017'));
file_path = 'ASVspoof2017/wav/train/T_1000001.wav';
% 以ASVspoof2017/wav/train/T_1000001.wav为例, 生成128*400的对数能量谱
[X, fs] = audioread(file_path);

[filepath, name, ext] = fileparts(file_path);

% CONSTANTS ===============================================================
window_length = 16/1000; % 16 ms window
D = window_length*fs;
L = 454;
filter_length = L;
SP = 0.5; % Overlap factor
%==========================================================================

sg = buffer(X,D,ceil(SP*D),'nodelay').';
sg_filt = filter([1 -0.97],1,sg.').';

% Windowing and FFT
no_frames = size(sg_filt,1);
window = repmat(hamming(D).',no_frames,1);

segFFT = fft((sg_filt.*window),filter_length,2);
segFFT = segFFT(:,1:filter_length/2);

abssegFFT = max(abs(segFFT).^2,eps);
logsegFFT = log((abssegFFT));

feats = [logsegFFT];
rep = size(feats,1); % 行数
feats = feats - repmat(mean(feats),rep,1); % feats = feats - mean(feats);
feats = feats./repmat(std(feats),rep,1);

if size(feats,1) > 227
    feats = feats(1:227,:);
elseif size(feats,1) < 227/2
    temp = repmat(feats, ceil(227/size(feats,1)), 1);
    feats = temp(1:227,:);
else
    feats = repmat(feats, 2, 1);
    feats = feats(1:227,:);
end

normalizedImage = uint8(255*mat2gray(feats));
xfeats(:,:,1) = normalizedImage;
xfeats(:,:,2) = normalizedImage;
xfeats(:,:,3) = normalizedImage;


[status, msg, msgID] = mkdir('fft_test_feature');
imwrite(xfeats,strcat('fft_test_feature/',name,'.jpg'));
   