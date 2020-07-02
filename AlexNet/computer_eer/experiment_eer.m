%% computer eer from 'LCNN_FFT/computer_eer'
addpath(genpath('bosaris_toolkit'));
addpath(genpath('computer_eer'));
eer1 = computeEERasvspoof2017('dev-eer');

%% computer eer from 'ASVspoof2017_computer_eer'
fileID = fopen('dev-eer');
protocol = textscan(fileID, '%.6f%s');
fclose(fileID);

% (score[i], key[i])
scores         = protocol{1};
keys           = protocol{2};

addpath(genpath('ASVspoof2017_computer_eer'));
humanscores = scores(strcmp(keys, 'genuine'));
spoofscores = scores(strcmp(keys, 'spoof'));

[eer2, eer_t] = compute_eer(humanscores, spoofscores)