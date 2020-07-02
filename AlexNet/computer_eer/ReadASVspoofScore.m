function [score, key] = ReadASVspoofScore(scorefile)
fileID = fopen(scorefile);
protocol = textscan(fileID, '%.6f%s');
fclose(fileID);

% (score[i], key[i])
score         = protocol{1};
key           = protocol{2};


end