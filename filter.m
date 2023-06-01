fs=256;
f1=0.5; %cuttoff low frequency to get rid of baseline wander
f2=35; %cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/fs; % cutt off based on fs
N = 3; % order of 3 less processing
[a,b] = butter(N,Wn);%bandpass filtering

% a1 = [0.0381         0   -0.1144         0    0.1144         0   -0.0381]
% b1 = [1.0000   -4.3182    7.7749   -7.5799    4.2804   -1.3295    0.1723]
% fvtool(a1,b1)
ecg_h = filtfilt(a,b,ecg);