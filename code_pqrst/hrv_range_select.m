function [sample_L, sample_R] = hrv_range_select(hrv)
if hrv<108
    sample_L = 40;
    sample_R = 80;
elseif (108<=hrv) && (hrv<132)
    sample_L = 48;
    sample_R = 96;
elseif (132<=hrv) && (hrv<156)
    sample_L = 56;
    sample_R = 112;
elseif (156<=hrv) && (hrv<180)
    sample_L = 64;
    sample_R = 128;
elseif (180<=hrv) && (hrv<204)
    sample_L = 72;
    sample_R = 144;
elseif (204<=hrv) 
    sample_L = 80;
    sample_R = 160;
% elseif (228<=hrv)
%     sample_L = 80;
%     sample_R = 160;
end