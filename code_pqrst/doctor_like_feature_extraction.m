clear;
nb=[100,101,103,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234];
% nb=[100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234];
% nb = [100,101,103,106,231,233,222,214,122,207];
% nb = [102,104,108,200,202,203,217];
% nb = [207];
nb = [100];
PATH = '/Users/xuxinzi/Documents/research/ECG_2020/project/MITBIHdata'; % need to change according to your data path
plot_index=2538;
nocha=2; %通道数
original_samplerate=360;
present_samplerate=256;


for i=1:length(nb)
% for i = 1
filenum=nb(i)

%读数据并降采样
[ecg_data,ANNOT,ATRTIME,RRtime] = read_data_5classes(filenum,PATH);
sampletime=(1:length(ecg_data))/original_samplerate;
resampletime=1/present_samplerate:1/present_samplerate:(length(ecg_data)/original_samplerate);
ecg_data_resample=interp1(sampletime,ecg_data,resampletime);
rpeak_time = double(int32(ATRTIME*present_samplerate/original_samplerate));

if filenum ~= 114
    [qrs_amp_raw,qrs_i_raw,delay,ecg_h,ecg_m,locs,SIG_buf,SIG1_buf,pole_buf]=pan_tompkin(ecg_data_resample(1:end,1),present_samplerate,0);
else
    [qrs_amp_raw,qrs_i_raw,delay,ecg_h,ecg_m,locs,SIG_buf,SIG1_buf,pole_buf]=pan_tompkin(ecg_data_resample(1:end,2),present_samplerate,0);
end


ANNOT_qrs= zhaolabel(qrs_i_raw,rpeak_time+6,ANNOT);

[q_on,s_off]=qrs_detect_min(ecg_h,qrs_i_raw',pole_buf,0.25,0.2);

[t_pre,p_cur,t_cur,p_post]=pt_detect_v2(ecg_h,q_on,s_off,qrs_i_raw',pole_buf);

% uncomment to watch the signals
% if filenum == 114
%     plot_and_save2(ecg_data_resample(:,2), filenum, rpeak_time,ANNOT,ecg_h,qrs_i_raw(2:end-2)',q_on(2:end-2),s_off(2:end-2),p_cur(1,:),t_cur(1,:),'wave_do',1000);
% else
%     plot_and_save2(ecg_data_resample(:,1), filenum, rpeak_time,ANNOT,ecg_h,qrs_i_raw(2:end-2)',q_on(2:end-2),s_off(2:end-2),p_cur(1,:),t_cur(1,:),'wave_do',1000);
% end

 
[ecg_seg,annot,feature_matrix,annot_pre] = save_features_v2(ecg_h,ANNOT_qrs,qrs_i_raw',pole_buf,q_on,s_off,t_pre,p_cur,t_cur,p_post,filenum);


end

